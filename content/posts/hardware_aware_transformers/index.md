---
title: "Hardware-Aware Attention for Long Sequence Modeling"
date: 2025-03-16
math: true
categories: ["gpu", "transformers", "ml systems"]
toc: true
---

# Attention Is All You Need --- if You Can Afford the $O(N^2)$ Complexity

Attention is key to the success of modern large language models (LLMs). By attending to all tokens in the input sequence at once, attention-based Transformers overcome RNNs' difficulty in modeling long-range dependencies, avoiding vanishing and exploding gradients. However, with the power to "attend to all" comes hefty costs. 

{{< figure src="https://www.dropbox.com/scl/fi/m8vdwmpqwt40c896ty24v/Screenshot-2025-03-15-at-11.37.40-PM.png?rlkey=t6852oqzse600dc48gjg7rfal&st=r3h14cla&raw=1" caption="In vanilla attention, writing $\mathbf{S}$, $\mathbf{A}$, and $\mathbf{O}$ to memory has an $O(N^2)$ IO complexity." width="600">}}

<!--more-->

Matrix multiplications and scaling take {{< sidenote "$O(N^2d)$" >}}Here's the trick for counting matrix multiplication complexity: The outer dimensions tell us how many inner products are performed, which is $N \times N$ in self-attention. The inner dimension tells us the complexity of each inner product, which is $2d - 1$ = $d$ (element-wise products) + $(d-1)$ (additions) in this case. Row-wise softmax takes $O(N^2)$ time, which is dominated by the $O(N^2d)$ matrix multiplication time.{{< /sidenote >}} time, where $N$ is the sequence length and $d$ is the embedding dimension. To make matters worse, the fast static random-access memory ([SRAM](https://en.wikipedia.org/wiki/Static_random-access_memory)) near the GPU compute has no room to store the resulting $N \times N$ matrices --- ferrying them to the slow high-bandwidth memory ([HBM](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)) takes $O(N^2)$ time. As such, vanilla attention doesn't scale well with $N$ 💀. 

Many flavors of "efficient attention" (see Lilian Weng's [blog](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#efficient-attention) for a nice summary) aim to reduce the $O(N^2d)$ compute cost, typically via sparse (e.g., [*Sparse Transformers*](https://arxiv.org/abs/1904.10509), [*Reformer*](https://arxiv.org/abs/2001.04451), [*Routing Transformer*](https://arxiv.org/abs/2003.05997)) or low-rank (e.g., [*Linformer*](https://arxiv.org/abs/2006.04768), [*Linear Transformer*](https://arxiv.org/abs/2006.16236), [*Performer*](https://openreview.net/forum?id=Ua6zuk0WRH), [*Loki*](https://arxiv.org/abs/2406.02542)) approximations. Strangely, they don't always reduce the wall time. 

Before diving into any details, the lesson is this: <span style="background-color: #abe0bb">To optimize any system, first identify its bottleneck; otherwise, you're wasting your time</span>. Horace He's [insight](https://horace.io/brrr_intro.html) inspired [FlashAttention](https://arxiv.org/abs/2205.14135) by Tri Dao's team: With a large enough $N$, attention is actually *not* bottlenecked by compute but by data movements between SRAM and HBM. So fitting everything on SRAM whenever possible will give us a true speedup. 

I've been increasingly fascinated by this type of software-hardware co-design for attention (or deep learning in general) and will review some classic works in this post. First, let's take a look at the "metal", GPUs.

# Know Your GPUs

## Game Cards Born for Deep Learning

Inside your Nintendo Switch are graphics processing units (GPUs) that render life-like scenes in real time. After the [game engine](https://en.wikipedia.org/wiki/Game_engine) determines how characters act and what objects look like based on rules and physics, GPUs break down the scene into small units (e.g., triangles and vertices) and render pixels, lighting, textures, and other graph elements in each unit simultaneously. Without this parallelism, the game world unfolds slowly, making any game too stuck to play.

{{< figure src="https://www.dropbox.com/scl/fi/oirp77bh00gkux8gtic5v/Screenshot-2025-03-15-at-2.58.36-PM.png?rlkey=qstiht7lzgzbvpm4wgqzhkt18&st=g3s1ufu5&raw=1" caption="Game graphics are rendered seamlessly by graphics processing units (GPUs) thanks to their massive parallel processing power." width="500">}}

Turns out matrix multiplications --- the building blocks of modern deep learning --- can similarly be broken down into small units and get processed in parallel. This is the very task GPUs are born for.

{{< figure src="https://www.dropbox.com/scl/fi/vvqvrqfw0eyotbzl2j1y7/Screenshot-2025-03-15-at-3.49.46-PM.png?rlkey=b6vgw73tnjb8rs937j7istoaz&st=269p6gyk&raw=1" caption="Multiplication between two matrices can be executed as parallel inner products. Each inner product can be executed as parallel element-wise multiplications, followed by an addition." width="600">}}

$\mathbf{A} \in \mathbb{R}^{n \times k} \times \mathbf{B} \in \mathbb{R}^{k \times m}$ can be broken down into $n \times m$ inner products between two $k$-vectors (the $i$-th row vector in $\mathbf{A}$ and the $j$-th column vector in $\mathbf{B}$) to be executed at once. Each inner product can be further decomposed into $k$ parallel element-wise multiplications, followed by $(k-1)$ additions to sum up $k$ products.

## GPU Memory Hierarchy

While the terminology might differ, a GPU is similar to many computer systems in that it has a static random-access memory (SRAM) that's physically close to the compute but limited in memory bandwidth, and a high-bandwidth memory (HBM) that computes slowly. Horace He has a wonderful [analogy](https://horace.io/brrr_intro.html) that the HBM is like a warehouse where raw materials and finished products are stored, whereas the SRAM is the storage in the factory where new products are being produced.

{{< figure src="https://www.dropbox.com/scl/fi/5g3edyxzwq1q83gj6cj46/Screenshot-2025-03-15-at-4.26.31-PM.png?rlkey=8h6wet9gim1ofjksu8ibbitk8&st=5k6f3fq2&raw=1" caption="The GPU memory hierarchy (source: Tri Dao's [slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/slides/cs224n-2024-lecture18-deployment-and-efficiency.pdf))." width="800">}}

As the GPU carries out an operation (called a "kernel"), it read inputs from HBM into SRAM and writes outputs from SRAM back to HBM. There are 3 types of costs associated with any GPU systems:

- **Bandwidth costs**: The time spent ferrying data between HBM and SRAM. Also known as "memory bandwidth" or IO cost.
- **Compute costs**: The time actually spent in the "factory" (SRAM) computing, often measured by FLOPs (floating points per second). This is what people think/hope they pay NVIDIA for.
- **Overhead costs**: All else --- e.g., deciding to which factory to send what materials for what products, or spinning up an idle factory.

Understanding the bottleneck of your system is key to efficiency improvements. For instance, if you can't ferry materials fast enough into the factories, then buying more expensive factory machines won't help increase your output. The million-dollar question is: how do you know if you're bound by memory, compute, or overhead?

## Are You Bound by Memory, Compute, or Overhead?

{{< figure src="https://www.dropbox.com/scl/fi/63qnev0qkwnmynp99kj31/Screenshot-2025-03-15-at-8.09.27-PM.png?rlkey=92ofkzpw8qxrhb6xwapsedkds&st=fqr7ekik&raw=1" caption="Diagnose the bottleneck of your system by gradually intensifying compute and measuring wall time (source: Horace He's [blog](https://horace.io/brrr_intro.html))." width="800">}}

Horace He proposed an elegant method for distinguishing memory-bound vs. compute-bound workloads --- all else being equal, if you increase the {{< sidenote "compute intensity" >}}It's the ratio of the number of calculations to the amount of data moved, typically measured in FLOPs per byte.{{< /sidenote >}} (e.g., repeat a toy operation $n$ times, make input matrix dimensions larger, use higher-precision numeric representations) but the runtime doesn't increase, you're likely memory-bound: some of your compute sits idle, ready to process any incoming data ASAP. If runtime starts to increase with compute intensity, you're compute-bound since all your compute is occupied.

To identify overhead costs, you can use the PyTorch profiler to check for large gaps between CPU kernels (sending "instructions") and GPU kernels (ferrying between HBM and SRAM and computing).

# FlashAttention: Fit Stuff on SRAM to Reduce HBM Read/Writes

If there's no room in the factory to store intermediate outputs and moving them to a distant warehouse is a waste of time, we can assemble one small part at a time and combine them into the final product. As long as each small part stays in the factory, we save time on transportation. This is the key intuition behind FlashAttention. 

First, let's review how vanilla attention is computed (see my {{< backlink "attention_as_dict" "post" >}} for an intuitive explanation). To begin, we look up token embeddings and add them with positional encodings. Then we project the $N \times d$ input matrix into 3 matrices, $\mathbf{Q}$ (queries), $\mathbf{K}$ (keys), and $\mathbf{V}$ (values).

{{< figure src="https://www.dropbox.com/scl/fi/e8frszo46hrpklzlhh363/Screenshot-2025-03-16-at-12.32.41-PM.png?rlkey=6r8hdpupn6m5va0u8wbtknnay&st=lvcue89u&raw=1" caption="Computing attention requires us to materialize large $N \times N$ matrices in HBM, which is the bottleneck for long sequence modeling (source: Tri Dao's [talk](https://horace.io/brrr_intro.html))." width="800">}}

We carry out a series of matrix operations in order to eventually obtain the "contextual embedding" of each input token:
1. Compute raw attention scores, **$\mathbf{S} = \mathbf{Q}\mathbf{K}^{\top} \in \mathbb{R}^{N \times N}$**;
2. Apply row-wise softmax on $\mathbf{S}$ so that each row sums to 1. In practice, we can keep 2 separate components:
   - Exponentiate each element in $\mathbf{S}$, $\mathbf{A} = \exp(\mathbf{S}) \in \mathbb{R}^{N \times N}$;
   - Track the sum of each row $i$, $\bm{l} = \sum_i \exp(\mathbf{S})_i$;
3. Compute the output matrix, $\mathbf{O}=\frac{\mathbf{A}}{\bm{l}}\mathbf{V} \in \mathbb{R}^{N \times d}$, which represents the "contextual embedding" of each input token.

Fitting $\mathbf{S}$ and $\mathbf{A}$ on SRAM is not possible for long sequences, so they are moved to HBM with an $O(N^2)$ IO complexity. *If we can split the inputs, may the intermediate results will fit?* People thought about it but hesitated. While inputs are naturally split along the $\mathbf{Q}$ dimension, it seems wrong to further split them along $\mathbf{K}$ and $\mathbf{V}$ since computing the softmax requires summing over each row in the *full* $\mathbf{A}$ matrix. 

## Tiling

With simple rescaling, we can obtain correct softmax results even when splitting inputs along $\mathbf{K}$ and $\mathbf{V}$. "Softmax + rescaling" is the key innovation behind *tiling* in FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)).


Suppose we split $\mathbf{K}$ and $\mathbf{V}$ into two blocks. We can compute $\mathbf{S}^{(1)}$ and $\mathbf{S}^{(2)}$ without issue since matrix multiplications in one block don't interfere with those in another. Similarly, we can obtain $\mathbf{A}^{(1)}$ and $\mathbf{A}^{(2)}$ with issue since element-wise exponentiation is independent.

{{< figure src="https://www.dropbox.com/scl/fi/eprfs3xgnvr8s6n8p8elm/Screenshot-2025-03-16-at-1.15.52-PM.png?rlkey=6oqxe4wjlta3uxecejei4bqla&st=hgq348vi&raw=1" caption="By splitting $\mathbf{K}$ and $\mathbf{V}$ into blocks and computing outputs block by block, we keep computations within the fast SRAM and never materialize large matrices, thereby breaking through the IO bottleneck (source: Tri Dao's [talk](https://horace.io/brrr_intro.html))." width="1000">}}

 However, if we scale $\mathbf{A}^{(1)}$ by $\bm{l}^{(1)} = \sum_i \exp(\mathbf{S}^{(1)})_i$ to obtain the output matrix $\mathbf{O}^{(1)}$, we get wrong results. This is because $\bm{l}^{(1)}$ sums each row within block 1, but we need to sum the entire row in the original $\mathbf{A}$. The good news is that once we process block 2, the final block here, we regain knowledge of the full row sums, $\bm{l}^{(2)} = \bm{l}^{(1)} + \sum_i \exp(\mathbf{S}^{(2)})_i$. This allows us to rescale $\mathbf{O}^{(1)}$ by $\frac{\bm{l}^{(1)}}{\bm{l}^{(2)}}$ to get the right answers. Note that the computation of each block happens *sequentially*, but since each fits within SRAM, it still provides a significant speedup compared to transferring large matrices to HBM.

## Recomputation

Each input token's $\bm{q} \in \mathbf{Q}$, $\bm{k} \in \mathbf{K}$, $\bm{v} \in \mathbf{V}$ projections are learnable --- by updating them through backpropagation, we train the model to make better predictions. To calculate gradients w.r.t. $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ in the backward pass, we normally store intermediate matrices such as $\mathbf{S}$ and $\mathbf{A}$ to avoid recomputation. However, by keeping {{< sidenote "only" >}}If the input sequence is masked, we also need to store the pseudo-random number generator states so we can generate correct `MASK` tokens on the fly.{{< /sidenote >}} $\mathbf{O}$ (output) and $\bm{l}$ (softmax denominators), $\mathbf{S}$ and $\mathbf{A}$ can be recomputed in SRAM. 

{{< figure src="https://www.dropbox.com/scl/fi/ap9fxmtuw3xt2w27c2uvn/Screenshot-2025-03-16-at-2.34.46-PM.png?rlkey=t1f82pgzksr0hrhga5jfumnz0&st=wkfpees1&raw=1" caption="By recomputing $\mathbf{S}$ and $\mathbf{A}$ in SRAM, we trade off extra work for memory saving (source: Tri Dao's [talk](https://horace.io/brrr_intro.html))." width="500">}}

By doing a bit more computation, we can decrease the memory cost from $O(N^2)$ to $O(N)$, increasing the overall training throughput. 

## FlashAttention-2: Fewer Non-`matmul` FLOPs + Better Parallelism

Once FlashAttention overcame the IO bottleneck, compute became a concern again --- GPUs are optimized for matrix multiplications (`matmul`) but significantly slower in non-`matmul` tasks like element-wise operations (e.g., scaling each element in $\mathbf{O}^{(1)}$ by $\frac{\bm{l}^{(1)}}{\bm{l}^{(2)}}$). Since the wide industry adoption of FlashAttention, researchers at companies like OpenAI, NVIDIA, and Meta have been exploring better ways to parallelize attention block computations. FlashAttention 2.0 ([Dao, 2023](https://arxiv.org/abs/2307.08691)) was developed to (1) reduce non-`matmul` FLOPs, (2) further parallelize over the sequence length dimension, and (3) better partition work between {{< sidenote "warps" >}}Each GPU has many thread blocks; each thread block has many threads, organized in groups of 32, each called a "warp".{{< /sidenote >}}, achieving a 2–4x speedup and a 10–20x memory reduction compared with FlashAttention 1.0.

To reduce non-`matmul` FLOPs, FlashAttention 2.0 avoids rescaling each previous block's output. Instead, it carries an updated $\bm{l}$ and rescales only the final output $\mathbf{O}^{(last)}$ to get the right results. For numerical stability of softmax, FlashAttention 1.0 stores not only $\bm{l}^{(j)}$ (row sums of exponentials of the $j$-th block) but also $\bm{m}^{(j)}$ (row maximums of the the $j$-th block) --- row maximums are subtracted from row elements before computing softmax. FlashAttention 2.0 only stores the log sum of exponentials, $L^{(j)} = \bm{m}^{(j)} + \log\bm{l}^{(j)}$.


{{< figure src="https://www.dropbox.com/scl/fi/z1203la1pkz7ckha0crn9/Screenshot-2025-03-16-at-5.07.07-PM.png?rlkey=66a3mw07p9225f3hae6nevo6j&st=t5jcnbux&raw=1" caption="The sequence dimension is divided into row (forward) and column (backward) blocks, with one thread block dedicated to each block (source: [Dao, 2023](https://arxiv.org/abs/2307.08691))." width="600">}}

FlashAttention 1.0 parallelize along the batch and the head dimensions, but not the sequence dimension --- each thread block is responsible for an entire head of an entire sequence. FlashAttention 2.0 further divides the sequence dimension into row/column blocks and assigns one thread block to each row/column block, allowing different "chunks" of a long sequence to be processed in parallel. 

{{< figure src="https://www.dropbox.com/scl/fi/lpp95ckjok0mlbh9i7wtt/Screenshot-2025-03-16-at-5.41.17-PM.png?rlkey=aq4pqccri0zs1cf5si1iv7mx4&st=qwfg21h5&raw=1" caption="Splitting by $\mathbf{Q}$ reduces reads/writes to shared memory (source: [Dao, 2023](https://arxiv.org/abs/2307.08691))." width="600">}}


FlashAttention 1.0 was motivated by the fact that data transfer is slow between SRAM and HBM. Even inside the compute, communication speed differs within vs. between warps, with the former being much faster. FlashAttention 1.0 splits $\mathbf{K}$ and $\mathbf{V}$ into 4 warps --- each warp computes a slice of $\mathbf{S}$ and writes it to the shared memory. To reduce shared memory reads/writes and achieve a speedup, FlashAttention 2.0 splits $\mathbf{Q}$ into 4 warps, allowing each query's output to be computed independently without communication between warps.

## FlashAttention-3: XX and XX


# Wanna Reduce Compute Anyways? Approximate Attention

## Low-Rank Approximation 

## Sparse Approximation

## Try 'Em All: DeepSeekMoE

# Implications for User Sequence Modeling

# References

## Understand Deep Learning Systems
1. Why GPUs are born for parallel processing 👉 [*Harnessing Parallelism: How GPUs Revolutionize Computing*](https://medium.com/accredian/harnessing-parallelism-how-gpus-revolutionize-computing-597f3479d955#:~:text=A%20GPU%20consists%20of%20a,which%20allows%20for%20parallel%20processing.) by Harshita Sharma.
2. Deep learning efficiency = compute + memory + overhead 👉 [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) by Horace He.
3. Software-hardware co-design 👉 *Hardware-aware Algorithms for Sequence Modeling* by Tri Dao, [talk](https://www.youtube.com/live/foG0ebzuw34?si=6FSChDzXjBUqAQX8&t=242) + [slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/slides/cs224n-2024-lecture18-deployment-and-efficiency.pdf) at Stanford MLSys.
4. [Triton](https://openai.com/index/triton/), the lingua franca for GPU programming 👉 [*Tutorials*](https://triton-lang.org/main/getting-started/tutorials/index.html) by OpenAI + Triton rewrite ([repo](https://github.com/unslothai/unsloth)) of popular LLMs by Unsloth AI.

## FlashAttention: IO-Aware, Exact Attention
5. FlashAttention 1.0 👉 [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135) (2022) by Dao et al., *NeurIPS*.
6. FlashAttention 2.0 👉 [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691) (2024) by Dao, *ICLR*.
7. FlashAttention 3.0 👉 [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*](https://arxiv.org/abs/2407.08608) (2024) by Shah et al., *arXiv*.

## Fast & Accurate Attention Approximations
8. Low-rank approximation 👉 [*Linformer*](https://arxiv.org/abs/2006.04768) (2020), [*Linear Transformer*](https://arxiv.org/abs/2006.16236) (ICML 2020), [*Performer*](https://openreview.net/forum?id=Ua6zuk0WRH) (ICLR 2021), [*Loki*](https://arxiv.org/abs/2406.02542) (NeurIPS 2024), 
9. Sparse approximation 👉 [*Sparse Transformers*](https://arxiv.org/abs/1904.10509) (2019), [*Reformer*](https://arxiv.org/abs/2001.04451) (ICLR 2020), [*Routing Transformer*](https://arxiv.org/abs/2003.05997) (ACL 2020)
10. DeepSeek combines blockwise compression/selection + sliding window attention 👉 [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*](https://arxiv.org/abs/2502.11089) (2025) by Yuan et al., *arXiv*.

# GPU Terminology
- **HBM**: XX
- **SRAM**: XX
- **Compute intensity**: XX
- **FLOPs**: XX
- **Kernel**: an operation
- **Kernel fusion**: xx
- **Tensor core**: xx
- **Wrap**: NVIDIA; AMD has a different name
- **CUDA**: a platform
- **Triton**: a programming language

<!-- ## Mamba: Attention is Not What You Need
11. Mamba 1.0 👉 [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752) (2023) by Gu and Dao, *COLM*.
12. Mamba 2.0 👉 [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) (2024) by Dao and Gu, *ICML*. -->