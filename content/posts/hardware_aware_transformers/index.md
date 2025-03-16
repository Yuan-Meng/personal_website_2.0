---
title: "Hardware-Aware Transformers for Long Sequence Modeling"
date: 2025-03-15
math: true
categories: ["gpu", "transformers", "ml systems"]
toc: true
---

# Attention Is All You Need --- if You Can Afford the $O(N^2)$ Complexity

Attention is key to the success of modern large language models (LLMs), which overcomes RNNs' difficulty in modeling long-range dependencies by attending to every token in the input sequence at once, without suffering from vanishing or exploding gradients. 

With the power to "see all" comes a time complexity of {{< sidenote "$O(N^2d)$" >}}Here's the trick for counting matrix multiplication complexity: The outer dimensions tell us how many inner products are performed. In step 3, for instance, we got $N \times N$. The inner dimension tells us the complexity of each inner product. In step 3, it's $2d - 1$ = $d$ (element-wise products) + $(d-1)$ (additions). The total complexity of step 3 is therefore $O(N^2d)$. Turns out steps 3 and 5 are equally expensive, dominating the final time complexity.{{< /sidenote >}} and a space complexity of $O(N^2)$, where $N$ is the length of the input sequence (# tokens) and $d$ the hidden embedding dimension.

{{< figure src="https://www.dropbox.com/scl/fi/m8vdwmpqwt40c896ty24v/Screenshot-2025-03-15-at-11.37.40-PM.png?rlkey=t6852oqzse600dc48gjg7rfal&st=r3h14cla&raw=1" caption="In vanilla attention, writing $\mathbf{S}$, $\mathbf{A}$, and $\mathbf{O}$ to memory has an $O(N^2)$ IO complexity." width="600">}}

<!--more-->

It takes 5 steps to compute input contextual embeddings via the attention mechanism (for an intuitive explanation, check out my {{< backlink "attention_as_dict" "post" >}}):
1. **Input embeddings**: Look up the token embedding and generate the positional encoding of each input token 👉 add them together. 
2. **$\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ projections**: Project the $N \times d$ input embedding matrix into 3 matrices, $\mathbf{Q}$ (queries), $\mathbf{K}$ (keys), and $\mathbf{V}$ (values).
3. **$\mathbf{S} = \mathbf{Q}\mathbf{K}^{\top} \in \mathbb{R}^{N \times N}$**: Compute raw attention scores. 
4. **$\mathbf{P} = \mathrm{softmax}(\frac{\mathbf{S}}{\sqrt{d_{\mathbf{K}}}}) \in \mathbb{R}^{N \times N}$**: Apply row-wise softmax so that each row sums to 1 ; for gradient updating stability, scale each element by $\sqrt{d_{\mathbf{K}}}$, where $d_{\mathbf{K}}$ is the hidden  dimension of $\mathbf{K}$.
5. **$\mathbf{O}=\mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d}$**: Compute the output matrix, which represents the "contextual embedding" of each input token.

The space complexity is $O(N^2)$, since in steps 3-5, we write $\mathbf{S}$, $\mathbf{P}$, and $\mathbf{O}$ to memory. Both complexities have room for improvement. Ingenuous solutions require a deep understanding of both the attention algorithm itself and the hardware it lives on. This post talks about software-hardware co-designs for faster and better attention by folks like [Tri Dao](https://tridao.me/) and [Horace He](https://horace.io/index.html). First, let's take a look at the "metal".

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

# Attention Is Bandwidth-Bound: Read/Write Less!
## FlashAttention 1.0

<!-- FlashAttention does a bit more compute to reduce the reads and writes between HBM and SRAM. -->

## FlashAttention 2.0

## FlashAttention 23.0

# Wanna Improve Compute Anyways? Approximate Attention

## Low-Rank Approximation 

## Sparse Approximation

## Try 'Em All: DeepSeekMoE

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