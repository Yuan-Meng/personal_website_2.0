---
title: "Hardware-Aware Transformers for Long Sequence Modeling"
date: 2025-03-15
math: true
categories: ["gpu", "transformers", "ml systems"]
toc: true
---

# GPU: How a Game Card is Born for Deep Learning

Inside your Nintendo Switch are graphics processing units (GPUs) that render life-like scenes in real time. After the [game engine](https://en.wikipedia.org/wiki/Game_engine) determines how characters act and what objects look like based on rules and physics, GPUs break down the scene into small units (e.g., triangles and vertices) and render pixels, lighting, textures, and other graph elements in each simultaneously. Without this parallelism, the game world will unfold one pixel at a time, making any game unplayable.

{{< figure src="https://www.dropbox.com/scl/fi/oirp77bh00gkux8gtic5v/Screenshot-2025-03-15-at-2.58.36-PM.png?rlkey=qstiht7lzgzbvpm4wgqzhkt18&st=g3s1ufu5&raw=1" caption="Game graphics are rendered seamlessly by graphics processing units (GPUs) thanks to their massive parallel processing power." width="600">}}

Many matrix operations underlying deep learning can also be broken into small units and processed in parallel --- a task GPUs are born for.

<!--more-->

{{< figure src="https://www.dropbox.com/scl/fi/vvqvrqfw0eyotbzl2j1y7/Screenshot-2025-03-15-at-3.49.46-PM.png?rlkey=b6vgw73tnjb8rs937j7istoaz&st=269p6gyk&raw=1" caption="Multiplication between two matrices can be executed as parallel inner products. Each inner product can be executed as parallel element-wise multiplications, followed by an addition." width="600">}}

For instance, $\mathbf{A} \in \mathbb{R}^{n \times k} \times \mathbf{B} \in \mathbb{R}^{k \times m}$ can be broken down into $n \times m$ inner products between two $k$-vectors, which can be executed at once. Then, each inner product can be decomposed into $k$ element-wise multiplications, which can also be executed in parallel, followed by $(k-1)$ additions to sum up the pairwise products.

<!-- # Attention is IO-Bound
FlashAttention 1, 2, 3 -->


<!-- how GPU works => why a game card is born for transformers and what kind(s) of computations it's optimized for or not

the Triton language


Mamba
sparse transformers

how it informs sequence modeling for recommender systems --> 

# References

## Understand Deep Learning Systems
1. Deep learning efficiency = compute + memory + overhead 👉 [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) by Horace He.
2. Software-hardware co-design 👉 *Hardware-aware Algorithms for Sequence Modeling* by Tri Dao, [talk](https://www.youtube.com/live/foG0ebzuw34?si=6FSChDzXjBUqAQX8&t=242) + [slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/slides/cs224n-2024-lecture18-deployment-and-efficiency.pdf) at Stanford MLSys.
3. [Triton](https://openai.com/index/triton/), the lingua franca for GPU programming 👉 [*Tutorials*](https://triton-lang.org/main/getting-started/tutorials/index.html) by OpenAI + Triton rewrite ([repo](https://github.com/unslothai/unsloth)) of popular LLMs by Unsloth AI.

## FlashAttention: IO-Aware, Exact Attention
4. FlashAttention 1.0 👉 [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135) (2022) by Dao et al., *NeurIPS*.
5. FlashAttention 2.0 👉 [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691) (2024) by Dao, *ICLR*.
6. FlashAttention 3.0 👉 [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*](https://arxiv.org/abs/2407.08608) (2024) by Shah et al., *arXiv*.

## Fast & Accurate Attention Approximations
7. Low-rank approximation 👉 [*Linformer*](https://arxiv.org/abs/2006.04768) (2020), [*Linear Transformer*](https://arxiv.org/abs/2006.16236) (ICML 2020), [*Performer*](https://openreview.net/forum?id=Ua6zuk0WRH) (ICLR 2021), [*Loki*](https://arxiv.org/abs/2406.02542) (NeurIPS 2024), 
8. Sparse attention 👉 [*Sparse Transformers*](https://arxiv.org/abs/1904.10509) (2019), [*Reformer*](https://arxiv.org/abs/2001.04451) (ICLR 2020), [*Routing Transformer*](https://arxiv.org/abs/2003.05997) (ACL 2020)
9. DeepSeek combines blockwise compression/selection + sliding window attention 👉 [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*](https://arxiv.org/abs/2502.11089) (2025) by Yuan et al., *arXiv*.

## Mamba: Beyond Attention
10. Mamba 1.0 👉 [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752) (2023) by Gu and Dao, *COLM*.
11. Mamba 2.0 👉 [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) (2024) by Dao and Gu, *ICML*.