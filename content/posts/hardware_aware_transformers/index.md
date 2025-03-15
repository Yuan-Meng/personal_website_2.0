---
title: "Hardware-Aware Transformers for Long Sequence Modeling"
date: 2025-03-15
math: true
categories: ["gpu", "transformers", "ml systems"]
toc: true
---

Coming soon...

<!--more-->

<!-- how GPU works => why a game card is born for transformers and what kind(s) of computations it's optimized for or not

the Triton language

FlashAttention 1, 2, 3
Mamba
sparse transformers

how it informs sequence modeling for recommender systems --> 

# References

## Understand Deep Learning Systems
1. Deep learning efficiency = compute + memory + overhead 👉 [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) by Horace He.
2. Software-hardware co-design 👉 [*Hardware-aware Algorithms for Sequence Modeling*](https://www.youtube.com/watch?v=foG0ebzuw34) by [Tri Dao](https://tridao.me/) (OG of FlashAttention + Mamba).
3. Triton, the lingua franca for GPU programming 👉 [*Tutorials*](https://triton-lang.org/main/getting-started/tutorials/index.html) by OpenAI + Triton rewrite ([repo](https://github.com/unslothai/unsloth)) of popular LLMs by Unsloth AI.

## FlashAttention: IO-Aware, Exact Attention
4. FlashAttention 1.0 👉 [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135) (2022) by Dao et al., *NeurIPS*.
5. FlashAttention 2.0 👉 [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691) (2024) by Dao, *ICLR*.
6. FlashAttention 3.0 👉 [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*](https://arxiv.org/abs/2407.08608) (2024) by Shah et al., *arXiv*.

## Fast & Accurate Attention Approximations
7. Mamba 1.0 👉 [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752) (2023) by Gu and Dao, *COLM*.
8. Mamba 2.0 👉 [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) (2024) by Dao and Gu, *ICML*.
9. Low-rank approximation 👉 one example... [*Loki: Low-rank Keys for Efficient Sparse Attention*](https://arxiv.org/abs/2406.02542) (2024) by Singhania et al., *NeurIPS*.
10. Sparse attention 👉 among many... [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509) (2019) by Child et al., *arXiv*.
11. DeepSeekMoE throws in blockwise representation/selection + sliding window 👉 [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*](https://arxiv.org/abs/2502.11089) (2025) by Yuan et al., *arXiv*.