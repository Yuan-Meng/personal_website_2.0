---
title: "40 RecSys + LLM Papers and Blogs to Catch Up On"
date: "2026-03-25"
categories: ["literature review"]
toc: true
math: true
--- 

This March has been such a blur that I saved a ton of papers on Xiaohongshu and LinkedIn but didn't get the chance to read many of them. Perhaps one day I shall build an agent to automatically dump my online collections into a TL;DR post like this. Until then, below are the cool ones that I handpicked and hope to catch up on before April.


# Recommender Systems

## Theory + Scaling Laws

Every company on Earth is now pursuing Generative Recommendation (GR), but the exact mechanisms that make GR more generalizable and scalable than the traditional Deep Learning Recommendation Model (DLRM) are not fully understood. The papers below provide theoretical support for why GR may not be the emperor's new clothes.

1. Meta Monetization Ranking AI (formerly CoreML): [How Well Does Generative Recommendation Generalize?](https://arxiv.org/html/2603.19809v1)
    - **TL;DR**: The two contributions in this theory paper are (1) defining criteria to categorize test examples into memorization (exact 1-hop transitions exist in training data) vs. generalization (transitions can be inferred from training data via symmetry, transitivity, or higher-order connections) and (2) theorizing why semantic ID-based GR models outperform item ID-based models on generalization. The theory is that <span style="background-color: #FFC31B">"item-level generalization can often be interpreted as token-level memorization within the semantic ID space"</span> --- i.e., while a specific item-to-item transition in the test data has never appeared in training before, the exact token n-gram prefix-to-prefix transition may have appeared and been memorized by semantic ID-based models.
2. Meta Recommendation Systems (formerly MRS): [Bending the Scaling Law Curve in Large-Scale Recommendation Systems](https://arxiv.org/html/2602.16986v1)
   - **TL;DR**: ULTRA-HSTU ("HSTU 2.0") is claimed to have been deployed widely to production, while the less efficient vanilla HSTU wasn't. The main improvements are (1) using a single token to represent an item and the action on it, rather than two ("action encoding"), (2) replacing the $O(L^2)$ full causal self-attention with the $O(L \cdot (K_1 + K_2))$ semi-local attention (SLA), (3) applying mixed-precision training and inference, (4) only using full user sequences in early HSTU layers but a selected segment in later layers ("Attention Truncation"), and (5) processing subsequences using separate HSTU modules ("Mixture of Transducers"). All these tricks led to 5.3x training and 21.4x inference efficiency improvements.
3. ByteDance: [Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens](https://arxiv.org/html/2601.22694v1)
    - **TL;DR**: Naively replacing item IDs with semantic IDs degrades model performance, because (1) semantic IDs by default encode content but not collaborative (i.e., user-item interaction) information and (2) generalization often comes at the price of memorization, as discussed in the first paper. As an antidote, the token-based recommendation model (TRM) (1) learns collaborative-filtering-aware embeddings based on positive query-item and item-item pairs from user engagements and in-batch negatives and (2) uses RQ-Kmeans to learn semantic tokens (original tokens for generalization + BPE tokens for memorization). TRM is trained with both the generative next-token-prediction loss and the discriminative BCE loss (i.e., whether or not the user had a positive engagement with the given item). We can plug TRM into models like RankMixer to bend the scaling laws.
4. Kuaishou: [A Survey of User Lifelong Behavior Modeling: Perspectives on Efficiency and Effectiveness](https://github.com/Kuaishou-RecModel/Survey-of-ULBM?tab=readme-ov-file)
    - **TL;DR**: A literature review on ultra-long sequence modeling, which boils down to search-based methods, compression-based methods, or a hybrid of both. Amazingly, its [repo](https://github.com/Kuaishou-RecModel/Survey-of-ULBM?tab=readme-ov-file) might just have links to all long-sequence papers under the sun!
<!-- 4. Academia (UMass Amherst): [Scaling Laws for Embedding Dimension in Information Retrieval](https://arxiv.org/abs/2602.05062) -->

## Sequence + Non-Sequence Unification

Sequence-only models such as Kuaishou's One series and Meta's HSTU aim to achieve on-par performance with DLRM without non-sequence features. They have questionable success in production. Many companies are backpedaling on this aggressive approach and looking to unify sequence and non-sequence features in ways that still benefit from scaling laws. On this new path, Kuaishou is pretty quiet, whereas ByteDance, Meta, and Alibaba dominate the "leaderboard". 

5. ByteDance: [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104)
   - **TL;DR**: <span style="background-color: #FFC31B">OneTrans $\approx$ [RankMixer](https://arxiv.org/abs/2507.15551) (feature interaction modules) + [LONGER](https://arxiv.org/abs/2505.04421) (user sequence modules)</span>. Instead of encoding user sequences, concatenating the static, compressed sequence representation with non-sequence features, and passing the concatenated features to feature interaction layers ("encode-then-interaction"), OneTrans uses a unified tokenizer to convert sequence (S) and non-sequence (NS) features into a single token sequence, which is then passed to a *pyramid* (each layer selecting a shorter sequence) of causal OneTrans Transformer blocks. S tokens share the same QKV projections and go through a shared FFN since they are homogeneous, and each S token can only attend to earlier S tokens. Instead of dedicating a separate NS token to each NS feature, we can group NS features based on conceptual understanding or perform an automatic splitting after projection --- the latter leads to smaller kernel launch overhead. By contrast, heterogeneous NS tokens each have their own QKV projections and FFN and can attend to all S tokens at once. We can enable KV cache for S tokens since they remain the same during each user request, whereas NS tokens might change with every candidate. OneTrans scales better than conventional *encode-then-interaction* baselines.
6. ByteDance: [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
    - **TL;DR**: HyFormer, also from ByteDance, aims to achieve the same goal as OneTrans --- unifying sequence modeling and feature interaction into a single backbone. It was published later than OneTrans with better results. HyFormer's core critique of OneTrans is that <span style="background-color: #FFC31B">merging multiple heterogeneous sequences into a single stream via `[SEP]` tokens hurts model performance</span>. Instead, HyFormer processes each sequence independently via two alternating steps per layer. First, NS features (concatenated with a pooled sequence summary) are projected through lightweight MLPs into "Global Tokens." The **Query Decoding** step then performs cross-attention between Global Tokens (Q) and the KV representations of each S sequence independently, so Global Tokens absorb sequence information. The **Query Boosting** step concatenates Global Tokens with NS tokens into a $T \times D$ matrix and mixes them via `HeadMixing` first introduced in RankMixer --- each input token is split into $H$ heads and each head is concatenated across all $T$ tokens to create $H$ mixed output tokens. Just like that, this parameter-free "transpose-and-reshape" token mixer allows every output token to see every input token, even if just a slice. These two steps alternate in each layer, producing progressively richer Global Tokens to be used by task-specific heads.
7. ByteDance: [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)
    - **TL;DR**: Yet another something-something `Former` from {{< sidenote "ByteDance" >}}While reading these "Former" papers, I realized certain companies really like certain terms, which are 99% endorsed by their research leadership. So even if a conference adopts double-blind reviews, I'd still bet my money on a paper mentioning the term "Large Recommendation Model" (LRM) coming from ByteDance.{{< /sidenote >}} aimed at unifying feature interaction and sequence modeling. MixFormer's core critique of OneTrans is that <span style="background-color: #FFC31B">sequence and non-sequence modules don't *co-scale*</span>. First, OneTrans concatenates S and NS tokens along the sequence length dimension, so attention cost is completely dominated by S tokens. Second, S and NS tokens use separate parameters, so they compete for capacity under a fixed budget. MixFormer addresses both issues. To address compute asymmetry, it adopts a decoder-style cross-attention where NS features serve as Qs ($N$ heads, typically ~16) and S features serve as KVs --- scaling either component contributes proportionally to total compute. To address parameter competition, each MixFormer block uses per-head FFNs in both the **Query Mixer** (feature interaction among NS heads via `HeadMixing` + per-head FFN like HyFormer) and **Output Fusion** (per-head FFN after cross-attention), where each head uses shared parameters to represent NS query and S sequence information together. MixFormer also introduces **User-Item Decoupling**: NS heads are partitioned into user-side and item-side subsets via a mask in `HeadMixing` --- user-side computations (including S KVs) can be shared across all candidates in each request via Request Level Batching (RLB), achieving ~36% serving FLOPs reduction.
8. ByteDance: [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)
    - **TL;DR**: This paper fixes issues that prevent RankMixer from scaling beyond 1B parameters. The first issue is that `HeadMixing` causes semantic alignment in residual connections --- each output token no longer represents a user action, but a mixture of everyone's $h$-th head. The residual $F(x') + x$ makes no sense since each input token in $x$ still represents a user action. TokenMixer-Large restores the mixed token dimension back to $T \times D$, so each output token can be construed as a user action for meaningful residuals. To further scale up, TokenMixer-Large (1) adds inter-layer residuals every 2-3 layers (except for the final layer), (2) removes fragmented operators such as DCNv2, and (3) endorses a "sparse training and sparse serving" strategy, splitting each expert into multiple sub-experts based on the per-token SwiGLU and activating them sparsely. As a result, TokenMixer-Large can scale to 7B online and 15B offline.
9. Meta Monetization Ranking AI: [InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction](https://arxiv.org/abs/2411.09852)
    - **TL;DR**: Meta's InterFormer is strikingly {{< sidenote "similar" >}}Interestingly, InterFormer was published before OneTrans.{{< /sidenote >}} to ByteDance's HyFormer. The core idea is allowing bidirectional information exchange between sequence and non-sequence features. The non-sequence summarization is prepended to each sequence as the `CLS` token and it also influences the sequence projection in the so-called "personalized" FFN. In turn, the sequence summarization is concatenated with non-sequence features, which is then passed through interaction layers (e.g., DCNv2, DHEN).
10. Meta Monetization Ranking AI: [Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design](https://arxiv.org/abs/2602.10016)
    - **TL;DR**: Kunlun claims to be the first to achieve power-law scaling in unified sequence + non-sequence models. It's built upon InterFormer and introduces critical efficiency improvements, including (1) reformulating PFFN (personalized FFN) as a fused Generalized Dot-Product Attention (GDPA) kernel, (2) replacing PMA (Pooling by Multihead Attention) with Hierarchical Seed Pooling (HSP) for sequence summarization, (3) using Sliding Window Attention instead of full attention in long sequences, (4) allocating different model capacity to different event types based on importance (Event-Level Personalization), and (5) selectively skipping modules in each layer (Computation Skip). Compared with InterFormer, Kunlun improves MFU (model flops utilization) from 17% to 37% and has 2x scaling efficiency (the slope of the performance-compute curve).
11. Alibaba: [EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling](https://arxiv.org/abs/2602.10811)
    - **TL;DR**: EST claims that previous unified sequence + non-sequence models like OneTrans and [MTGR](https://arxiv.org/html/2505.18654v4) are only "partially unified" but it somehow achieves "full unification". I fail to understand this argument but do think this paper makes two solid contributions: (1) through effective rank analysis, the authors show that S-NS cross-attention carries almost all useful signal while S-S and NS-NS interactions are largely redundant, motivating "Lightweight Cross-Attention" (LCA) that keeps only the S-NS block; and (2) instead of adding multimodal content as token embeddings, EST uses frozen content embeddings to compute intra-sequence item-to-item similarities so each S token attends only to its top-5 most similar neighbors ("Content Sparse Attention"). 
12. Alibaba: [Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems](https://arxiv.org/abs/2602.09387)
    - **TL;DR**: This paper is an independent effort from TokenMixer-Large to improve RankMixer. First, RankMixer splits NS features without respecting semantic boundaries; in this paper,  `HeMix` performs bit-wise feature interaction before splitting ("interact-then-split"). Second, RankMixer treats global (lifelong) and real-time (same-day) sequences the same way, whereas `HeMix` splits NS tokens into two groups, each attending to one sequence, and augments them with learnable query tokens (inspired by BLIP-2's Q-Former) to capture candidate-invariant patterns. Lastly, `HeadMixing` in RankMixer is a parameter-free reshape that can't capture nonlinear dependencies, whereas `HeteroMixer` in this paper processes each mixed token with a dedicated low-rank MLP ("fuse → low-rank MLP interaction → reconstruct").

<!-- 13. Alibaba: [PRECTR-V2: Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization](https://arxiv.org/abs/2602.20676) -->

## Reasoning & Agentic Models

Everything hot and new in LLMs is bound to be applied to RecSys, like reasoning models or agents. Scaling these models remains challenging.

13. Meta Monetization Ranking AI: [Generative Reasoning Re-Ranker](https://arxiv.org/abs/2602.07774)
14. Meta Recommendation Systems: [RecoWorld: Building Simulated Environments for Agentic Recommender Systems](https://arxiv.org/html/2509.10397v1)
15. Google DeepMind: [Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents](https://arxiv.org/abs/2602.10226)
16. Kuaishou: [$S^2GR$: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation](https://arxiv.org/abs/2601.18664)
17. Kuaishou: [QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling](https://arxiv.org/html/2602.08559v1)

## Pre-Ranking

I used to work on L1 ranking (or "pre-ranking"), which feels like a shot in the dark compared to L2 ranking that has direct engagement labels. Below are two cool new papers on this mysterious stage.

18. Alibaba: [Generative Pseudo-Labeling for Pre-Ranking with LLMs](https://arxiv.org/abs/2602.20995)
19. ByteDance: [Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-Ranking in Recommender Systems](https://arxiv.org/abs/2603.03770)

## Inference Optimizations

The papers below have ingenious ideas for optimizing inference.

20. Google DeepMind + YouTube: [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://huggingface.co/papers/2602.22647)
    - An engineering paper from Google DeepMind on using vectorized tries to accelerate generative retrieval.
21. Ant Group: [Trie-Aware Transformers for Generative Recommendation](https://arxiv.org/abs/2602.21677)
22. PyTorch: [Generalized Dot-Product Attention: Tackling Real-World Challenges in GPU Training Kernels](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/)
23. Meta Monetization Ranking AI: [A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2602.19938)

## Practical Lessons

Not necessarily SOTA, but practical lessons on adopting classic GR.

24. LinkedIn: [An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking](https://arxiv.org/abs/2602.12354)
25. LinkedIn: [Semantic Search At LinkedIn](https://arxiv.org/abs/2602.07309)
26. Uber: [Transforming Ads Personalization with Sequential Modeling and Hetero-MMoE at Uber](https://www.uber.com/en-EG/blog/transforming-ads-personalization/)
27. Coinbase: [Scaling Personalization with User Foundation Models](https://www.coinbase.com/blog/scaling-personalization-with-user-foundation-models)
28. Airbnb: [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873)
29. Netflix: [MediaFM: The Multimodal AI Foundation for Media Understanding at Netflix](https://netflixtechblog.com/mediafm-the-multimodal-ai-foundation-for-media-understanding-at-netflix-e8c28df82e2d)
30. Google: [One Model, Two Markets: Bid-Aware Generative Recommendation](https://arxiv.org/pdf/2603.22231)

# Large Language Models

## Fundamental Theory

A fundamentally new attention mechanism devised by the Kimi team.

31. Meta AI (former FAIR's Yuandong Tian is on the author list): [STEM: Scaling Transformers with Embedding Modules](https://arxiv.org/abs/2601.10639)
32. Moonshot AI: [Attention Residuals](https://arxiv.org/abs/2603.15031)

## Reasoning & Agentic Models

OpenAI coined a new term, "harness engineering," that aims to replace prompt engineering as the fundamental role AI practitioners play. The next three are papers and posts on new trends in agent research.

33. OpenAI: [Harness Engineering: Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/)
34. OpenAI: [Reasoning Models Struggle to Control their Chains of Thought](https://arxiv.org/abs/2603.05706)
35. Personal Blog ([Zhoutong Fu](https://zhoutongfu.github.io/zhoutong-ai-blog/)) [The Rise of Static Memory in LLMs](https://zhoutongfu.github.io/zhoutong-ai-blog/posts/static_llm_memory/)
36. Memento Team: [Memento-Skills: Let Agents Design Agents](https://arxiv.org/abs/2603.18743)

## Cognitive Science

Finally, some CogSci papers on human-agent interactions.

37. Academia (Princeton): [Why Human Guidance Matters in Collaborative Vibe Coding](https://arxiv.org/html/2602.10473v1)
38. Academia (Princeton): [Ads in ChatGPT? An Analysis of How Large Language Models Navigate Conflicts of Interest](https://openreview.net/forum?id=kioO6a0oHM)
49. Academia (Princeton): [Cognitive Dark Matter: Measuring What AI Misses](https://arxiv.org/abs/2603.03414)
40. Academia (Princeton): [Mind Your Step (by Step): Chain-of-Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse](https://arxiv.org/abs/2410.21333)