---
title: "TL;DR: 40 RecSys + LLM Papers and Blogs to Catch Up On"
date: "2026-03-24"
categories: ["literature review"]
toc: true
math: true
--- 

March has been such a blur that I saved a ton of papers on Xiaohongshu and LinkedIn but didn't get the chance to read many of them. Maybe one day I'll write an assistant to automatically dump my online collections into a post like this. Until then, below are the cool ones that I handpicked and hope to catch up on before April.


# Recommender Systems

## Theory + Scaling Laws

Every company on Earth is now pursuing Generative Recommendation (GR), but the exact mechanisms that make GR more generalizable and scalable than the traditional Deep Learning Recommendation Model (DLRM) are not fully understood. Papers below provide theoretical support on why GR is perhaps not the emperor's new clothes. 

1. Meta Ads Ranking AI (formerly CoreML): [How Well Does Generative Recommendation Generalize?](https://arxiv.org/html/2603.19809v1)
    - **TL;DR**: The two contributions in this theory paper are (1) defining criteria to categorize test examples into memorization (exact 1-hop transitions exist in training data) vs. generalization (transitions can be inferred from training data via symmetry, transitivity, or higher-order connections) and (2) theorizing why semantic ID-based GR models outperform item ID-based models on generalization. The theory is that <span style="background-color: #74A12E66">"item-level generalization can often be interpreted as token-level memorization within the semantic ID space"</span> --- i.e., while a specific item-to-item transition in the test data has never appeared in the training before, the exact token n-gram prefix-to-prefix transition may have appeared and been memorized by semantic ID-based models.
2. Meta Recommendation Systems (formerly MRS): [Bending the Scaling Law Curve in Large-Scale Recommendation Systems](https://arxiv.org/html/2602.16986v1)
   - **TL;DR**: ULTRA-HSTU ("HSTU 2.0") is claimed to have been deployed widely to production, when the less efficient vanilla HSTU wasn't. The main improvements are (1) using a single token to represent an item and the action on it, rather than two ("action encoding"), (2) replacing the $O(L^2)$ full causal self-attention with the $O(L \cdot (K_1 + K_2))$ semi-local attention (SLA), (3) applying mixed-precision training and inference, (4) only using full user sequences in early HSTU layers but a selected segment in later layers ("Attention Truncation"), and (5) processing subsequences using separate HSTU modules ("Mixture of Transducers"). All these tricks led to 5.3x training and 21.4x inference efficiency improvements.
3. ByteDance: [Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens](https://arxiv.org/html/2601.22694v1)
4. Academia (UMass Amherst): [Scaling Laws for Embedding Dimension in Information Retrieval](https://arxiv.org/abs/2602.05062)

## Sequence + Non-Sequence Unification

Models like OneRec and HSTU aim to achieve on par performance with DLRM using sequence-only features. Many companies are backpedaling on this aggressive approach and look to unify sequence and non-sequence features in ways that still enjoy scaling laws. 

5. Kuaishou: [A Survey of User Lifelong Behavior Modeling: Perspectives on Efficiency and Effectiveness](https://www.preprints.org/manuscript/202601.1559)
6. ByteDance: [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
7. ByteDance: [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)
8. ByteDance: [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104)
9. ByteDance: [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)
10. Meta AI: [InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction](https://arxiv.org/abs/2411.09852)
11. Meta Ads Ranking AI: [Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design](https://arxiv.org/abs/2602.10016)
12. Alibaba: [EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling](https://arxiv.org/abs/2602.10811)
13. Alibaba: [Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems](https://arxiv.org/abs/2602.09387)
14. Alibaba: [PRECTR-V2: Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization](https://arxiv.org/abs/2602.20676)

## Reasoning & Agentic Models

Everything hot and new in LLMs is bound to be applied to RecSys, like reasoning models or agents. Scaling these models is challenging.

15. Meta Recommendation: [RecoWorld: Building Simulated Environments for Agentic Recommender Systems](https://arxiv.org/html/2509.10397v1)
16. Google DeepMind: [Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents](https://arxiv.org/abs/2602.10226)
17. Kuaishou: [$S^2GR$: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation](https://arxiv.org/abs/2601.18664)
18. Kuaishou: [QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling](https://arxiv.org/html/2602.08559v1)

## Pre-Ranking

I used to work on L1 ranking (or "pre-ranking"), which feels like a shot in the dark compared to L2 ranking with clear engagement labels. Below are two cool new papers on this mysterious stage. 

19. Alibaba: [Generative Pseudo-Labeling for Pre-Ranking with LLMs](https://arxiv.org/abs/2602.20995)
20. ByteDance: [Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-Ranking in Recommender Systems](https://arxiv.org/abs/2603.03770)

## Inference Optimizations

Papers below have ingenious ideas to optimize GR / LLM inference.

21. Google DeepMind + YouTube: [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://huggingface.co/papers/2602.22647)
    - An engineering paper from Google DeepMind on using vectorized tries to accelerate generative retrieval.
22. Ant Group: [Trie-Aware Transformers for Generative Recommendation](https://arxiv.org/abs/2602.21677)
23. PyTorch: [Generalized Dot-Product Attention: Tackling Real-World Challenges in GPU Training Kernels](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/)
24. Meta AI: [A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2602.19938)

## Practical Lessons

Please don't kill me --- papers below are not necessarily SOTA, but carry practical lessons on how companies adopt now-classic GR ideas.  

25. LinkedIn: [An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking](https://arxiv.org/abs/2602.12354)
26. LinkedIn: [Semantic Search At LinkedIn](https://arxiv.org/abs/2602.07309)
27. Uber: [Transforming Ads Personalization with Sequential Modeling and Hetero-MMoE at Uber](https://www.uber.com/en-EG/blog/transforming-ads-personalization/)
28. Coinbase: [Scaling Personalization with User Foundation Models](https://www.coinbase.com/blog/scaling-personalization-with-user-foundation-models)
29. Airbnb: [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873)
30. Netflix: [MediaFM: The Multimodal AI Foundation for Media Understanding at Netflix](https://netflixtechblog.com/mediafm-the-multimodal-ai-foundation-for-media-understanding-at-netflix-e8c28df82e2d)

# Large Language Models

## Fundamental Theory 

A fundamentally new attention mechanism devised by the Kimi team. 

31. Meta AI (former FAIR's Yuandong Tian is on the author list): [STEM: Scaling Transformers with Embedding odules](https://arxiv.org/abs/2601.10639)
32. Moonshot AI: [Attention Residuals](https://arxiv.org/abs/2603.15031)

## Reasoning & Agentic Models

OpenAI coined a new term "harness engineering" that aims to replace prompt engineering as the fundamental role AI practitioners play. The next three are papers / posts on new trends in agent research. 

33. OpenAI: [Harness Engineering: Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/)
34. OpenAI: [Reasoning Models Struggle to Control their Chains of Thought](https://arxiv.org/abs/2603.05706)
35. Personal Blog ([Zhoutong Fu](https://zhoutongfu.github.io/zhoutong-ai-blog/)) [The Rise of Static Memory in LLMs](https://zhoutongfu.github.io/zhoutong-ai-blog/posts/static_llm_memory/)
36. Memento Team: [Memento-Skills: Let Agents Design Agents](https://arxiv.org/abs/2603.18743)


## Cognitive Science

Finally, some CogSci papers on human-agent interactions. 

37. Academia (Princeton): [Why Human Guidance Matters in Collaborative Vibe Coding](https://arxiv.org/html/2602.10473v1)
38. Academia (Princeton): [Ads in ChatGPT? An Analysis of How Large Language Models Navigate Conflicts of Interest](https://openreview.net/forum?id=kioO6a0oHM)
39. Academia (Princeton): [Cognitive Dark Matter: Measuring What AI Misses](https://arxiv.org/abs/2603.03414)
40. Academia (Princeton): [Mind Your Step (by Step): Chain-of-Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse](https://arxiv.org/abs/2410.21333)