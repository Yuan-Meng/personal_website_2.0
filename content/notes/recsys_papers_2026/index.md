---
title: "40 RecSys + LLM Papers / Blogs to Catch Up On (March 2026)"
date: "2026-03-24"
categories: ["literature review"]
toc: true
math: true
--- 

March has been such a blur that I have a ton of papers saved on Xiaohongshu and LinkedIn but didn't get the chance to read many of them. Below are the cool ones I hope to catch up on before April. 

# Recommender Systems

## Theory + Scaling Laws

Every company on Earth is now pursuing Generative Recommendation (GR), but the exact mechanisms that make GR more generalizable and scalable than the traditional Deep Learning Recommendation Model (DLRM) are not fully understood. Papers below provide theoretical support on why GR is perhaps not the emperor's new clothes. 

1. Meta Ads Ranking AI (formerly CoreML): [How Well Does Generative Recommendation Generalize?](https://arxiv.org/html/2603.19809v1)
    - A theoretical paper from Meta Ads Ranking AI, devising a new measurement of ranking model generalizability, finding models using Semantic IDs generalize better than those using atomic/arbitrary IDs, at some expense of memorization. 
2. Meta Recommendation (formerly MRS): [Bending the Scaling Law Curve in Large-Scale Recommendation Systems](https://arxiv.org/html/2602.16986v1)
   - The so-called ULTRA-HSTU or "HSTU 2.0" paper leverages sparse attention to optimize HSTU.
3. ByteDance: [Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens](https://arxiv.org/html/2601.22694v1)
4. Meta AI: [STEM: Scaling Transformers with Embedding Modules](https://arxiv.org/abs/2601.10639)
5. Academia (UMass Amherst): [Scaling Laws for Embedding Dimension in Information Retrieval](https://arxiv.org/abs/2602.05062)

## Sequence + Non-Sequence Unification

Models like OneRec and HSTU aim to achieve on par performance with DLRM using sequence-only features. Many companies are backpedaling on this aggressive approach and look to unify sequence and non-sequence features in ways that still enjoy scaling laws. 

6. Kuaishou: [A Survey of User Lifelong Behavior Modeling: Perspectives on Efficiency and Effectiveness](https://www.preprints.org/manuscript/202601.1559)
7. ByteDance: [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
8. ByteDance: [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)
9. ByteDance: [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104)
10. ByteDance: [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)
11. Meta AI: [InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction](https://arxiv.org/abs/2411.09852)
12. Meta Ads Ranking AI: [Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design](https://arxiv.org/abs/2602.10016)
13. Alibaba: [EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling](https://arxiv.org/abs/2602.10811)
14. Alibaba: [Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems](https://arxiv.org/abs/2602.09387)
15. Alibaba: [PRECTR-V2: Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization](https://arxiv.org/abs/2602.20676)

## Reasoning & Agentic Models

Everything hot and new in LLMs is bound to be applied to RecSys, like reasoning models or agents. Scaling these models is challenging.

16. Meta Recommendation: [RecoWorld: Building Simulated Environments for Agentic Recommender Systems](https://arxiv.org/html/2509.10397v1)
17. Google DeepMind: [Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents](https://arxiv.org/abs/2602.10226)
18. Kuaishou: [$S^2GR$: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation](https://arxiv.org/abs/2601.18664)
19. Kuaishou: [QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling](https://arxiv.org/html/2602.08559v1)

## Pre-Ranking

I used to work on L1 ranking (or "pre-ranking"), which feels like a shot in the dark compared to L2 ranking with clear engagement labels. Below are two cool new papers on this mysterious stage. 

20. Alibaba: [Generative Pseudo-Labeling for Pre-Ranking with LLMs](https://arxiv.org/abs/2602.20995)
21. ByteDance: [Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-Ranking in Recommender Systems](https://arxiv.org/abs/2603.03770)

## Inference Optimizations

Papers below have ingenious ideas to optimize GR / LLM inference.

22. Google DeepMind + YouTube: [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://huggingface.co/papers/2602.22647)
    - An engineering paper from Google DeepMind on using vectorized tries to accelerate generative retrieval.
23. Ant Group: [Trie-Aware Transformers for Generative Recommendation](https://arxiv.org/abs/2602.21677)
24. PyTorch: [Generalized Dot-Product Attention: Tackling Real-World Challenges in GPU Training Kernels](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/)
25. Meta AI: [A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2602.19938)

## Practical Lessons

Please don't kill me --- papers below are not necessarily SOTA, but carry practical lessons on how companies adopt now-classic GR ideas.  

26. LinkedIn: [An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking](https://arxiv.org/abs/2602.12354)
27. LinkedIn: [Semantic Search At LinkedIn](https://arxiv.org/abs/2602.07309)
28. Uber: [Transforming Ads Personalization with Sequential Modeling and Hetero-MMoE at Uber](https://www.uber.com/en-EG/blog/transforming-ads-personalization/)
29. Coinbase: [Scaling Personalization with User Foundation Models](https://www.coinbase.com/blog/scaling-personalization-with-user-foundation-models)
30. Airbnb: [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873)
31. Netflix: [MediaFM: The Multimodal AI Foundation for Media Understanding at Netflix](https://netflixtechblog.com/mediafm-the-multimodal-ai-foundation-for-media-understanding-at-netflix-e8c28df82e2d)

# Large Language Models

## Fundamental Theory 

A fundamentally new attention mechanism devised by the Kimi team. 

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