---
title: "Is Generative Recommendation the Future of RecSys?"
date: 2025-07-20
math: true
categories: ["generative recommendation", "large language models"]
toc: true
---

For nearly a decade, recommender systems have remained largely the same. System-wise, most companies adopt the cascade pipeline in the iconic [YouTube paper](https://research.google.com/pubs/archive/45530.pdf), retrieving tens of thousands of candidates from a massive corpus, trimming them down to thousands of roughly relevant items with a lightweight ranker (L1), before selecting the top dozen using a heavy ranker (L2) and making adjustments based on policy and business logic (L3). Architecture-wise, the L2 ranker hasn't drifted far from the [Deep & Wide network](https://arxiv.org/abs/1606.07792). Years of incremental improvements on feature interaction (e.g., [DCN-v2](https://arxiv.org/abs/2008.13535), [MaskNet](https://arxiv.org/abs/2102.07619)) and multi-task learning (e.g., [MMoE](https://arxiv.org/abs/2311.09580), [PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)) culminated in Meta's [DHEN](https://arxiv.org/abs/2203.11014) that combines multiple interaction modules and experts to push the limits of this "Deep Learning Recommender System" (DLRM) paradigm. 

{{< figure src="https://www.dropbox.com/scl/fi/96m8zb5yps9ffz9geheu7/Screenshot-2025-07-20-at-11.07.10-PM.png?rlkey=q4xtbxt3r50okrs2zo9vac2xq&st=fzobjxgt&raw=1" caption="Since 2016, web-scale recommender systems mostly use the cascade pipeline and DLRM-style 'Embedding & Interaction & Expert' model architectures." width="1800">}}

In 2025, the tide seems to have finally turned after Meta's [HSTU](https://arxiv.org/abs/2402.17152) delivered the biggest model performance and business metric gains that the company has seen in years --- other top companies such as Google, Netflix, Kuaishou, Xiaohongshu, Alibaba, Tencent, Baidu, and Meituan are starting to embrace a new "Generative Recommendation" (GM) paradigm, reframing the discriminative pAction prediction task to a generative task, akin to token prediction in language modeling. 

<!--more-->

What makes Generative Recommendation so magical? Why is it able to unlock the scaling law in recommender systems in ways that DLRM wasn't able to? Is GM a genuine paradigm shift or a short-lived fad? In this blogpost, let's take a look at GM models coming out from the aforementioned companies and see what the fuss is all about 🕵️. 


# References
## Precursors to Generative Recommendation
1. RQ-VAE, the technique behind Semantic ID learning 👉 [*Autoregressive Image Generation using Residual Quantization*](https://arxiv.org/abs/2203.01941) (2022) by Lee et al., *CVPR*.
2. Google DeepMind first introduced Semantic IDs 👉 [*Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations*](https://dl.acm.org/doi/abs/10.1145/3640457.3688190) (2024) by Singh et al., *RecSys*.
3. COBRA addresses information loss from quantization 👉 [*Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations*](https://arxiv.org/abs/2503.02453) (2025) by Yang et al., *arXiv*.

## Generative Recommendation for Retrieval
4. Google DeepMind's TIGER 👉 [*Recommender Systems with Generative Retrieval*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html) (2023) by Rajput et al., *NeurIPS*.
5. Alibaba's URM 👉 [*Large Language Model as Universal Retriever in Industrial-Scale Recommender System*](https://arxiv.org/abs/2502.03041) (2025) by Jiang et al., *arXiv*.
6. Baidu's GBS 👉 [*Generative Retrieval for Book Search*](https://arxiv.org/abs/2501.11034) (2025) by Tang et al., *KDD*.
7. Scaling law analysis 👉 [*Exploring Training and Inference Scaling Laws in Generative Retrieval*](https://dl.acm.org/doi/abs/10.1145/3726302.3729973) (2025) by Cai et al., *SIGIR*.

## Generative Recommendation for Ranking
8. Meta's HSTU 👉 [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152) (2024) by Zhai et al., *ICML*.
   - Related analysis: [*Understanding Scaling Laws for Recommendation Models*](https://arxiv.org/abs/2208.08489) by the same team at Meta.
9. Netflix 👉 [*Foundation Model for Personalized Recommendation*](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) (2025) by  Hsiao et al., *Netflix Technology Blog*.
10. Xiaohongshu's RankGPT 👉 [*Towards Large-Scale Generative Ranking*](https://arxiv.org/abs/2505.04180) (2025) by Huang et al., *arXiv*.
11. Kuaishou's OneRec 👉 [*OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*](https://arxiv.org/abs/2502.18965) (2025) by Deng et al., *arXiv*.
12. Meituan's MTGR 👉 [*MTGR: Industrial-Scale Generative Recommendation Framework in Meituan*](https://arxiv.org/abs/2505.18654) (2025) by Han et al., *arXiv*.
13. Alibaba's LUM 👉 [*Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model*](https://arxiv.org/abs/2502.08309) (2025) by Yan et al., *arXiv*.
14. Alibaba's GPSD 👉 [*Scaling Transformers for Discriminative Recommendation via Generative Pretraining*](https://arxiv.org/abs/2506.03699) (2025) by Wang et al., *KDD*.
15. Tencent 👉 [*Scaling Law of Large Sequential Recommendation Models*](https://dl.acm.org/doi/abs/10.1145/3640457.3688129) (2025) by Zhang et al., *RecSys*.
