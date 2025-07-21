---
title: "Is Generative Recommendation the Future of RecSys?"
date: 2025-07-20
math: true
categories: ["generative recommendation", "large language models"]
toc: true
---

For nearly a decade, recommender systems have remained largely the same. Most companies adopt YouTube's cascade pipeline ([Covington, 2016](https://research.google.com/pubs/archive/45530.pdf)), which sifts through a massive corpus to retrieve $10^3$ to $10^4$ candidates (L0), narrows them down to $10^2$ to $10^3$ with a lightweight ranker (L1), before selecting the top few using a heavy ranker (L2) and making adjustments based on policy and business logic (L3). L2 architectures haven't drifted far from Google's Wide & Deep network ([Cheng et al., 2016](https://arxiv.org/abs/1606.07792)), with incremental improvements on feature interaction (e.g., [DCN-v2](https://arxiv.org/abs/2008.13535), [MaskNet](https://arxiv.org/abs/2102.07619)) and multi-task learning (e.g., [MMoE](https://arxiv.org/abs/2311.09580), [PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)). These efforts culminated in Meta's [DHEN](https://arxiv.org/abs/2203.11014) that combines multiple interaction modules and experts to push the limits of the so-called "Deep Learning Recommender System" (DLRM) paradigm.


{{< figure src="https://www.dropbox.com/scl/fi/96m8zb5yps9ffz9geheu7/Screenshot-2025-07-20-at-11.07.10-PM.png?rlkey=q4xtbxt3r50okrs2zo9vac2xq&st=fzobjxgt&raw=1" caption="Recommender Systems." width="1800">}}

Interesting works came out when large language models (LLMs) first gained traction, such as using LLMs to re-rank documents (e.g., [Qin et al., 2023](https://arxiv.org/pdf/2306.17563)) --- but they felt more like toys than contenders to DLRM-based recommender systems serving millions of users with low latency and costs. This year, the tide seems to have turned. Companies with top recommender systems such as Meta, Google, Netflix, Kuaishou, Xiaohongshu, Alibaba, Baidu, Tencent, Meituan, etc. are embracing a new "Generative Recommendation" (GM) paradigm, reframing recommendation as a generative task, akin to token prediction in language. This shift has delivered the biggest model performance and business metric gains these companies have seen in years. 

So, what makes GM so magical, improving both model performance and efficiency? What serving changes are needed to productionize GM-based recommenders? In this blogpost, I review the past year's GM work and distill key lessons for companies looking to follow.


<!--more-->

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
