---
title: "Is Generative Recommendation the Future of RecSys?"
date: 2025-07-26
math: true
categories: ["generative recommendation", "large language models"]
toc: true
---

For nearly a decade, recommender systems have remained largely {{< sidenote "the same" >}}It used to be (still is?) the case that if you're familiar with the cascade pipeline and the most popular L1 (e.g., two-tower models and embedding-based retrieval) and L2 (e.g., "Embedding & MLP" style `pAction` models, sequence modeling) architectures, you're golden in almost every ML system design interview. Perhaps a year from now, GenRec talents and experience will be what top companies seek instead.{{< /sidenote >}}. It's hard to even imagine a system without a cascade pipeline in the iconic [YouTube paper](https://research.google.com/pubs/archive/45530.pdf), which retrieves tens of thousands of candidates from a massive corpus, trims them down to thousands of roughly relevant items with a lightweight ranker (L1), selects the top dozen using a heavy ranker (L2), and makes adjustments based on policy and business logic (L3). Architecture-wise, the L2 ranker hasn't drifted far from the seminal [Deep & Wide network](https://arxiv.org/abs/1606.07792), which embeds input features, passes them through some interaction modules, and uses the resulted embeddings to predict binary action probabilities. Years of upgrades to feature interaction (e.g., [DCN-v2](https://arxiv.org/abs/2008.13535), [MaskNet](https://arxiv.org/abs/2102.07619)) and multi-task learning (e.g., [MMoE](https://arxiv.org/abs/2311.09580), [PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)) culminated in Meta's [DHEN](https://arxiv.org/abs/2203.11014), which combines multiple interaction modules and experts to push the limits of this "Deep Learning Recommender System" (DLRM) paradigm. 

{{< figure src="https://www.dropbox.com/scl/fi/96m8zb5yps9ffz9geheu7/Screenshot-2025-07-20-at-11.07.10-PM.png?rlkey=q4xtbxt3r50okrs2zo9vac2xq&st=fzobjxgt&raw=1" caption="Since 2016, web-scale recommender systems mostly use the cascade pipeline and DLRM-style 'Embedding & Interaction & Expert' model architectures." width="1800">}}

In 2025, the tide seems to have finally turned after Meta's [HSTU](https://arxiv.org/abs/2402.17152) delivered perhaps the biggest model performance, business metric, and serving efficiency gains that the company has seen in years --- other top companies such as Google, Netflix, Kuaishou, Xiaohongshu, Alibaba, Tencent, Baidu, Meituan, and JD.com are starting to embrace a new "Generative Recommendation" (GM) paradigm for retrieval and ranking, reframing the discriminative `pAction` prediction task as a generative task, akin to token predictions in language modeling. 

<!--more-->

What makes Generative Recommendation so magical? Why is it able to unlock the scaling laws in recommender systems in ways that DLRM wasn't able to? Is GM a genuine paradigm shift or a short-lived fad? In this blogpost, let's take a look at GM models coming out from the aforementioned companies and see what the fuss is all about 🕵️. 

# Compositionality, Language, and Intelligence

> [...] a syntactically complex phrase is a function of the meanings of its constituent parts and the way they are combined. --- [*Compositionality*](https://en.wikipedia.org/wiki/Principle_of_compositionality), Ryan M. Nefdt and Christopher Potts

Netflix has a PopSci-ish paper ([Steck et al., 2021](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/18140)) arguing that deep learning recommender systems differ from other deep learning applications such as image classification in that item IDs are "atomic" and readily available in the data --- there are no low-to-high level feature representations to extract, such as pixels to objects in images. As such, deep learning recommenders only require a shallow network to learn user and item embeddings from user-item interactions, which can be seen as some form of "dot product" operations; they don't benefit from having deeper architectures to learn low-level features. 


This observation is incredibly deep, perhaps more so from the first to the second point (I'm biased as a former Cognitive Scientist):
- **No scaling laws**: Why do we need the full power of deep learning if learning dot products is all we need? And if a shallow network suffices, scaling laws --- where model performance improves with more parameters, data, and compute --- wouldn't exist.
- **No compositionality, no intelligence**: We'll never learn language or think thoughts if each word is an arbitrary sound unrelated to others --- imagine saying "blim" for "snake" and "plok" for "rattlesnake". By contrast, without knowing German, we can quickly guess that "klapperschlange" (rattlesnake) perhaps relates to "schlange" (snake). The way language combines smaller units of meanings into complex concepts allows humans to express infinite thoughts with a finite vocabulary acquired in a finite amount of time. Traditional item IDs, however, are arbitrary and atomic --- "intelligence" likely will never emerge when the system reasons with a huge vocabulary consisted of unrelated tokens and concepts, where every item has to be engaged with and there is no generalization between related items.
 

<!-- But how do we decompose an arbitrary item ID into meaningful smaller units? Right now, Semantic IDs is a go-to method and RQ-VAE is the most popular to learn Semantic IDs. -->

<!-- # Semantic IDs -->


# References
## Precursors to Generative Recommendation
1. RQ-VAE, the technique behind Semantic ID learning 👉 [*Autoregressive Image Generation using Residual Quantization*](https://arxiv.org/abs/2203.01941) (2022) by Lee et al., *CVPR*.
2. Recommender systems speak "Semantic IDs" 👉 [*Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations*](https://dl.acm.org/doi/abs/10.1145/3640457.3688190) (2024) by Singh et al., *RecSys*.
3. COBRA addresses information loss from RQ-VAE quantization 👉 [*Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations*](https://arxiv.org/abs/2503.02453) (2025) by Yang et al., *arXiv*.

## Generative Recommendation for Retrieval
4. Google DeepMind's TIGER 👉 [*Recommender Systems with Generative Retrieval*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html) (2023) by Rajput et al., *NeurIPS*.
5. Alibaba's URM 👉 [*Large Language Model as Universal Retriever in Industrial-Scale Recommender System*](https://arxiv.org/abs/2502.03041) (2025) by Jiang et al., *arXiv*.
6. Baidu's GBS 👉 [*Generative Retrieval for Book Search*](https://arxiv.org/abs/2501.11034) (2025) by Tang et al., *KDD*.
7. Scaling laws in retrieval 👉 [*Exploring Training and Inference Scaling Laws in Generative Retrieval*](https://dl.acm.org/doi/abs/10.1145/3726302.3729973) (2025) by Cai et al., *SIGIR*.

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
16. JD.com 👉 [*Generative Click-through Rate Prediction with Applications to Search Advertising*](https://arxiv.org/abs/2507.11246) (2025) by Kong et al., *arXiv*.
17. Pinterest's PinFM 👉 [*PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform*](https://arxiv.org/abs/2507.11246) (2025) by Chen et al., *RecSys*.
