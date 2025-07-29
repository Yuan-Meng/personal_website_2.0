---
title: "Is Generative Recommendation the ChatGPT Moment of RecSys?"
date: 2025-07-29
math: true
categories: ["generative recommendation", "large language models"]
toc: true
---

# Has the Tide Turned? From DLRM to GM

For nearly a decade, recommender systems have remained largely {{< sidenote "the same" >}}It used to be (still is?) the case that if you're familiar with the cascade pipeline and the most popular L1 (e.g., two-tower models and embedding-based retrieval) and L2 (e.g., "Embedding-MLP" style `pAction` models, sequence modeling) architectures, you're golden in almost every ML system design interview. Perhaps a year from now, GenRec talents and experience will be what top companies seek instead.{{< /sidenote >}}. It's hard to even imagine a system without a cascade pipeline in the iconic [YouTube paper](https://research.google.com/pubs/archive/45530.pdf), which retrieves tens of thousands of candidates from a massive corpus, trims them down to hundreds of relevant items using a lightweight ranker (L1), selects the top dozen using a heavy ranker (L2), and makes adjustments based on policy and business logic (L3). Architecture-wise, the L2 ranker hasn't drifted far from the seminal [Deep & Wide network](https://arxiv.org/abs/1606.07792), which embeds input features, passes them through interaction modules, and transforms representations for task heads (e.g., clicks, purchase, video watch). Upgrades to feature interaction (e.g., [DCN-v2](https://arxiv.org/abs/2008.13535), [MaskNet](https://arxiv.org/abs/2102.07619)) and multi-task learning (e.g., [MMoE](https://arxiv.org/abs/2311.09580), [PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)) culminated in Meta's [DHEN](https://arxiv.org/abs/2203.11014), which combines multiple interaction modules and experts to push the limits of this "Deep Learning Recommender System" (DLRM) paradigm. 

{{< figure src="https://www.dropbox.com/scl/fi/96m8zb5yps9ffz9geheu7/Screenshot-2025-07-20-at-11.07.10-PM.png?rlkey=q4xtbxt3r50okrs2zo9vac2xq&st=fzobjxgt&raw=1" caption="Since 2016, web-scale recommender systems mostly use the cascade pipeline and DLRM-style 'Embedding & Interaction & Expert' model architectures." width="1800">}}

In 2025, the tide seems to have finally turned after Meta's [HSTU](https://arxiv.org/abs/2402.17152) delivered perhaps the biggest offline/online metric and serving efficiency gains in recent years --- other top companies such as {{< sidenote "Google" >}}Google DeepMind published TIGER a year before HSTU, but it was used for retrieval. Meta may have been the major influence behind using Generative Recommendation for both retrieval and ranking.{{< /sidenote >}}, Netflix, Kuaishou, ByteDance, Xiaohongshu, Tencent, Baidu, Alibaba, JD.com, and Meituan are starting to embrace a new "Generative Recommendation" (GM) paradigm for retrieval and ranking, reframing the discriminative `pAction` prediction task as a generative task, akin to token predictions in language modeling. 

<!--more-->

What makes Generative Recommendation so magical? Why is it able to unlock the scaling laws in recommender systems in ways that DLRM wasn't able to? Is GM a genuine paradigm shift or a short-lived fad? In this blogpost, let's take a look at GM models coming out from the aforementioned companies and see what the fuss is all about 🕵️. 

# Compositionality, Language, and Intelligence

## The One-Epoch Curse of DLRM Recommenders

If in the old days sciences had a ["physics envy"](https://en.wikipedia.org/wiki/Physics_envy), then today's machine learning definitely has an "LLM envy". By duplicating a deceptively simple architecture --- the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) --- many times over and training models on massive amounts of data, large language models seem to have unlocked {{< sidenote "human-like" >}}Or have they? The new ICML 2025 paper "What Has a Foundation Model Found?" shows foundation models struggle to learn underlying world knowledge and apply it to new tasks, such as discovering Newtonian mechanics from orbital trajectory training and applying it to new physics tasks.{{< /sidenote >}}intelligence. Other AI/ML domains are eager to replicate this success, especially recommender systems, which remain the lifeline of the tech industry.

Why not? After all, there are many similarities between recommender systems and the NLP domain, as noted by Meta researchers in [*Breaking the Curse of Quality Saturation with User-Centric Ranking*](https://arxiv.org/abs/2305.15333):

> The key idea, with an analogy to NLP, is to think of items as tokens and users as documents, i.e., each user is modeled by a list of items that they engaged with, in chronological order according to the time of engagements.

A user is thus a "book" written by a history of engaged items, and web-scale recommender systems provide endless training data. In language models, performance predictably improves with model size or training data (e.g., linear or power-law relationships), a phenomenon called "scaling laws" ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). In DLRM recommender systems, however, scaling laws have never emerged. In recommenders employing Transformer modules, increasing the number of Transformer layers doesn't improve performance --- more often than not, one layer is all you need (e.g., Chen et al., [KDD 19](https://dl.acm.org/doi/abs/10.1145/3326937.3341261), Gu et al., CIKM [20](https://dl.acm.org/doi/abs/10.1145/3340531.3412697), [21](https://dl.acm.org/doi/abs/10.1145/3459637.3481953)). Regardless of the architecture, training DLRM recommenders for more than one epoch typically results in rapid performance deterioration (the "one-epoch phenomenon"; Zhang et al., [CIKM 22](https://dl.acm.org/doi/abs/10.1145/3511808.3557479)).

So how do we explain this stark contrast between language and recommendation models? Back in 2021, Netflix published a "popular science" paper that foresaw the challenges of DLRM-style recommenders. In this paper, [Steck et al. (2021)](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/18140) argued that deep learning recommender systems differ  from other deep learning applications such as image classification in that item IDs are "atomic" and directly available in the data --- there are no low-to-high level feature representations to extract (e.g., pixels to objects in images). As such, deep learning recommenders only require a shallow network to learn user and item embeddings from user-item interactions, effectively learning "dot product" operations. They don't benefit from deeper architectures meant to capture low-level features. 


This observation is profound, highlighting inherent flaws in DLRM:
- **Lack of task complexity**: Why do we need the full power of deep learning to learn a task as simple as performing dot products? If shallow networks are enough, how could scaling laws emerge?
> [...] a syntactically complex phrase is a function of the meanings of its constituent parts and the way they are combined. --- [*Compositionality*](https://oecs.mit.edu/pub/e222wyjy/release/1), Ryan M. Nefdt and Christopher Potts
- **No compositionality, no intelligence**: Imagine if each word as an arbitrary sound unrelated to others --- e.g., saying "blim" for "snake" and "plok" for "rattlesnake" --- learning any {{< sidenote "language" >}}Or music, or planning… For example, while sound frequencies are continuous and infinite, Western music relies on just 12 distinct frequencies and their multiples. Notes form chords, chords form progressions, and so forth. Without hierarchical relationships, composing music would be nearly impossible.{{< /sidenote >}} would be impossible in our lifetime, as we'd spend an eternity just acquiring the vocabulary. Recommender systems face precisely this challenge: a popular social media platform, for instance, may have billions of items, with new ones constantly being added, making the vocabulary enormous, ever-changing, and non-stationary. Moreover, item IDs are arbitrary and atomic, with no relationship between one another that learners can exploit to speed up learning. By contrast, without knowing German, we might guess that "klapperschlange" (rattlesnake) relates to "schlange" (snake). The compositionality of language, i.e., smaller units are combined into complex phrases and concepts, allows humans to express infinite ideas using a finite vocabulary acquired in finite time --- a luxury that recommender systems don't have.



## What Makes Language Special: Hockett's Design Features

In the 1960s, American linguist Charles Hockett proposed 16 ["design features"](https://abacus.bates.edu/acad/depts/biobook/Hockett.htm) to distinguish human language from animal communication. The <span style="background-color: #FFC31B">highlighted</span> ones below (with my paraphrasing) are what I found most interesting and relevant to recommender systems:

1. Vocal-auditory channel: Communication must occur through some medium that transmits a message from the communicator to the receiver. (Note: Hockett's emphasis on vocal communication is now outdated, overlooking sign languages.)
2. Broadcast transmission and directional reception: Message is transmitted in all directions, yet the receiver can tell from where the message came. (Without knowing who's communicating with us from where, we can't communicate back.)
3. Rapid fading: Linguistic message does not persist over time, unlike permanent forms such as stone carvings or knot-tying.
4. Interchangeability: Anyone can say anything (e.g., I can claim "I am your king" even if I'm not) --- unlike only queen ants can produce certain chemicals. Communicators and receivers can switch roles.
5. Total feedback: The communicator can receive their own message, allowing them to control what message to send.
6. Specialization: Communication is for exchanging messages rather than practical purposes (e.g., dogs panting to cool down).
7. <span style="background-color: #FFC31B">Semanticity</span>: Symbols carry stable meanings. (The same symbol shouldn't change meanings from one message to the next.)
8. <span style="background-color: #FFC31B">Arbitrariness</span>: Which symbols map to which meanings is arbitrary.
9. <span style="background-color: #FFC31B">Discreteness</span>: Smaller symbols can be combined into complex symbols in rule-governed ways (noun + "s" $\rightarrow$ plural).
10. <span style="background-color: #FFC31B">Duality of patterning</span>: Atomic symbols have no meaning of their own, yet they can be combined into meaningful message.
11. Displacement: Language can discuss subjects not immediately present (e.g., dinner plans for tomorrow).
12. Prevarication: We can say things that are false or hypothetical.
13. <span style="background-color: #FFC31B">Productivity</span>: We can say things that no one in history has said before, yet those new utterances can be readily understood.
14. Traditional transmission: Language is socially learned, not innate.
15. <span style="background-color: #FFC31B">Learnability</span>: Language can be learned (with ease in childhood).
16. Reflexiveness: Language can describe itself (e.g., "grammar," "sentence," "word," "token," "noun").

For recommender systems to have "intelligence," item IDs need not have inherent meanings from the get-go ("arbitrariness"), but should be decomposable into smaller units in a hierarchical, rule-governed manner ("discreteness," "duality of patterning"), with stable mappings from tokens to meanings ("semanticity"). Hopefully as a result, the system will be able to learn the "item language" ("learnability") and generalize knowledge to new items ("productivity"). Spoiler alert: This is the exact idea behind Semantic IDs ([Rajput et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)).

<details>
  <summary style="color: #FFC31B; cursor: pointer;">Reflections on Linguistics and Recommender Systems</summary>
  <div style="color: #002676; padding-left: 10px;">
    Nine years ago when I first learned about Hockett's design features in a linguistics seminar as a first-year Cognitive Science PhD student, I had {{< sidenote "zero interest" >}}Perhaps because it was too much for me to talk about how to talk about language in a language I didn't grow up speaking. Might that be a lack of reflexive-reflexiveness LOL?{{< /sidenote >}} in linguistics. Three years ago when I started my career as a Machine Learning Engineer, I had no interest in language models, dead set on becoming a recommender system expert — despite closely following OpenAI since 2016 while at Berkeley. It's funny how recognizing the parallels between language and recommender systems finally helped me see the magic (structure $\rightarrow$ learnability) in the former and the beauty (can it be potentially intelligent?) in the latter.
  </div>
</details>


<!-- But how do we decompose an arbitrary item ID into meaningful smaller units? Right now, Semantic IDs is a go-to method and RQ-VAE is the most popular to learn Semantic IDs. 

# Semantic IDs -->

# Break the Non-Scalability Curse in RecSys
## Conjure Up Compositionality via Semantic IDs
Introduce RQ-VAE. Discuss how it's used in TIGER. Talk about COBORA.
## Crank Up Task Complexity via Generative Training


# Flavors of Generative Recommenders
## Generative Training, Generative Inference: HSTU, OneRec
## Generative Pretraining, Discriminative Inference: GPSD

# Lessons (for Non-Meta and Non-Kuaishou Companies)


# References (TODO: clean up by theme)
## Foundations of Generative Recommendation
1. RQ-VAE, the most popular technique for learning Semantic IDs 👉 initially invented to generate audios ([Zeghidour et al., 2021](https://arxiv.org/abs/2107.03312)) and images ([Lee et al., 2022](https://arxiv.org/abs/2203.01941)) with low costs and high fidelity
2. Google DeepMind's TIGER ([Rajput et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)) first applied RQ-VAE to learning semantic IDs and using them for retrieval 👉 later, another Google paper ([Singh et al., 2024](ttps://dl.acm.org/doi/abs/10.1145/3640457.3688190)) applied Semantic IDs to ranking as well
3. Baidu's COBRA ([Yang et al., 2025](https://arxiv.org/abs/2503.02453)) tackles information loss from RQ-VAE quantization
4. Scaling laws in recommender systems
   - Meta: [*Understanding Scaling Laws for Recommendation Models*](https://arxiv.org/abs/2208.08489) by the same team at Meta.
   - Tencent: [*Scaling Law of Large Sequential Recommendation Models*](https://dl.acm.org/doi/abs/10.1145/3640457.3688129) (2025) by Zhang et al., *RecSys*.
   - Alibaba discovered the one-epoch phenomenon in CTR prediction models: Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models
   - limitations: Breaking the curse of quality saturation with user-centric ranking

## Generative Recommendation for Retrieval

5. Alibaba's URM 👉 [*Large Language Model as Universal Retriever in Industrial-Scale Recommender System*](https://arxiv.org/abs/2502.03041) (2025) by Jiang et al., *arXiv*.
6. Baidu's GBS 👉 [*Generative Retrieval for Book Search*](https://arxiv.org/abs/2501.11034) (2025) by Tang et al., *KDD*.
7. Scaling laws in retrieval 👉 [*Exploring Training and Inference Scaling Laws in Generative Retrieval*](https://dl.acm.org/doi/abs/10.1145/3726302.3729973) (2025) by Cai et al., *SIGIR*. -->

## Generative Recommendation for Ranking

### Generative Pretraining 👉 Generative Prediction
8. Meta's HSTU 👉 [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152) (2024) by Zhai et al., *ICML*.
11. Kuaishou's OneRec 👉 [*OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*](https://arxiv.org/abs/2502.18965) (2025) by Deng et al., *arXiv*.

### Generative Pretraining 👉 Discriminative Prediction
14. Alibaba's GPSD 👉 [*Scaling Transformers for Discriminative Recommendation via Generative Pretraining*](https://arxiv.org/abs/2506.03699) (2025) by Wang et al., *KDD*.
16. JD.com 👉 [*Generative Click-through Rate Prediction with Applications to Search Advertising*](https://arxiv.org/abs/2507.11246) (2025) by Kong et al., *arXiv*.

9. Netflix 👉 [*Foundation Model for Personalized Recommendation*](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) (2025) by  Hsiao et al., *Netflix Technology Blog*.
10. Xiaohongshu's RankGPT 👉 [*Towards Large-Scale Generative Ranking*](https://arxiv.org/abs/2505.04180) (2025) by Huang et al., *arXiv*.

12. Meituan's MTGR 👉 [*MTGR: Industrial-Scale Generative Recommendation Framework in Meituan*](https://arxiv.org/abs/2505.18654) (2025) by Han et al., *arXiv*.
13. Alibaba's LUM 👉 [*Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model*](https://arxiv.org/abs/2502.08309) (2025) by Yan et al., *arXiv*.

15. Tencent's LC-Rec 👉 [*Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation*](https://arxiv.org/abs/2311.09049) (2024) by Zheng et al., *ICDE*.
17. ByteDance's RankMixer 👉 [*RankMixer: Scaling Up Ranking Models in Industrial Recommenders*](https://arxiv.org/abs/2507.15551) (2025) by Zhu et al., *arXiv*.
