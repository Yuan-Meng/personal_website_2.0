---
title: "Is Generative Recommendation the ChatGPT Moment of RecSys?"
date: 2025-08-03
math: true
categories: ["generative recommendation", "large language models"]
toc: true
---

# Has the Tide Turned? From DLRM to GR

For nearly a decade, recommender systems have remained largely {{< sidenote "the same" >}}It used to be (still is?) the case that if you're familiar with the cascade pipeline and the most popular L1 (e.g., two-tower models and embedding-based retrieval) and L2 (e.g., "Embedding-MLP" style `pAction` models, sequence modeling) architectures, you're golden in almost every ML system design interview. Perhaps a year from now, GenRec talents and experience will be what top companies seek instead.{{< /sidenote >}}. It's hard to even imagine a system without a cascade pipeline in the iconic [YouTube paper](https://research.google.com/pubs/archive/45530.pdf), which retrieves tens of thousands of candidates from a massive corpus, trims them down to hundreds of relevant items using a lightweight ranker (L1), selects the top dozen using a heavy ranker (L2), and makes adjustments based on policy and business logic (L3). Architecture-wise, the L2 ranker hasn't drifted far from the seminal [Deep & Wide network](https://arxiv.org/abs/1606.07792), which embeds input features, passes them through interaction modules, and transforms representations for task heads (e.g., clicks, purchase, video watch). Upgrades to feature interaction (e.g., [DCN-v2](https://arxiv.org/abs/2008.13535), [MaskNet](https://arxiv.org/abs/2102.07619)) and multi-task learning (e.g., [MMoE](https://arxiv.org/abs/2311.09580), [PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)) culminated in Meta's [DHEN](https://arxiv.org/abs/2203.11014), which combines multiple interaction modules and experts to push the limits of this "Deep Learning Recommender System" (DLRM) paradigm. 

{{< figure src="https://www.dropbox.com/scl/fi/96m8zb5yps9ffz9geheu7/Screenshot-2025-07-20-at-11.07.10-PM.png?rlkey=q4xtbxt3r50okrs2zo9vac2xq&st=fzobjxgt&raw=1" caption="Since 2016, web-scale recommender systems mostly use the cascade pipeline and DLRM-style 'Embedding & Interaction & Expert' model architectures." width="1800">}}

In 2025, the tide seems to have finally turned after Meta's [HSTU](https://arxiv.org/abs/2402.17152) delivered perhaps the biggest offline/online metric and serving efficiency gains in recent years --- other top companies such as {{< sidenote "Google" >}}Google DeepMind published TIGER a year before HSTU, but it was used for retrieval only. Meta might have been the major influence behind using Generative Recommendation for both retrieval and ranking.{{< /sidenote >}}, Kuaishou, Meituan, Alibaba, Netflix, Xiaohongshu, ByteDance, Tencent, Baidu, and JD.com are starting to embrace a new "Generative Recommendation" (GR) paradigm for retrieval and ranking, reframing the discriminative `pAction` prediction task as a generative task, akin to token predictions in language modeling. 

<!--more-->

What makes GR magical? Why can it unlock scaling laws in recommender systems in ways that DLRM wasn't able to? Is GR a genuine paradigm shift or a short-lived fad? In this blogpost, let's check out GR models from above companies and see what the fuss is all about 👀. Since I work on ranking, I mainly focus on ranking applications in this post, but GR is first and widely applied in retrieval.

{{< figure src="https://www.dropbox.com/scl/fi/x0y8rm8ph7dz7u1s2bli1/Screenshot-2025-07-29-at-10.18.10-PM.png?rlkey=yww2exuhetiu3jipkqyl5hvtn&st=ojhg4d1j&raw=1" caption="The landscape of Generative Recommenders (GR) in the past year." width="1800">}}

# Compositionality, Language, and Intelligence

## The One-Epoch Curse of DLRM Recommenders

If in the old days sciences had a ["physics envy"](https://en.wikipedia.org/wiki/Physics_envy), then today's machine learning definitely has an "LLM envy". By duplicating a deceptively simple architecture --- the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) --- many times over and training models on massive amounts of data, large language models seem to have unlocked {{< sidenote "human-like" >}}Or have they? The new ICML 2025 paper "What Has a Foundation Model Found?" shows foundation models struggle to learn underlying world knowledge and apply it to new tasks, such as discovering Newtonian mechanics from orbital trajectory training and applying it to new physics tasks.{{< /sidenote >}}intelligence. Other AI/ML domains are eager to replicate this success, especially recommender systems, which remain the lifeline of the tech industry.

Why not? After all, there are many similarities between recommender systems and the NLP domain, as noted by Meta researchers in [*Breaking the Curse of Quality Saturation with User-Centric Ranking*](https://arxiv.org/abs/2305.15333):

> The key idea, with an analogy to NLP, is to think of items as tokens and users as documents, i.e., each user is modeled by a list of items that they engaged with, in chronological order according to the time of engagements.

A user is thus a "book" written by a history of engaged items, and web-scale recommender systems provide endless training data. In language models, performance predictably improves with model size or training data (e.g., linear or power-law relationships), a phenomenon called "scaling laws" ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). In DLRM recommender systems, however, scaling laws have never emerged. In recommenders employing Transformer modules, increasing the number of Transformer layers doesn't improve performance --- more often than not, one layer is all you need (e.g., Chen et al., [KDD 19](https://dl.acm.org/doi/abs/10.1145/3326937.3341261), Gu et al., CIKM [20](https://dl.acm.org/doi/abs/10.1145/3340531.3412697), [21](https://dl.acm.org/doi/abs/10.1145/3459637.3481953)). Regardless of the architecture, training DLRM recommenders for more than one epoch typically results in rapid performance deterioration (the "one-epoch phenomenon"; Zhang et al., [CIKM 22](https://dl.acm.org/doi/abs/10.1145/3511808.3557479)).

<!-- {{< figure src="https://www.dropbox.com/scl/fi/kwizvxifw76foyarxynqn/Screenshot-2025-08-02-at-9.51.24-AM.png?rlkey=f63gie8g7ttx9zl9p87i3eh5e&st=kgc9inhv&raw=1" caption="Hypothesis behind one epoch phenomenon." width="1800">}} -->

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

For recommender systems to have "intelligence," item IDs need not have inherent meanings from the get-go ("arbitrariness"), but should be decomposable into smaller units in a hierarchical, rule-governed manner ("discreteness," "duality of patterning"), with stable mappings from tokens to meanings ("semanticity"). Hopefully as a result, the system will be able to learn the "item language" ("learnability") and generalize knowledge to new items ("productivity"). Spoiler alert: This is the exact idea behind Semantic IDs (e.g., [Rajput et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html), [Singh et al., 2024](https://dl.acm.org/doi/abs/10.1145/3640457.3688190), [Yang et al., 2025](https://arxiv.org/abs/2503.02453)), which we'll discuss in the next section.

<details>
  <summary style="color: #FFC31B; cursor: pointer;">Reflections on Linguistics and Recommender Systems</summary>
  <div style="color: #002676; padding-left: 10px;">
    Nine years ago when I first learned about Hockett's design features in a linguistics seminar as a first-year Cognitive Science PhD student, I had {{< sidenote "zero interest" >}}Perhaps because it was too much for me to talk about how to talk about language in a language I didn't grow up speaking. Might that be a lack of reflexive-reflexiveness LOL?{{< /sidenote >}} in linguistics. Three years ago when I started my career as a Machine Learning Engineer, I had zero interest in language models, dead set on becoming a recommender system expert — despite closely following OpenAI since 2016 while at Berkeley. It's funny how recognizing the parallels between language and recommender systems finally helped me see the magic (structure $\rightarrow$ learnability) in the former and the beauty (can it be potentially intelligent?) in the latter.
  </div>
</details>

# Break the Scaling Law Curse in RecSys
## Conjure Up Compositionality via Semantic IDs
Words in language and notes in music discretize sound waves, concepts, etc., which would've been atomic, continuous, and infinite. That's what Semantic IDs do for items in recommender systems. 

Residual-Quantized VAE (RQ-VAE) is the best known algorithm for Semantic ID generation. It was invented in the auditory ([Zeghidour et al., 2021](https://arxiv.org/abs/2107.03312)) and visual domains ([Lee et al., 2022](https://arxiv.org/abs/2203.01941)) and popularized by DeepMind's TIGER paper ([Rajput et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)) applying it to Generative Retrieval. RQ-VAE maps a pretrained semantic embedding $x$ to an $m$-level Semantic ID $(c_0, \ldots, c_{m-1})$ in a recursive manner: 

{{< figure src="https://www.dropbox.com/scl/fi/w9d3w2ra4uhueoftv41ex/Screenshot-2025-07-29-at-11.03.04-PM.png?rlkey=89zuzha8bxwh9lpl1n86l6cnv&st=5ox5ujza&raw=1" caption="RQ-VAE generates Semantic IDs by encoding input embeddings into quantized vectors and mapping the latter to $m$-level codes." width="1800">}}

1. Initialization
   - Encoder $\mathcal{E}$ maps input $x$ to latent representation $\mathbf{z} := \mathcal{E}(\mathbf{x})$ 
   - At each level $d$, initialize a new codebook $\mathcal{C}\_d := \\{ \mathbf{e}_k \\}\_{k=1}^{K}$ of size $K$; each code $k$ starts with embedding $\mathbf{e}_k$
     - Note: to prevent "codebook collapse" where most inputs are disproportionately mapped to a few codebooks, k-mean clustering can be used for codebook initialization
   
2. Level 0: Assign inputs to codes
   - The initial "residual" is $\mathbf{r}\_0 := \mathbf{z}$, the input latent vector
   - Conduct nearest neighbor search to find the code whose embedding is the closest to $\mathbf{r}\_0$, $c\_0 = \arg\min_i \lVert \mathbf{r}\_0 - \mathbf{e}\_k \rVert$
   - The difference between $\mathbf{r}\_0$ and the closet code embedding at level 0, $\mathbf{e}\_{c_0}$, becomes the next residual $\mathbf{r}\_1 := \mathbf{r}\_0 - \mathbf{e}\_{c_0}$ 
3. Recursion: Assign residuals to codes
   - At level $d$, the residual is $\mathbf{r}_d$ after assignment at $(d-1)$
   - Conduct nearest neighbor search to find the code whose embedding is the closest to $\mathbf{r}\_d$, $c\_d = \arg\min_i \lVert \mathbf{r}\_d - \mathbf{e}\_k \rVert$
   - The difference between $\mathbf{r}\_d$ and the closet code embedding at level $d$, $\mathbf{e}\_{c_d}$, becomes the next residual $\mathbf{r}\_{d+1} := \mathbf{r}\_d - \mathbf{e}\_{c_d}$ 

The process above can repeat infinitely. The deeper the codebooks, the finer the representations but the more the computation. After the latent vector $\mathbf{z}$ is allocated the Semantic ID $(c_0, \ldots, c_{m-1})$, its quantized representation $\mathbf{\hat{z}} := \sum\_{d=0}^{m=1}\mathbf{e}\_{c_i}$ is passed to a decoder to reconstruct the input $\mathbf{x}$ into $\mathbf{\hat{x}}$. RQ-VAE loss is defined as $\mathcal{L}(\mathbf{x}) := \mathcal{L}\_{\mathrm{recon}} + \mathcal{L}\_{\mathrm{rqvae}}$, where $\mathcal{L}\_{\mathrm{recon}} =\lVert \mathbf{x} - \mathbf{\hat{x}}\rVert^2$ measures how close the reconstructed output is to the original input and $\mathcal{L}\_{\mathrm{rqvae}} := \sum_{d=0}^{m-1} \lVert\mathrm{sg}[\mathbf{r_i}] - \mathbf{e}\_{c_i}\rVert^2 + \beta\lVert\mathbf{r_i} - \mathrm{sg}[\mathbf{e}\_{c_i}]\rVert^2$ ($\mathrm{sg}$ is stop-gradient) measures how good the code assignments are. Upon training, the RQ-VAE model uses the trained encoder to map the given item embedding into a quantized latent vector and generates a Semantic ID by assigning $m$-level codes to the quantized latent vector.

In a Semantic ID $(c_0, \ldots, c_{m-1})$, each code $c_d$ is like a character in a word. In natural languages, we usually don't tokenize at the character level --- even though the vocabulary will be manageable, the sequence length would be too long and it's hard for a character to have stable "meanings". Today, subword-level [tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) are most popular in NLP, which learn to bind frequently co-occurring characters into tokens. Famous examples include [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte-pair_encoding), [WordPiece](https://research.google/blog/a-fast-wordpiece-tokenization-system/), and [SentencePiece](https://discuss.huggingface.co/t/training-sentencepiece-from-scratch/3477). In TIGER, the authors used SentencePiece to learn Semantic ID tokenization, which performed better than naive unigram and bigram tokenization in downstream retrieval tasks.

In Generative Retrieval, a sequence of items $(\mathrm{item}\_1, \ldots, \mathrm{item}\_n)$ are converted into a sequence of codes using Semantic IDs $(c\_{1,0}, \ldots, c\_{1, m-1}, c\_{2,0}, \ldots, c\_{2, m-1},\ldots,c\_{n,0}, \ldots, c\_{n, m-1})$. To retrieve $\mathrm{item}_{n+1}$, we can use an encoder-decoder model to decode the next $m$ codes $(c\_{n+1,0}, \ldots, c\_{n+1, m-1})$, which is the Semantic ID of $\mathrm{item}\_{n+1}$ that hopefully maps to actual items in the corpus.

{{< figure src="https://www.dropbox.com/scl/fi/9detgaylbt0v4ed9kc1mp/Screenshot-2025-08-01-at-3.21.33-PM.png?rlkey=jxtcge2xxgy6kf3ztmjxe38y1&st=rnwuyfuc&raw=1" caption="Semantic IDs are not designed to map to human-readable categories, but map quite nicely to item taxonomies hand crafted by taxonomists." width="1800">}}

While there is no deliberate design, it's amazing how well Semantic IDs map to hand-crafted item taxonomies in the recommendation corpus. Of course, there are "bad" cases where different items share the same Semantic ID (hash collision) or a Semantic ID doesn't map to any actual items (invalid IDs). To prevent the former, an extra token can be appended to a Semantic ID to make it unique --- e.g., if two items share the same Semantic ID (12, 24, 52), we can assign (12, 24, 52, 0) to one and (12, 24, 52, 1) to the other. In cold-start scenarios, hash collisions can actually help retrieve new items that share Semantic IDs with existing items. Invalid codes are pretty rare, making up only 0.1\%-1.6% of all Semantic IDs generated in the TIGER paper.

{{< figure src="https://www.dropbox.com/scl/fi/a12lvlznxux8s7gwf4box/Screenshot-2025-08-01-at-3.53.58-PM.png?rlkey=ue8fm74yeyhcniu4o5gaq84x3&st=5ovirb0g&raw=1" caption="A high-level comparison between TIGER vs. COBORA. The latter combines sparse Semantic IDs and fine-grained item representations for Generative Retrieval." width="1800">}}

Information loss is inevitable when we discretize continuous representations --- imagine the finer-grained intervals lost between semitones in Western music. Baidu's COBORA ([Yang et al., 2025](https://arxiv.org/abs/2503.02453)) combines sparse (one-level Semantic IDs) and dense (real-valued embeddings) representations to depict each item in Generative Retrieval. In this hybrid framework, the sparse ID paints a coarse picture of the "item essence" (e.g., categories, brands, etc.) whereas the dense representation offers refinements on item details.

{{< figure src="https://www.dropbox.com/scl/fi/zyoay6335ztooy17fo39r/Screenshot-2025-08-01-at-4.46.06-PM.png?rlkey=h9oigl01bjuqej2kdw7e7t9nz&st=gxdrgcbr&raw=1" caption="In COBORA, the sparse ID is generated by RQ-VAE and the dense vector is contextual item embeddings after going through a BERT-style encoder." width="1800">}}

During training, a single-level (can be multi-level) Semantic ID is first decoded and converted into an embedding, which is then appended to the input embedding sequence to decode the dense representation, $P(ID\_{t+1}, \mathbf{v}\_{t+1}|S_{1:t}) = P(ID\_{t+1}|S_{1:t})P(\mathbf{v}\_{t+1}|ID\_{t+1},S_{1:t})$. The loss function is given by $\mathcal{L} = \mathcal{L}\_\mathrm{sparse} + \mathcal{L}\_\mathrm{dense}$, where:

- $\mathcal{L}\_\mathrm{sparse} = -\sum\_{t=1}^{T-1}\log(\frac{\exp(z\_{t+1}^{ID\_{t+1}})}{\sum\_{j=1}^C\exp(z\_{t+1}^j)})$, which is the standard cross-entropy loss
- $\mathcal{L}\_\mathrm{dense} = -\sum\_{t=1}^{T-1}\log(\frac{\exp(\cos(\hat{\mathbf{v}\_{t+1}}\cdot \mathbf{v}\_{t+1}))}{\sum\_{\mathrm{item}_j}\in\mathrm{Batch}\exp(\cos(\hat{\mathbf{v}\_{t+1}}\cdot \mathbf{v}\_{\mathrm{item}_j}))})$, which pushes the cosine similarity between positive pair embeddings to be greater than that between negative pair embeddings

During inference, only the dense representation is used for retrieval via embedding-based retrieval (see {{< backlink "ebr" "my post">}}). COBORA outperforms methods that only have the sparse or the dense components.

It's crazy how fast moving the Generative Recommendation field is --- minutes after I wrote the paragraph above and scrolled LinkedIn, Snap open-sourced their framework called Generative Recommendation with Semantic ID (GRID). I plan to read their [code](https://github.com/snap-research/GRID) and [paper](https://arxiv.org/pdf/2507.22224) in details --- at first glance, it looks like TIGER with additional user tokens.

## Crank Up Task Complexity via Generative Training

For language models, scale often does wonders for model performance. The seminal *Attention Is All You Need* paper used 6 Transformer blocks, GPT-2 scaled it up to 12, and GPT-3 to 96. While GPT-4's number of Transformer blocks is undisclosed, [rumor](https://semianalysis.com/2023/07/10/gpt-4-architecture-infrastructure/) has it that it uses 120 layers and has 100x more parameters than its predecessor GPT-3 (18 trillion vs. 175 billion). The data and compute used to train larger models have also increased manyfold.

In Recommender Systems, Transformer blocks are often used in user sequence modeling (see {{< backlink "seq_user_modeling" "my post">}} for an overview of this domain) to weight historically engaged items (via self-attention or target attention), and the output is typically plugged into a DRLM model for discriminative  `pAction`  predictions. As mentioned earlier, many companies (Chen et al., [KDD 19](https://dl.acm.org/doi/abs/10.1145/3326937.3341261), Gu et al., CIKM [20](https://dl.acm.org/doi/abs/10.1145/3340531.3412697), [21](https://dl.acm.org/doi/abs/10.1145/3459637.3481953)) found the best performance when using just one Transformer layer.

Turns out the generative vs. discriminative task setup makes a huge difference in the emergence of scaling laws. The WeChat team ([Zhang et al., 2025](https://dl.acm.org/doi/abs/10.1145/3640457.3688129)) trained a *stand-alone* sequence model using a pure item ID sequence to predict the next item ID. They scaled up the model from 98.3K to 0.8B parameters by stacking Transformer blocks. 

{{< figure src="https://www.dropbox.com/scl/fi/yf6frvifnlg8g8pj1bumc/Screenshot-2025-08-01-at-11.28.00-PM.png?rlkey=1iwin5iu1xc782vejo6fw3p7a&st=i4iwezql&raw=1" caption="In the scaling law study, WeChat trained a simple autoregressive sequence model with only item IDs and scaled it up by stacking Transformer blockers." width="1800">}}

They saw power-law curves where model performance increases with both data size and model size. Larger models are more "data efficient," in that validation loss drops more quickly with data size in larger models. WeChat models differ from aforementioned sequence models that don't scale in that they perform an end-to-end generative task (i.e., decoding the next ID), whereas the latter perform a discriminative task (i.e., predicting whether a user will act on an item) in the end.

{{< figure src="https://www.dropbox.com/scl/fi/6fwyco3xq3l8pjf5nax2h/Screenshot-2025-08-02-at-10.16.24-AM.png?rlkey=v9g0wv1swml28ft9stkcs5640&st=wfmb5rf2&raw=1" caption="Scaling laws were observed where model performance increases both with model size and data size. Larger models improves faster with data size." width="1800">}}

What makes generative vs. discriminative tasks different? In terms of task formulation, both `pAction` prediction and next-token prediction solve classification problems --- the former is a binary classifier (i.e., predicting whether a user will act on an item or not given some context) and the latter is an extreme multi-class classifier (e.g., selecting the most likely or top-k like tokens; checkout AI Coffee Break's awesome [video](https://www.youtube.com/watch?v=o-_SZ_itxeA) on decoding strategies). What makes `pAction` prediction discriminative is that it models the conditional probability $P(\mathrm{action} | \mathrm{user},\,\mathrm{item},\,\mathrm{context})$. By contrast, next-token prediction requires modeling the joint distribution $P(x_1,x_2,\ldots,x_n,x_{n+1})=P(x_{n+1})P(x_1,x_2,\ldots,x_{n}|x_{n+1})$. Learning the joint distribution is simultaneously a harder task but it also enjoys more supervision --- all preceding tokens can supervise the current token generation, whereas for `pAction` predictions, each example only provides a single binary label. As such, generative models need more data to train and they also get what they wish for. 

In his Zhihu [blogpost](https://zhuanlan.zhihu.com/p/1918350919508140128), the first author of Kuaishou's [OneRec technical report](https://arxiv.org/abs/2506.13695) offers deeper thoughts on the discriminative vs. generative distinction --- below are English translations of the original text: 

> In my view, a large language model's ability to think and reason largely stems from the fact that increasing the decoding steps corresponds to increasing the depth of search. If we think of the decoding process as a kind of 100K-token tree search at each step, then generating more tokens means going deeper in the search tree. And each generated token becomes part of the future input, helping the model activate more relevant information from its parameters (via the causal transformer) to make judgments about the final answer.

> In recommendation models, however, expanding the candidate set is fundamentally a breadth-based search, not a depth-based one. Each item is evaluated independently, so evaluating more items doesn't help the model make a better judgment about any single one. There's a performance ceiling.

In the past year, companies like Kuaishou and Meta made a bold move, ditching the traditional DLRM paradigm for end-to-end generative Generative Recommendation. More companies such as Alibaba, Netflix, and Xiaohongshu integrated GR components into their DLRM systems. In the next section, let's look at several famous case studies. 

# Approaches to Generative Recommendation

## Fully Generative Architectures

### Meta's HSTU

Meta's [*Actions Speak Louder than Words*](https://arxiv.org/abs/2402.17152) may be the Recommender System paper with the most Greek letters you'll ever see, but the core ideas are straightforward: <span style="background-color: #D9CEFF">Retrieval can be framed as a next-item prediction problem and ranking a next-action prediction problem </span>. 

{{< figure src="https://www.dropbox.com/scl/fi/9s8qs8alvl5s5qwdg24vo/Screenshot-2025-08-02-at-1.54.07-PM.png?rlkey=dq12i5j8tgoa200zpfyar1ynn&st=nqp621ua&raw=1" caption="Meta has 'completely' overhauled the DLRM framework, but many components in DLRM either map directly to GR or after some tweaks." width="1800">}}

{{< figure src="https://www.dropbox.com/scl/fi/nyyma8ul631e45kw7t87s/Screenshot-2025-08-02-at-2.23.22-PM.png?rlkey=01es2kkiy5y87qylat9q8feu7&st=o5e58nl2&raw=1" caption="Ranking vs. retrieval task framing in the Generative Recommendation." width="600">}}

In their notations, $\Phi\_i$ denotes a content and $a\_i$ denotes an action on the content. The total number of contents a user has engaged with is $n_c$. A user sequence is given as $(\Phi\_0, a\_0, \Phi\_1, a\_1,\ldots,\Phi\_{n_c-1}, a_{n\_c-1})$. The retrieval task is to predict the most likely next content $\Phi\_{i+1}$ given user history $u\_i$, $\arg\max\_{\Phi\in\mathcal{X}_c} p(\Phi\_{i+1}|u\_i)$. Only positive actions serve as supervision for next-content predictions, meaning that $u_i = (\Phi\_1',\Phi\_2',\ldots,\Phi\_{n_c-1}')$, where $\Phi_i'=\Phi_i$ if $a_i$ is positive and empty $\emptyset$ otherwise. In ranking, the next item is given, so the task is to predict the next-action probability, $P(a\_{i+1}|\Phi\_0,a\_0, \Phi\_1,a\_1,\ldots,\Phi\_{i+1})$.

The main module in Meta's model is a causal autoregressive Transformer called the "Hierarchical Sequential Transduction Unit" (HSTU). Like regular Transformers (see {{< backlink "attention_as_dict" "my post">}} for a refresher), HSTU consists of 3 sub-layers, but each comes with modifications:

{{< figure src="https://www.dropbox.com/scl/fi/5at8fwvcl16cztms1ipdv/Screenshot-2025-08-02-at-1.51.58-PM.png?rlkey=zjtvd90ox2mbv5eiav8w69526&st=gt68a43z&raw=1" caption="HSTU is a modified causal autoregressive Transformer." width="1800">}}

- Pointwise projection: 4 linear projections are created from the input. On top of $Q, K, V$, HSTU also has gating weights $U$ ---
  - $U(X), V(X), Q(X), K(X) = \mathrm{Split}(\phi_1(f_1(X)))$
- Spatial aggregation: Similar to regular Transformers, HSTU uses attention scores to pool $V(X)$ to create a "contextual embedding" for each token. Instead of positional encodings, HSTU uses the relative attention bias ([Raffel et al., 2020](https://arxiv.org/html/2402.17152v3#bib.bib43)) that incorporates positional ($p$) and temporal ($t$) information
  - $A(X)V(X) = \phi_2(Q(X)K(X)^T + \mathrm{rab}^{p,t})V(X)$
- Pointwise aggregation: This is perhaps the most special thing about HSTU. Regular self-attention uses row-wise softmax to normalize attention scores across the entire sequence. By contrast, HSTU normalizes each attention score independently, replacing softmax with a nonlinear activation function like SiLU. As a result, attention scores in each row may not sum up to 1. Finally, gating weights $U(X)$ control which dimensions in each token embedding matter more and modulate them accordingly
  - $Y(X) = f\_2(\mathrm{Norm}(A(X)V(X)\odot U(X)))$

While the overall architecture isn't too crazy, many optimizations are done to bring this model to life, including selecting a subsequence of *stochastic length* (SL) from each user's history (TL;DR: the older an action, the less likely it will be selected), row-wise AdamW optimizers, and the new M-FALCON algorithm for micro-batching during serving.

### Kuaishou's OneRec

Kuaishou's OneRec brings all-around improvements to HSTU, most notably Kuaishou's own RQ-Kmeans algorithm ([Luo et al., 2024](https://arxiv.org/abs/2411.11739)) that incorporates collaborative signals into Semantic IDs and maximizes codebook utilization, session-wise list generation that generates a list of interdependent videos rather than unrelated individual videos, and a post-training preference alignment stage that uses [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) to optimize for conflicting business objectives. The paper ([Deng et al., 2025](https://arxiv.org/abs/2502.18965)) itself is pretty short. A more detailed technical report ([Zhou et al., 2025](https://arxiv.org/abs/2506.13695)) came out a few months later and has made rounds in the recommendation industry. 

{{< figure src="https://www.dropbox.com/scl/fi/v0j337owy5w3k0eewczl9/Screenshot-2025-08-02-at-1.50.08-PM.png?rlkey=722s4amolmzdk9pbo4mwhpkc9&st=whfld5n7&raw=1" caption="OneRec pretrains an encoder-decoder model that takes user sequences as input and outputs Semantic IDs for a list of videos. In post-training preference alignment, responses are rewarded for model/business metric gains and having a valid format." width="1800">}}

#### *Semantic ID Generation*
Each video's content embedding is first obtained by passing multimodal inputs (e.g., texts: captions, tags, speech-to-text, image-to-text; images: the cover image, 5 uniformly sample frames) into a visual-language encoder, which, in this case, is miniCPM-V-8B. For efficiency, the authors used QFormer from BLIP-2 ([Li et al., 2023](https://arxiv.org/abs/2301.12597)) to compress 1280 video tokens into 4 query tokens. To inject collaborative signals into content embeddings, the authors sampled item pairs engaged by similar users ("collaboratively similar") and used contrastive learning to align representations of such videos. 

RQ-VAE initializes codebooks randomly, often resulting in many residuals being mapped to the same codebook (the "hourglass phenomenon", [Kuai et al., 2024](https://arxiv.org/abs/2407.21488)). To ensure full codebook utilization, RQ-Kmeans constructs codebooks on the fly by clustering residuals using K-means. Coarse-to-fine Semantic IDs generated by RQ-Kmeans, $\\{s\_m^1, s\_m^2,\ldots,s\_m^{L\_t}\\}$, are the "vocabulary" in OneRec.

#### *Encoder-Decoder Training*
The main module in OneRec is a T5-style ([Roberts et al., 2020](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/)) encoder-decoder architecture. Rather than encoding a single user sequence like HSTU does, OneRec encodes 4 user sequences:
- User static pathway: Static user features, such as ID, gender, age, etc. --- $[\mathbf{e}\_{\mathrm{uid}};\mathbf{e}\_{\mathrm{gender}};\mathbf{e}\_{\mathrm{age}};\ldots]$, where $\mathbf{e}\_{\mathrm{uid}};\mathbf{e}\_{\mathrm{gender}};\mathbf{e}\_{\mathrm{age}} \in \mathbb{R}^{64}$

- Short-term pathway: Most recent (e.g., 20) engagements, each characterized by $[\mathbf{e}\_{\mathrm{vid}}^s;\mathbf{e}\_{\mathrm{aid}}^s;\mathbf{e}\_{\mathrm{tag}}^s;\mathbf{e}\_{\mathrm{ts}}^s;\mathbf{e}\_{\mathrm{playtime}}^s,\mathbf{e}\_{\mathrm{dur}}^s;\mathbf{e}\_{\mathrm{label}}^s]$; embedding dimensions vary between different entities
- Positive-feedback pathway: Most engaged (e.g., 256) items, each characterized by $[\mathbf{e}\_{\mathrm{vid}}^p;\mathbf{e}\_{\mathrm{aid}}^p;\mathbf{e}\_{\mathrm{tag}}^p;\mathbf{e}\_{\mathrm{ts}}^p;\mathbf{e}\_{\mathrm{playtime}}^p,\mathbf{e}\_{\mathrm{dur}}^p;\mathbf{e}\_{\mathrm{label}}^p]$
- Lifelong pathway: Up to 100,000 past engaged videos. Kuaishou's own TWIN V2 ([Si et al., 2024](https://arxiv.org/html/2407.16357v1)) is used to compress this sequence into $K$ clusters created with hierarchical K-means clustering. Each compressed token representation a cluster is characterized by $[\mathbf{e}\_{\mathrm{vid}}^l;\mathbf{e}\_{\mathrm{aid}}^l;\mathbf{e}\_{\mathrm{tag}}^l;\mathbf{e}\_{\mathrm{ts}}^l;\mathbf{e}\_{\mathrm{playtime}}^l,\mathbf{e}\_{\mathrm{dur}}^l;\mathbf{e}\_{\mathrm{label}}^l]$

Outputs from the 4 pathways are projected into $\mathbf{h}\_u \in \mathbb{R}^{1\times d\_{\mathrm{model}}}$, $\mathbf{h}\_s \in \mathbb{R}^{L\_{s}\times d\_{\mathrm{model}}}$, $\mathbf{h}\_p \in \mathbb{R}^{L\_{p}\times d\_{\mathrm{model}}}$, and $\mathbf{h}\_l \in \mathbb{R}^{N\_{q}\times d\_{\mathrm{model}}}$ (QFormer further compresses the already compressed sequence of length $L\_l$ into $N\_q$ query tokens), respectively. The 4 outputs are concatenated along the sequence length dimension, with a positional encoding $\mathbf{e}\_{\mathrm{pos}} \in \mathbb{R}^{(1+L\_s + L\_p + N\_q)}$ added, before being fed into the encoder. 

The decoder generates Semantic IDs for a series of videos, each starting with a special beginning-of-sentence token --- example Semantic ID of the $m$-th video, $\mathcal{S}\_m = \\{s\_{[\mathrm{BOS}]},s\_m^1,s\_m^2,\ldots,s\_m^{L\_t}\\}$. A standard cross-entropy loss is used for next-token prediction. A nuanced is that OneRec applies cross-attention between the encoder output and the latest state of the decoder, so that the decoder attends to both previously decoded tokens (causal self-attention) and the user's entire history (cross-attention). A sparse MoE layer is used in the feed-forward network to for training efficiency and cost reduction.

#### *Preference Alignment via DPO*

In Machine Learning Engineering, half the work is to build a fancy model and half is to persuade others why the new predictions are better. For instance, do users find treatment results more "relevant" (however defined)? Do they click on more ads? Do they have longer sessions and engage with the content more deeply (e.g., commenting, sharing)?... In DRLM, each objective can be optimized by a task tower and outputs from different towers are combined into a weighted sum.

{{< figure src="https://www.dropbox.com/scl/fi/66etdouoq1m6vdfol7chh/Screenshot-2025-08-02-at-10.24.57-PM.png?rlkey=vzsxzhbubgwqhst2odrrfdfmf&st=7hoecdhu&raw=1" caption="After training a 'seed model', cross-entropy loss and DPO loss (based on `pAction`, format, and business needs) are both used during preference alignment." width="1800">}}

In OneRec, the score combining different objectives is called the P-Score (Preference Score), which serves as one of the rewards in the preference alignment stage. Other rewards include a format reward (e.g., whether the generated Semantic IDs are valid) and an "industrial reward" (e.g., safety, monetization, diversity, cold-start). A variant of DPO, called ECPO (Early Clipped [GRPO](https://arxiv.org/abs/2402.05749)), is used to optimize these rewards. Different from GRPO, ECPO uses a clipped policy gradient objective to stabilize training in early stages. To maximize alignment, OneRec uses an Iterative Preference Alignment (IPA) strategy: at each iteration, beam search generates multiple candidates, the reward model selects the best and worst responses, and both standard next-token prediction loss and DPO loss are used to update the model.

Reading the OneRec technical report reminds me of how I felt when reading the DeepSeek reports in February: so much thought given to every detail, loads of engineering ingenuity, and truly "no guts, no glory". The lead of OneRec, Guorui Zhou, wrote in his [post](https://zhuanlan.zhihu.com/p/1918350919508140128) that his motivation came from both practical concerns (e.g., reducing compute and overhead costs by eliminating L0 $\rightarrow$ L1 $\rightarrow$ L2 data transfer to increase the profit margin of ads) and a profound intellectual quest (e.g., how "intelligence" can emerge and why it has long evaded recommender systems). I hope that, in my own career, I'll have the honor of being part of a revolution sparked by early conviction.

### Meituan's MTGR

In the middle of writing this blogpost, I had dinner with a former colleague and described the idea from the TIGER paper to her. Her first reaction was, "Eh, how do we personalize recommendations?"

That was probably the catch with Generative Recommendation --- in DLRM, user-item cross features (e.g., how often a user clicked through to an ad or purchased an item) often have the highest feature importances and are key to personalization, but it's not immediately apparent how this kind of information can be incorporated into a Generative Recommender (e.g., which Semantic IDs should be decoded probably doesn't depend on who I am). Meituan's MTGR ([Han et al., 2025](https://arxiv.org/abs/2505.18654))  brings cross features back into GR.

{{< figure src="https://www.dropbox.com/scl/fi/959t68kyyq1hux8vwgy8q/Screenshot-2025-08-02-at-1.56.05-PM.png?rlkey=f4oyy0av007kaotk8lmmsoaqu&st=7sepacbq&raw=1" caption="MTGR organizes data at the user-item level, allowing user-item interactions to be added as candidate item features." width="1800">}}

Typically in DRLM, if a user interacted with $N$ items for a combined total of $M$ times in a time window, it results in $M$ rows of training data. Typically in GR, $M$ interactions are organized into one  sequence. In MTGR, each user-item pair has one sequence, resulting in $N$ rows of data: $\mathbb{D} = [\mathbf{U}, \mathbf{\overrightarrow{S}}, \mathbf{\overrightarrow{R}}, [\mathbf{C}, \mathbf{I}]\_1\ldots, [\mathbf{C}, \mathbf{I}]\_K]$. Under this user-item level data arrangement, cross features (e.g., user-item CTR) are treated as features of the candidate (the interacted item). In each sequence, the model can predict for multiple occurrences of the same candidate at once, reducing training costs. To avoid information leakage, dynamic masking is added so that user features $(\mathbf{U},  \mathbf{\overrightarrow{S}})$ are visible to all tokens, real-time interactions $\mathbf{\overrightarrow{R}}$ are only visible to later tokens, whereas candidate tokens $(\mathbf{C},\mathbf{I})$ are only visible to themselves.

Comparing with GR, MTGR sacrifices some efficiency for better personalization --- by making $N$ times more predictions than a typical GR, it can now include user-item cross features in predictions.

## Hybrid Generative-Discriminative Architectures

Rather than overhauling DLRM and existing cascade pipelines and team structures (e.g., most companies have separate retrieval/L1/L2 teams), more companies chose to integrate components of Generative Recommendation into DRLM to enjoy the best of both worlds. 

### Alibaba's GPSD

As alluded to earlier, DLRM is haunted by a strange "one-epoch phenomenon", where model performance on the test set suddenly drops at the beginning of the second epoch ([Zhang et al., 2022](https://arxiv.org/abs/2209.06053)). As a result, you can't train a DLRM model for longer to improve it.

{{< figure src="https://www.dropbox.com/scl/fi/yq7hf8i736jjjfbxtip0n/Screenshot-2025-08-03-at-11.46.02-AM.png?rlkey=6h1h4r7gvtonslost65kujw8e&st=g2dzfqco&raw=1" caption="A possible explanation for the one-epoch phenomenon." width="500">}}

A common hypothesis is that the joint distribution of trained samples, $\mathcal{D}(\mathrm{EMB}(\mathbf{x}\_{\mathrm{trained}}), y)$, differs significantly from that of untrained samples, $\mathcal{D}(\mathrm{EMB}(\mathbf{x}\_{\mathrm{untrained}}), y)$. Before training, embeddings are at their initial values. After one epoch, embeddings for training samples are updated, allowing the model to overfit to $\mathcal{D}(\mathrm{EMB}(\mathbf{x}\_{\mathrm{trained}}), y)$ at the start of the second epoch. In contrast, test samples still have $\mathcal{D}(\mathrm{EMB}(\mathbf{x}\_{\mathrm{untrained}}), y)$ since their embeddings remain unchanged. As a result, the model performs poorly on the test set. Due to their sparsity, high-cardinality ID embeddings are easiest to overfit, since each ID appears infrequently in the data. In contrast, low-cardinality features are seen more often and are less prone to overfitting.

To address this issue, Alibaba's GPSD ([Wang et al., 2025](https://arxiv.org/abs/2506.03699)) pretrains ID embeddings in a separate generative model (think Transformers + next-token prediction tasks) and integrates learned ID embeddings into the downstream discriminative CTR model. Another Chinese e-commerce giant JD.com follows the same generative-to-discriminative transfer strategy in their GenCTR model ([Kong et al., 2025](https://arxiv.org/abs/2507.11246)).

{{< figure src="https://www.dropbox.com/scl/fi/nye4quw0lsu8ukts6dwfh/Screenshot-2025-08-03-at-11.56.20-AM.png?rlkey=x3101ljcjgl3ibthbll4oinkm&st=sfvui9ba&raw=1" caption="To mitigate the one-epoch phenomenon, GPSD pretrains a generative model on user sequences and transfers parameters to a downstream discriminative model." width="1800">}}

The authors compared 4 ways to integrate pretrained embeddings into discriminative CTR models (baseline: train everything from scratch):
- *Full Transfer*: Transfer sparse + dense parameters from the generative model to the discriminative model; allow sparse + dense parameter updates during CTR model training.
- *Sparse Transfer*: Transfer sparse parameters and train dense parameters from scratch; allow sparse parameter updates.
- *Full Transfer & Sparse Freeze*: Transfer sparse + dense parameters; freeze sparse parameters but allow dense parameters updates.
- *Sparse Transfer & Sparse Freeze*: Transfer sparse parameters; freeze sparse parameters and train dense parameters from scratch.

{{< figure src="https://www.dropbox.com/scl/fi/9thmlc2l4tx83ctcitnst/Screenshot-2025-08-03-at-12.47.45-PM.png?rlkey=nkoqyoj3uiw901sxn4mah852t&st=nv5pll27&raw=1" caption="GPSD breaks the one-epoch phenomenon and the scaling law curse, with model performance improving with training data and model size." width="1800">}}

As one can see, the one-epoch curse is broken --- all methods allowed model performance to improve after one epoch. Moreover, scaling laws based on model size have emerged --- the larger the model (L4H256A4 > L4H128A4 > L4H64A4 > L4H32A4), the better the overall performance. In the two smaller models (32 and 64), "Sparse Transfer & Sparse Freeze" worked the best whereas in the two larger models (128 and 256), "Full Transfer & Sparse Freeze" was the best.

Besides GPSD, Alibaba have several other Generative Recommenders, such as LUM ([Yan et al., 2025](https://arxiv.org/abs/2502.08309)) and URM ([Jiang et al., 2025](https://arxiv.org/abs/2502.03041)). Just yesterday (August 2, 2025), Alibaba published the [RecGPT Technical Report](https://huggingface.co/papers/2507.22879), showing their ambition to turn recommendations into a ChatGPT-like product that truly interacts with users intelligently. 

### Xiaohongshu's RankGPT

There are good, bad, and great recommender systems --- and then there's [Xiaohongshu](https://www.xiaohongshu.com/explore), the only app powerful enough to dominate our lives. A restaurant can go from the brink of shutting down to having a two-hour dinner queue overnight, all for a post on Xiaohongshu. My friends and I constantly joke that no matter how unique or personal an experience feels, a post echoing that exact thought or sentiment will show up in our feed 5 minutes later. RankGPT ([Huang et al., 2025](https://arxiv.org/abs/2505.04180)) is the {{< sidenote "mastermind" >}}RankGPT doesn't seem like a huge innovation from Meta's HSTU. Perhaps as a wise person once said, data is the king in recommendations.{{< /sidenote >}} behind Xiaohongshu's uncanny recommendations.

{{< figure src="https://www.dropbox.com/scl/fi/vwxgmdxd24zxtaw50p2am/Screenshot-2025-08-03-at-2.10.20-PM.png?rlkey=bf9auxoysas4k19k6ustszkw9&st=upm449ip&raw=1" caption="RankGPT is similar to HSTU, but organizes training data around actions and treats items as positional information." width="1800">}}

RankGPT's architecture is modified from HSTU. HSTU assigns 2 tokens to each engagement, a content token $\Phi\_i$ and an action token $a\_i$. The upside is that HSTU unifies retrieval and ranking --- retrieval is achieved by predicting the next content and ranking is achieved by predicting the next action. The downside is that the user sequence is now doubled in length, which then quadruples the required compute. 

{{< figure src="https://www.dropbox.com/scl/fi/bt0tc3aktaatsushbxseb/Screenshot-2025-08-03-at-2.33.47-PM.png?rlkey=ai3d5zj95pmxscbvckih23ph2&st=0a59pdz8&raw=1" caption="Turns out all 3 positional encodings are not as good as parameter-free ALiBi." width="1800">}}

Since RankGPT isn't designed to handle retrieval, there's no need to interleave separate content and action tokens --- it only has to predict actions on candidates. So item and action embeddings can be added together to represent engagement $i$, $e\_i = \varphi_i + \phi\_i$, where $\varphi(\cdot)$ and $\phi(\cdot)$ denote item and action embedding modules, respectively. To prevent information leakage, candidate action tokens are masked. The authors call this the "action-oriented organization". 

As for injecting positional information, the authors originally explored adding 3 forms of positional encodings to token embeddings:
- *Position embeddings*: A learnable positional embedding based on the item index in the user sequence.
- *Request index embeddings*: A learnable positional embedding based on the request index --- all items belonging to the same request share the same embedding.
- *Pre-request time embeddings*: A learnable positional embedding capturing the bucketed timestamp difference between each action timestamp and the request timestamp. 

Interestingly, all those methods combined didn't beat ALiBi ([Press et al., 2021](https://arxiv.org/abs/2108.12409)), which simply replaces positional encodings with a parameter-free penalty that increases with query-key distances. Compared to HSTU, the action-oriented organization decreased AUC by 0.03%, while ALiBi improved AUC by 0.09%. Overall, RankGPT achieved a 94.8% speed-up with a net AUC gain of 0.06%.

{{< figure src="https://www.dropbox.com/scl/fi/dbuzokx2ycz7po0pmjzio/Screenshot-2025-08-03-at-2.10.26-PM.png?rlkey=nm23z4yhi1fcnret94cq7e1l8&st=sc8aj0wd&raw=1" caption="RankGPT offers a huge speed-up compared to HSTU with moderate AUC gains." width="500">}}

### Foundation Models at Netflix and Pinterest
Apart from Google's TIGER and Meta's HSTU, most models above come from Chinese companies. In the United States, Netflix's Foundation Model ([Hsiao et al., 2025](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)) and Pinterest's PinFM ([Chen et al., 2025](https://arxiv.org/abs/2507.12704)) are among the better known models. Rather than making end-to-end recommendations as online models, they seem more like offline models that generate features for online `pAction` models. 

As with most Generative Recommenders, Netflix's Foundation Model is a Transformer-based autoregressive model trained on user sequences to predict the next token(s). To speed up training, it uses sparse attention (the [blogpost](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) doesn't say which one) and compresses similar movies in a row into the same token. To drive life-term satisfaction, the model predictions the next-$n$ tokens rather than just the next one (reminiscent of the "Dense All Action" loss in Pinterest's PinnerFormer, [Pancha et al., 2022](https://arxiv.org/abs/2205.04507)). To enhance generalization, the model not only predicts item IDs but also metadata (e.g., genre, tone). This model is integrated into downstream (discriminative) models.

{{< figure src="https://www.dropbox.com/scl/fi/pe7jpmw5qocir4xiamvg7/Screenshot-2025-08-03-at-3.46.38-PM.png?rlkey=ksseh27dyje6caag7pe60r1m3&st=8dc7hmmx&raw=1" caption="Netflix compresses similar movies into a single token." width="1800">}}

Pinterest's PinFM ([Chen at al., 2025](https://arxiv.org/abs/2507.12704)) is similar to Netflix's model, which is an autoregressive Transformer trained on raw user sequences to predict multiple future tokens, with innovations here and there. During pretraining, the model uses 2 years of engagement history up to length 16,000 and only the low-dimension features (basically excluding pretrained item embeddings). The loss consists of 3 parts:
- Next-token loss: $\mathcal{L}\_{ntl} = \sum\_{i+1}^{m-1}l(\mathbf{H}\_i, z\_{i+1})\mathbb{1}[a\_{i+1} \in A\_{\mathrm{pos}}]$, infoNCE loss for predicting the next positively engaged item.
- Multi-token loss: $\mathcal{L}\_{mtl} = \sum\_{i+1}^{m-1}\sum\_{j=i+1}^{i+L'}l(\mathbf{H}\_i, z\_j)\mathbb{1}[a\_j \in A\_{\mathrm{pos}}]$, loss for predicting positively engaged item in window $L'$.
- Future-token loss: $\mathcal{L}\_{ftl} = \sum\_{j=n+1}^{n+L'}l(\mathbf{H}\_{L\_{d}}, z\_j)\mathbb{1}[a\_j \in A\_{\mathrm{pos}}]$ --- during serving, a shorter sequence of length $L\_d$ is sent to the model; this loss penalizes prediction errors within $L\_d$.

{{< figure src="https://www.dropbox.com/scl/fi/1478ofq5zfpzsohx8kzrk/Screenshot-2025-08-03-at-4.04.35-PM.png?rlkey=tdjex9ls02vwdhh4ff9dklw8o&st=2bb8lpa4&raw=1" caption="PinFM is pretrained on lifelong user sequences and fine-tuned in ranking models." width="1800">}}

In downstream ranking models, the candidate item is appended to the user sequence as the input to PinFM to bring "candidate awareness". Rather than freezing model parameters (like some variants of Alibaba's GPSD), PinFM is fine-tuned along with `pAction` predictions.

To productionize PinFM, the team took aggressive cost-saving measures. For instance, to reduce large embedding table costs, they applied int4 quantization. To avoid duplicate computations during serving (# of unique users $\ll$ # of candidates), they developed the Deduplicated Cross-Attention Transformer (DCAT), computing candidate-independent user sequence representations once for each unique user and broadcasting them to the same users in the batch to perform user-candidate feature crossing. Despite deploying a 20B model, cost and latency increases were almost neutral.

# Lessons from Case Studies
1. *Scaling laws don't emerge in one way*: Poor learning of sparse ID features seems to be the reason why DLRM models struggle to scale, but there isn't just one way to unlock scaling laws. You can reframe the entire ranking problem as a generative task (e.g., next-action prediction), or you can pretrain ID embeddings in a generative model (usually an autoregressive Transformer) and transfer the parameters to a downstream discriminative model.
2. *Sequence compression is important*: Training a Generative Recommendation model usually requires data on each user's lifelong history. As we all know, for Transformers, both time and space complexity scale quadratically with sequence length (see {{< backlink "hardware_aware_transformers" "my post" >}} for more analysis). All the models we've seen apply some form of sequence compression --- e.g., Kuaishou combines TWIN V2 + QFormer, Meta uses a rather rough stochastic length sampling, Netflix merges similar tokens that appear in a row, and so on. Finding an efficient and lossless compression method will greatly improve both efficiency and data quality for GR training.
3. *Personalization may be the next frontier*: Personalization usually means knowing "for this particular user, how good is this item", which is naturally captured by user-item interaction features (e.g., aggregated CTR, action counts, etc. at the user-item level) and is readily usable in DLRM models. In Generative Recommenders, it's much more awkward to plug in these features. Meta's HSTU got rid of them; Meituan's MTGR had to expand the training data $N$-fold ($N$ being the number of candidate items to rank in a given time window). How to more efficiently utilize personalized features will be an interesting topic to explore.

# References

## Overview & Scaling Laws in Recommender Systems
1. A comprehensive lit review on Generative Recommendation 👉 [*GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models*](https://arxiv.org/abs/2507.06507) (2025) by Yang et al., *arXiv*.
2. "One-epoch phenomenon" 👉 [*Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Prediction Models*](https://arxiv.org/abs/2209.06053) (2022) by Zhang et al., *CIKM*.
3. Quality saturation under the "item-centric ranking" framework 👉 [*Breaking the Curse of Quality Saturation with User-Centric Ranking*](https://arxiv.org/abs/2305.15333) (2023) by Zhao et al., *KDD*.
4. Netflix foresaw the lack of task complexity and item ID compositionality in DLRM 👉 [*Deep Learning for Recommender Systems: A Netflix Case Study*](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/18140) (2021) by Steck et al., *AI Magazine*.
5. Power-law scaling typically hits diminishing returns in DLRM 👉 [*Understanding Scaling Laws for Recommendation Models*](https://arxiv.org/abs/2208.08489) (2022) by Ardalani et al., *arXiv*.
6. Generative training even on pure IDs leads to power-law scaling laws 👉 [*Scaling Law of Large Sequential Recommendation Models*](https://dl.acm.org/doi/abs/10.1145/3640457.3688129) (2025) by Zhang et al., *RecSys*.

## From Atomic Item IDs to Semantic IDs
7. RQ-VAE, the most popular technique for learning Semantic IDs 👉 initially invented to generate audios ([Zeghidour et al., 2021](https://arxiv.org/abs/2107.03312)) and images ([Lee et al., 2022](https://arxiv.org/abs/2203.01941)) with low costs and high fidelity
8. Google DeepMind's TIGER ([Rajput et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)) applied RQ-VAE to learning semantic IDs and used them in retrieval 👉 another Google paper ([Singh et al., 2024](https://dl.acm.org/doi/abs/10.1145/3640457.3688190)) applied Semantic IDs to ranking
9. Baidu's COBRA tackled information loss in RQ-VAE by also generating dense representations 👉 [*Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations*](https://arxiv.org/abs/2503.02453) (2025) by Yang et al., 2025, *arXiv*.
10. Kuaishou's QARM used RQ-Kmeans to maximize codebook utilization 👉 [*QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou*](https://arxiv.org/abs/2411.11739) (2024) by Luo et al., *arXiv*.
11. Snap's GRID 👉 [*Generative Recommendation with Semantic IDs: A Practitioner's Handbook*](https://www.arxiv.org/abs/2507.22224) (2025) by Ju et al., *arXiv*.

## Ditch DLRM for End-to-End Generative Architectures
12. Meta's HSTU 👉 [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152) (2024) by Zhai et al., *ICML*.
13. Kuaishou's OneRec 👉 [*OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*](https://arxiv.org/abs/2502.18965) (2025) by Deng et al., *arXiv*.
14. Meituan's MTGR 👉 [*MTGR: Industrial-Scale Generative Recommendation Framework in Meituan*](https://arxiv.org/abs/2505.18654) (2025) by Han et al., *arXiv*.

## Weave Generative Architectures into DLRM
15. Alibaba has many different GR models 👉 check out their latest [RecGPT Technical Report](https://huggingface.co/papers/2507.22879) --- below are two well-known examples:
    - GPSD 👉 [*Scaling Transformers for Discriminative Recommendation via Generative Pretraining*](https://arxiv.org/abs/2506.03699) (2025) by Wang et al., *KDD*.
    - LUM 👉 [*Unlocking Scaling Law in Industrial Recommendation Systems with a Three-Step Paradigm Based Large User Model*](https://arxiv.org/abs/2502.08309) (2025) by Yan et al., *arXiv*.
    - URM 👉 [*Large Language Model as Universal Retriever in Industrial-Scale Recommender System*](https://arxiv.org/abs/2502.03041) (2025) by Jiang et al., *arXiv*.
16. JD.com's GenCTR 👉 [*Generative Click-through Rate Prediction with Applications to Search Advertising*](https://arxiv.org/abs/2507.11246) (2025) by Kong et al., *arXiv*. 
17. Xiaohongshu's RankGPT 👉 [*Towards Large-Scale Generative Ranking*](https://arxiv.org/abs/2505.04180) (2025) by Huang et al., *arXiv*.
18. Netflix's Foundation Model 👉 [*Foundation Model for Personalized Recommendation*](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) (2025) by  Hsiao et al., *Netflix Technology Blog*.
19. Pinterest's PinFM 👉 [*PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform*](https://arxiv.org/abs/2507.12704) (2025) by Chen et al., *RecSys*.
20. Tencent's LC-Rec is an "older" model that aims to align the semantic space of LLMs with the collaborative space of RecSys 👉 [*Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation*](https://arxiv.org/abs/2311.09049) (2024), *ICDE*.
<!-- 21. ByteDance's RankMixer 👉 [*RankMixer: Scaling Up Ranking Models in Industrial Recommenders*](https://arxiv.org/abs/2507.15551) (2025) by Zhu et al., *arXiv*. -->