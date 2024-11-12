---
title: "Down the Rabbit Hole: Sequential User Modeling"
date: 2024-11-12
math: true
categories: ["recommender systems", "information retrieval"]
toc: true
---

# Catch the Train of Actions

Shown below is my Amazon browsing history last week. Any recommendations on what I might buy next?

{{< figure src="https://www.dropbox.com/scl/fi/t3fs6pgs8rxlysnpoqeq9/Screenshot-2024-11-11-at-4.25.33-PM.png?rlkey=kv37yorsm6qdlro4y4ot496s8&st=p200ep0j&raw=1" caption="Yuan's Amazon browsing history last week; distinct sessions are color-coded." width="1800">}}

<!--more-->

A model performing average/sum pooling over engaged items may recommend books, cleaning supplies, or blue shampoos --- these were the items I engaged the most with and align with my long-term interests in tidiness, reading, and paying punk homage. But in this session, I want more pull-up bar recommendations as I'm comparing options. After that, I may wanna look at more fitness accessories...

Rather than treating engaged items as a *static*, *unordered* collection, sequential user modeling traces the train of action sequences to predict the next action, adapting to evolving, dynamic user interests. 

# Flavors of Sequence Modeling

Sequential user modeling can be framed as a next-item-prediction problem: <span style="background-color: #abe0bb">given a user's action sequence $S = \\{i_1, \ldots, i_L\\}$ and a target item $i_t$, output a utility score for the item $p(i_t|i_{i:L})$</span>. Each interaction $i_j$ consists of a $\langle \mathrm{user}, \mathrm{action}, \mathrm{item} \rangle$ triple, where the action could be a click, an add-to-cart, a conversion, or other meaningful engagements with recommended or sponsored content.

Any methods suitable for modeling sequences (e.g., tokens in language, pixels in images, genes in DNAs) can be applied to this problem, from Markov chains, RNNs, CNNs, and GNNs that pre-date Transformers, to the Transformer architecture (whether using only the attention mechanism or the full encoder) adapted to recommender systems (see [Wang et al. 2019](https://arxiv.org/abs/2001.04830) for a comprehensive review).

{{< figure src="https://www.dropbox.com/scl/fi/bo2lmx0zswr9ntdlofs0c/Screenshot-2024-11-12-at-12.10.38-AM.png?rlkey=d3cmaxh5i1dckdnonvczfwm7i&st=vmlrlwvc&raw=1" caption="An overview of classic sequential user modeling methods ([Wang et al., 2019](https://arxiv.org/abs/2001.04830))." width="600">}}

One thing I find interesting (but also expected) is that, since sequential user modeling borrows heavily from natural language processing (NLP), the evolution of the former closely follows that of the latter.

## Pre-Transformer

### Markov Chains

In his [*A Mathematical Theory of Communication*](http://cm.bell-labs.com/cm/ms/what/shannonday/shannon1948.pdf) that laid the foundation for information theory, {{< sidenote "Claude" >}}it only dawned on me after a year that Anthropic's Claude is named after Shannon.{{< /sidenote >}} Shannon used a [Markov chain](https://www.cs.princeton.edu/courses/archive/spr05/cos126/assignments/markov.html) model to predict the transitional probability from one alphabet to another as an attempt to model natural language. The toy implementation below defines state transitions and randomly selects the next states to generate sequences. Transitional probabilities can be learned. 

```python3
class MarkovChain:
    def __init__(self):
        # dictionary to store transitions: {current state : [next state]}
        self.transitions = {}

    def addTransition(self, v, w):
        # add a transition from state v to state w
        if v not in self.transitions:
            self.transitions[v] = []
        self.transitions[v].append(w)

    def next(self, v):
        # pick a transition leaving state v uniformly at random
        if v not in self.transitions or not self.transitions[v]:
            return None  # no transitions available
        return random.choice(self.transitions[v])

    def toString(self):
        # return a string representation of the Markov chain
        result = []
        for state, transitions in self.transitions.items():
            result.append(f"{state} -> {', '.join(transitions)}")
        return "\n".join(result)
```

In a recommender system, a Markov-chain model predicts the transition probability from one action sequence ($S = \\{\mathrm{printer}, \mathrm{ink}\\}$) to another ($S = \\{\mathrm{printer}, \mathrm{ink}, \mathrm{paper}\\}$), either directly (e.g., [Garcin et al., 2013](https://arxiv.org/abs/1303.0665)) or by embedding Markov chains into a latent space to compute transitions from Euclidean distances (e.g., [Feng et al., 2015](https://www.ijcai.org/Proceedings/15/Papers/293.pdf)). 

{{< figure src="https://www.dropbox.com/scl/fi/pgjlb1nee84lihensin30/Screenshot-2024-11-12-at-12.59.47-PM.png?rlkey=a43d16ssyuhjjtxmzyldqawpu&st=nfyg4npx&raw=1" caption="Markov chains recommend items by transition probabilities ([Feng et al., 2015](https://www.ijcai.org/Proceedings/15/Papers/293.pdf))." width="600">}}

A fatal shortcoming of Markov chains lies in the ["memorylessness"](https://en.wikipedia.org/wiki/Memorylessness) assumption --- that future states depend only on the current state, ignoring preceding states. For example, if I bought cat food (because I suddenly remembered 😂) after buying ink, the system won't be more likely to recommend paper to me than if I didn't buy ink at all. Real shoppers jump between diverse interests and engage with unrelated items, which are nuances that Markov chains cannot capture.

### Recurrent Neural Networks

Compared to Markov chains that memory-less, Recurrent Neural Networks (RNNs) are able to retain activation from previous states several steps ago.

### Convolutional Neural Networks
### Graph Neural Networks

## Target Attention

<!-- DIN started the target attention traditional. general idea is different items play different roles for the same target.  -->

### One-Stage: DIN Family

### Two-Stage: GSU + ESU

## "Language Modeling"

### Masked Action Modeling

### Next-Action Prediction

<!-- BERT-style models. 
Q: why not GPT style w/ causal mask, which is more natural for future prediction? -->

## Is Attention What You Need?

> The efficacy of model simplification often hinges on precise prior knowledge, prompting an inquiry into why certain simplifications to the Transformer architecture prove effective and what insights they offer. --- Wang et al., [*ICLR 2024*](https://openreview.net/forum?id=Gny0PVtKz2)

### Google: ConvFormer

### Meta: HSTU

<!-- Meta and Google => strip away parts of Transformers -->

---
# Code Examples 

## DIN (Alibaba, 2017)

## TransAct (Pinterest, 2023)

[repo](https://github.com/pinterest/transformer_user_action)

---
# References

## Overview & Collections
1. "Old" but popular lit review 👉 [*Sequential Recommender Systems: Challenges, Progress and Prospects*](https://arxiv.org/abs/2001.04830) (2019) by Wang et al., *IJCAI*.
2. A Meta MLE's awesome post 👉 [*User Action Sequence Modeling: From Attention to Transformers and Beyond*](https://mlfrontiers.substack.com/p/user-action-sequence-modeling-from) 
3. GitHub repos of sequential user modeling 👉 papers ([Awesome-Sequence-Modeling-for-Recommendation](https://github.com/HqWu-HITCS/Awesome-Sequence-Modeling-for-Recommendation)) + code ([FuxiCTR](https://github.com/reczoo/FuxiCTR))

## Approach: Target Attention
4. Overview: [*Target Attention Is All You Need: Modeling Extremely Long User Action Sequences in Recommender Systems*](https://mlfrontiers.substack.com/p/target-attention-is-all-you-need) by Samuel Flender.

5. The OG architecture 👉 DIN: [*Deep Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/1706.06978) (2017) by Zhou et al., *KDD*.
   - And its many a Alibaba siblings: [DIEN (2018)](https://arxiv.org/abs/1809.03672), [DSIN (2019)](https://arxiv.org/abs/1905.06482), [DHAN (2020)](https://arxiv.org/abs/2005.12981), [DMIN (2020)](https://dl.acm.org/doi/abs/10.1145/3340531.3412092), [DAIN (2024)](https://arxiv.org/abs/2409.02425), ...
6. Go crazy on sequence length 👉 SIM: [*Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction*](https://arxiv.org/abs/2006.05639) (2020) by Qi et al., *CIKM*.
   - Ultra long: [ETA (2021)](https://arxiv.org/abs/2108.04468), [TWIN (2023)](https://arxiv.org/abs/2302.02352), [TWIN-v2 (2024)](https://arxiv.org/html/2407.16357v1), ...
   - Review post: [*Towards Life-Long User History Modeling in Recommender Systems*](https://mlfrontiers.substack.com/p/towards-life-long-user-history-modeling) by Samuel Flender.
7. Squeeze every ounce of sequences 👉 TIM: [*Ads Recommendation in a Collapsed and Entangled World*](2024) by Pan et al, *KDD*.
   - Paper summary: [*Breaking down Tencent's Recommendation Algorithm*](https://mlfrontiers.substack.com/p/breaking-down-tencents-recommendation) by Samuel Flender.


## Approach: Language Modeling

8. The OG 👉 [*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*](https://arxiv.org/abs/1904.06690) (2019) by Sun et al., *CIKM*.
9. Play with objectives 👉 [*PinnerFormer: Sequence Modeling for User Representation at Pinterest*](https://arxiv.org/abs/2205.04507) (2022) by Pancha et al., *KDD*.
10. Capture short-term interests 👉 [*TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest*](https://arxiv.org/abs/2306.00248) (2024) by Xia et al., *KDD*.
11. Applications at Pinterest 👉 organic ranking ([*Large-scale User Sequences at Pinterest*](https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9)) + ads ranking ([*User Action Sequence Modeling for Pinterest Ads Engagement Modeling*](https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e))


## Approach: Beyond Attention
12. Meta AI 👉 [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152) (2024) by Zhai et al., *ICML*.
13. Google Research 👉 [*ConvFormer: Revisiting Token-mixers for Sequential User Modeling*](https://openreview.net/forum?id=Gny0PVtKz2) (2024) by Wang et al., *ICLR*.

