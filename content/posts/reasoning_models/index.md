---
title: "What Does a Recommendation Model Get from Reasoning?"
date: 2026-10-11
math: true
categories: ["recommender systems", "reasoning models"]
toc: true
---

One night, I got a baffling message from a friend: "Don't go to that restaurant!" Turns out she'd tried a new place from a Xiaohongshu post I had forwarded to her, and it was terrible. "Oh, I wasn't recommending it," I explained. "Then what *was* your intent?" she asked. "I was scrolling my feed before bed and forwarded the first new place I saw. There was no thought behind it." "Well," she joked, "I guess I was a shitty reasoning model that made up all that intent."

Like any human behavior, engagements with recommender systems are noisy --- my sleepy tap "tricked" my friend as if it were a well-reasoned recommendation. A reasoning model may "see through" the intent behind a behavior and recommend one's heart's desire, or it might just conjure something out of nothing. How can recommender systems benefit from reasoning? If not out of the box, how do we steer a recommender model toward reasoning that's actually useful?

# What Is a Reasoning Model, Anyway?

## From Next-Token Prediction to Thinking Before Answering

<!-- Definition: chain-of-thought, test-time compute, RL with verifiable rewards. Keep short; point to Raschka for details. -->

## Million-Dollar Questions

<!-- Frame the whole post around these three; each later section answers one. -->

1. Does better reasoning mean better answers? (How well does reasoning quality predict answer quality?)
2. How is reasoning quality measured and improved?
3. What counts as a "good answer" in LLMs vs. RecSys, and how is it measured?

# Bringing Reasoning into Recommender Systems

## Success Stories So Far

## 500 Ways Reasoning Flops

<!-- Bridge section (new): what makes RecSys different from math/code — noisy labels (the restaurant story), no verifiable reward, no natural-language intent, millisecond latency budgets. Sets up why "just add CoT" doesn't work. -->

<!-- # Success Stories in RecSys So Far -->

<!-- Organize by *where* reasoning lives rather than by paper. Fill in the models you have in mind. -->

<!-- ## Reasoning Offline: LLMs as Intent Annotators -->

<!-- LLM-generated user/item profiles, intent labels, or features fed into a conventional ranker. -->

<!-- ## Reasoning Inside the Ranker: Latent Chains of Thought -->

<!-- Generative rankers that "think" in tokens/latents before predicting the next action. -->

<!-- ## Reasoning at the Feed Level: Explaining Instead of Scoring -->

<!-- Optional: conversational / explainable recs, if you want to cover it. -->

# Steering Recommenders Toward Useful Reasoning

<!-- Your "future directions," reframed around the three questions above. -->

<!-- ## What's the Reward? Verifiable Signals in a Noisy World -->

<!-- ## What's the Verifier? Measuring Reasoning When There's No Ground Truth -->

<!-- ## Reasoning on a Latency Budget -->

# References

## Reasoning Models in LLMs

1. A hands-on introductory book on building reasoning models 👉 [*Build a Reasoning Model From Scratch*](https://sebastianraschka.com/reasoning-from-scratch/) (2026) by Sebastian Raschka.
2. The go-to lecture on LLM reasoning, from the founder of DeepMind's reasoning team 👉 [*Large Language Model Reasoning*](https://dennyzhou.github.io/LLM-Reasoning-Stanford-CS-25.pdf) (2025) by Denny Zhou, *Stanford CS25*.
3. My colleague's blogpost summarizing CS25 👉 [*Reasoning in Large Language Models: A Research-Centric Overview*](https://renee-jia.github.io/ai%20learning%20guide/ai%20research/large%20language%20models/machine%20learning/llm-reasoning-research-overview/) (2025) by Renee Jia.
4. A survey on how representative reasoning LLMs (DeepSeek-R1, o1, QwQ, etc.) are trained 👉 [*From System 1 to System 2: A Survey of Reasoning Large Language Models*](https://arxiv.org/abs/2502.17419) (2025) by Li et al., *arXiv*.
5. An epic summary of 300+ LLM reasoning papers by reasoning paradigm and failure mode 👉 [*The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes*](https://arxiv.org/abs/2606.11470) (2026) by Anand et al., *arXiv*.

## Reasoning Models in RecSys

1. Kuaishou's OneRec-Think --- the first in-text reasoning in a generative recommender; thinking mode didn't beat non-thinking 👉 [*OneRec-Think: In-Text Reasoning for Generative Recommendation*](https://arxiv.org/abs/2510.11639) (2025) by Liu et al., *arXiv*.
2. Kuaishou's OneReason 👉 [*OneReason Technical Report*](https://arxiv.org/abs/2606.06260) (2026) by Zhao et al., *arXiv*.
3. Kuaishou's OneLatent 👉 [*OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models*](https://arxiv.org/abs/2607.26621) (2026) by Jiang et al., *arXiv*.
4. Kuaishou's OneSearch-V2 👉 [*OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework*](https://arxiv.org/abs/2603.24422) (2026) by Chen et al., *arXiv*.
5. Kuaishou's PROMISE --- a process reward model verifies each Semantic ID step, unlocking test-time scaling laws where small models + more compute beat big ones 👉 [*PROMISE: Process Reward Models Unlock Test-Time Scaling Laws in Generative Recommendations*](https://arxiv.org/abs/2601.04674) (2026) by Guo et al., *arXiv*.
6. Shopee's OnePiece --- context engineering + block-wise latent reasoning in a production cascade ranker 👉 [*OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System*](https://arxiv.org/abs/2509.18091) (2025) by Dai et al., *arXiv*.
7. Tencent's TGR --- TGR-Reason injects offline-generated reason tokens at decode time, no request-time rollout 👉 [*TGR: Advancing Industrial Recommendation from Generative-Paradigm Ranking toward Unified Generation and Reasoning*](https://arxiv.org/abs/2609.00986) (2026) by the TGR Team, *arXiv*.
8. Alibaba's RecGPT-V2 --- multi-agent intent reasoning with an agentic judge 👉 [*RecGPT-V2 Technical Report*](https://arxiv.org/abs/2512.14503) (2025) by the RecGPT Team, *arXiv*.
9. Alibaba's RecGPT-V3 --- latent intent reasoning over text tags + Semantic IDs, decodable back into explanations 👉 [*RecGPT-V3 Technical Report*](https://arxiv.org/abs/2607.15591) (2026) by Zheng et al., *arXiv*.
10. Meta's GR2 --- reasoning-trace distillation + RL with verifiable rewards for re-ranking 👉 [*GR2: Generative Reasoning Re-ranker*](https://arxiv.org/abs/2602.07774) (2026) by Liang et al., *arXiv*.
11. Meta's GR2 at industrial scale --- SFT collapses, on-policy distillation doesn't; rerankers hack rewards via position bias 👉 [*GR2 Technical Report*](https://arxiv.org/abs/2606.31984) (2026) by Li et al., *arXiv*.
12. R²ec --- unified reasoning + recommendation head, the academic landmark 👉 [*R²ec: Towards Large Recommender Models with Reasoning*](https://arxiv.org/abs/2505.16994) (2025) by You et al., *NeurIPS*.
13. Where latent reasoning steps should go --- allocate by information gain across SID positions 👉 [*Where Reasoning Matters: Rethinking Latent Reasoning in Semantic ID-based Generative Recommendation*](https://arxiv.org/abs/2607.12425) (2026) by Yang et al., *arXiv*.

> The ByteDance camp --- "no reasoning needed, just fix the modeling"

14. ByteDance's X-Rec 👉 [*X-Rec Technical Report*](https://arxiv.org/abs/2609.29180) (2026) by Shen et al., *arXiv*.
15. ByteDance's GEAR 👉 [*Generative End-to-end Ad Retrieval at Douyin*](https://arxiv.org/abs/2609.39327) (2026) by Zeng et al., *arXiv*.
16. ByteDance's OneTrans-V2 👉 [*OneTrans-V2: Unifying Retrieval, Pre-rank, and Fine-rank with One Transformer in Industrial Recommender*](https://arxiv.org/abs/2609.28589) (2026) by Cao et al., *arXiv*.

> LLM agents reasoning *about* the recommender, not inside it

17. Google's self-evolving RecSys --- Gemini agents as autonomous MLEs proposing and launching model changes at YouTube 👉 [*Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents*](https://arxiv.org/abs/2602.10226) (2026) by Wang et al., *arXiv*.