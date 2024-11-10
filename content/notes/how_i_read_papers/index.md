---
title: "How I Read Papers for Blogposts"
date: 2024-11-10
categories: ["writing", "literature review"]
---

Former OpenAI researcher Lilian Weng has taught foundational and SOTA in AI/ML to a generation of machine learning practitioners through her awesome [blogposts](https://lilianweng.github.io/faq/). It all started with her personal learning notes and a passion for learning through teaching.

I started my own blog for the same reasons: By sharing my learnings publicly, I motivate myself to articulate my understanding explicitly. 

So, how do I begin? To dive deeper into search/recommendations/ads ("搜广推") topics, for instance, I usually read papers in 3 batches. Take my favorite post [*An Evolution of Learning to Rank*](https://www.yuan-meng.com/posts/ltr/) as an example --- 

1. **Batch 1**: Highly-cited literature review, such as Microsoft Research's [ NeurIPS 2011 paper](https://www.semanticscholar.org/paper/A-Short-Introduction-to-Learning-to-Rank-Li/d74a1419d75e8743eb7e3da2bb425340c7753342) on classic architectures for learning to rank (LTR)
   - **Why**: To gain an overview of the key problems in this field (e.g., correctly order documents by relevance) and the general types of approaches (e.g., objectives: pointwise, pairwise, listwise; architectures: SVM, GBDT, DNN, etc.).
   - **How**: Search model types (e.g., "learning to rank", "sequential user modeling") in Google Scholar or [Papers with Code](https://paperswithcode.com/).
2. **Batch 2**:Industry papers by research teams, such as Google Research's [ICLR 2021 paper](https://research.google/pubs/are-neural-rankers-still-outperformed-by-gradient-boosted-decision-trees/) first getting a neural ranker to perform on a par with GBDT baselines on LTR benchmarks 
   - **Why**: Connect theoretical details to real-world applications.
   - **How**: Check out papers by Google Research and Microsoft Research, which are rich in theory and tested extensively on vertical products (e.g., YouTube, Google Play, Bing).
3. **Batch 3**: Industry papers by product teams, such as Airbnb Search's journey into deep learning ([2019](https://arxiv.org/pdf/1810.09591), [2020](https://arxiv.org/pdf/2002.05515), [2023](https://arxiv.org/pdf/2305.18431))
   - **Why**: To see how different companies tailor generic ideas to specific products, users, UI, or contexts ("量体裁衣").
   - **How**: Papers by Meta, LinkedIn, Pinterest, etc. usually have less mathematical proof but discuss design choices and model serving at length. Chinese companies such as Alibaba and JD often push model complexity to the next level.

The trick is knowing what to learn. For example, behind great search and recommendations are great user + content understanding, efficient and high-recall retrieval, unbiased and high-precision ranking, and methods to mitigate position bias, cold start, and staleness, among others. My knowledge comes from working on my own projects, chatting with my search/personalization/ads colleagues, going down the rabbit hole of paper citations, or perhaps just stumbling on papers recommended in my LinkedIn or 小红书 feed --- such serendipity is why I love ranking in the first place 😉.