---
title: "ML Stuff I Wish to Learn (More) About in 2025"
date: 2024-11-23
categories: ["machine learning"]
---

MLE friends and colleagues sometimes tell me about a sense of *lack*: how there's so much to learn about the ML engineering craft --- from mathematical foundations and deep learning history, to state-of-the-art research, industry gold standards, and the nitty-gritty engineering details --- that it never feels like there's time to go deep on anything.

<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
  <figure style="margin: 0; text-align: center;">
    <img src="https://www.dropbox.com/scl/fi/mupdtd7fuolajkav4t0a5/315980733_528828279256236_9045916108072816684_n.jpg?rlkey=zk08zehs7uja0hlu3t7tk3f3q&st=bwby1re1&raw=1" alt="Image description" style="width: 300px; margin: 0 auto;">
    <figcaption style="margin-top: 8px;">Credit: <a href="https://www.fosslien.com/" target="_blank" rel="noopener noreferrer">Liz Fosslien</a></figcaption>
  </figure>
</div>

Here's how I see it: if you've kept your job (by delivering impact without getting fired), you're already doing great. Hold on to who you {{< sidenote "are" >}}In my case, I'm a pragmatic ML engineer curious about RecSys/LLM research, but I'm not an ML researcher or platform engineer.{{< /sidenote >}} and gradually foray into the unknowns that excite you --- like opening a box of chocolates, take it one piece at a time and savor it.

Below are boxes of chocolates that I desire to open in 2025 😋🍫. 

{{< admonition >}}
👀 This list will be updated regularly, as I see/think of more to learn.
{{< /admonition >}}

# Foundations of Deep Learning
1. **Ilya Sutskever's [reading list](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE)**: 30 papers curated by Ilya for his friend John Carmack to get up to speed with AI's greatest hits.
2. **Mathematics for deep learning**: 3Blue1Brown ([linear algebra](https://www.3blue1brown.com/topics/linear-algebra), [calculus](https://www.3blue1brown.com/topics/calculus)) is intuitive, Stanford CS229 [handout](https://github.com/ctanujit/lecture-notes/blob/main/ML/Mathematics%20for%20Machine%20Learning%20by%20Stanford%20University.pdf) is a quick refresher, and the [Bishop](https://www.bishopbook.com/) {{< sidenote "book" >}}Just about every deep learning book has a chapter or an appendix on probability theory, linear algebra and calculus. When I get the time, I wish to learn the notations more systematically and dive a bit deeper into more advanced linear algebra.{{< /sidenote >}} is most comprehensive and up to date
      - **Read before and good to review:** [*Linear Algebra: Theory, Intuition, Code*](https://www.amazon.com/Linear-Algebra-Theory-Intuition-Code/dp/9083136604) and Kevin Murphy's books ([1](https://probml.github.io/pml-book/book1.html) and [2](https://probml.github.io/pml-book/book2.html))

# LLM Trends + Details 
3. [**Build a Large Language Model (From Scratch)**](https://www.manning.com/books/build-a-large-language-model-from-scratch): this book teaches end-to-end LLM training (pre-training, fine-tuning, instruction-following) and includes a systematic PyTorch review.
4. **LLM technical reports:** [Claude 3](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) by Anthropic, Gemini [1](https://arxiv.org/abs/2312.11805)/[1.5](https://arxiv.org/abs/2403.05530) by Google DeepMind, [GPT-4](https://arxiv.org/abs/2303.08774) by OpenAI, Lllama [1](https://arxiv.org/abs/2302.13971)/[2](https://arxiv.org/abs/2307.09288)/[3](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) by Meta AI
5. **Multimodal models**: Sebastian Raschka's [summary blog post](https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html)
6. **Agents**{{< sidenote "" >}}In discussions with several friends, a common sentiment is that despite the explosion of LLM startups and applications, there seems to be a fundamental lack of imagination regarding how LLMs can reshape life and society more profoundly. Apart from chatbots and embedding/completion/speech-to-text/text-to-speech APIs sold by OpenAI, Anthropic, and similar companies, RAG and agents appear to be the only solid use cases widely applied across the industry.{{< /sidenote >}}: survey [paper](https://arxiv.org/abs/2401.03568) (Durante et al., 2024) from Fei-Fei Li's group, [XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) (a new framework from [Tianqi Chen](https://tqchen.com/)'s group for generating structured responses), [LLM-Agents-Papers repo](https://github.com/AGI-Edgerunners/LLM-Agents-Papers)


# Model Serving and Deployment

**IMO:** It may make no sense to dive into 7-9 unless you already or are about to work at one of these companies (e.g., Meta, LinkedIn, Pinterest, Google) --- after all, who needs detailed instructions on using a platinum cookware set without access to ingredients 🤑?

7. **GPU serving:** GPU-accelerated neutral retrieval (Meta [2023](https://arxiv.org/abs/2306.04039v1) & [2024](https://arxiv.org/abs/2407.15462), [LinkedIn 2024](https://arxiv.org/abs/2407.13218)), GPU-based inference (e.g., [Pinterest](https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d))
8. **Mixed precision serving:** originally used in LLM training and inference ([Nvidia](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)) to reduce costs 👉 adapted by Pinterest to serve their ranking models (e.g., [ads L2 ranking](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html))
9. **ML compilers**: start with [*Understanding ML Compilers: The Journey From Your Code Into the GPU*](https://substack.com/home/post/p-151758494) by Sam Flender


# Emerging Topics in RecSys
10. **Semantic ID**: to help with *cold start* and *generalization*, a new id with similar semantic content to an existing id should be hashed into the same bucket 👉 see the DeepMind paper ([RecSys '24](https://arxiv.org/pdf/2306.08121)) for details and the Tencent paper ([KDD '24](https://arxiv.org/abs/2403.00793)) for applications
11. **LLM-ify RecSys**: survey paper by [Wu et al. (2023)](https://arxiv.org/abs/2305.19860), Coding Monkey posts ([in-context learning](https://pyemma.github.io/How-to-use-GPT-for-recommendation-task/), [align semantic and collaborative filtering spaces](https://pyemma.github.io/Machine-Learning-System-Design-Sparse-Features/)), Meta [(ICML '24)](https://arxiv.org/abs/2402.17152), Google ([arXiv](https://arxiv.org/abs/2409.11699))
12. **Scaling law for RecSys**: [*Scaling Laws for Online Advertisement Retrieval*](https://arxiv.org/abs/2411.13322) by Kuaishou (Wang et al., 2024)
