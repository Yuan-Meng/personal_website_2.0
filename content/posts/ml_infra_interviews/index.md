---
title: Preparing for ML Infra System Design Interviews
date: 2025-11-25
math: true
categories: ["career", "machine learning", "infra", "interview"]
toc: true
---

# The Knowledge Dilemma: Those Who Build Models for Scalable RecSys Don't Work on Scalable Infra

A handful of companies (Netflix, Reddit, Snap, DoorDash, Notion, etc.) conduct an ML *infra* design round for MLE candidates. The focus isn't on model development --- like collecting training data and generating labels, choosing features and model architectures, or running online/offline experiments --- but on offline + online pipelines that support an ML system. You might walk through the entire online + offline pipelines or design a specific component, such as feature stores, real-time feature updates, or distributed training. The goal is to see whether you have a full picture of how different pieces of ML infra work together and know how to scale each part and make trade-offs.

Here's the funny thing: If you are an ML engineer working at a company with large-scale recommender systems like Meta, Google, or Pinterest, you may have a fairly shallow understanding of ML infra only as a user, since there are dedicated infra teams handling logging, training, and inference for you. Your job is mostly model iteration, experiments, and maybe some observability. However, if you don't need to make recommendations at scale, this knowledge is irrelevant.

Before you prepare for this round, first decide if it's worth it.
- **Case 1**: The role is called MLE but actually focuses on infra
  - You should definitely find it out by asking your recruiter (ideal) or your phone interviewer (the latest).
  - In this case, think about whether this role fits your career goals. If not, turn down the interview to save everyone time.

- **Case 2**: The role focuses on ML but still tests infra knowledge
  - You'll probably want a solid understanding of ML infra anyway --- it makes collaboration with infra partners much easier and helps you design better end-to-end solutions.
  - If this is is the case, treat the preparation as a chance to learn something fun/new! (Perhaps rest assured that other model-focused ML engineers are doing the same to jump the hoop.)

# Interview Type 1: Full Pipeline Walk-Through

## Online Inference

## Offline Processing

# Interview Type 2: Component Design

## Feature Stores

## Real-Time Feature Updates

## Distributed Training

# References

## End-to-End Systems
### Abstract ML Systems
To begin, get an abstract overview of end-to-end systems:

1. [Distributed Machine Learning Patterns](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022) 👉 good overview of end-to-end ML systems (specific stacks may be dated)
2. [Introducing Bento, Snap's ML Platform](https://eng.snap.com/introducing-bento) 👉 Snap's all-in-one ML platform for feature engineering, training data generation, model training, and online inference
3. [Scaling AI/ML Infrastructure at Uber](https://www.uber.com/blog/scaling-ai-ml-infrastructure-at-uber/?uclick_id=d2051111-296f-44e0-b45d-0a6bd4cc98b4) 👉 evolution of Uber's Michaelangelo platform
4. Metaflow 👉 Netflix's open-source framework for model training and inference: [basics](https://docs.metaflow.org/metaflow/basics), [tech overview](https://docs.metaflow.org/internals/technical-overview)


### Model Use Cases
Then, dig into systems designed for specific models:

5. [Evolution and Scale of Uber's Delivery Search Platform](https://www.uber.com/blog/evolution-and-scale-of-ubers-delivery-search-platform/) 👉 Uber Eats' search platform
6. [Introducing DoorDash's In-House Search Engine](https://careersatdoordash.com/blog/introducing-doordashs-in-house-search-engine/) 👉 Argo, DoorDash's search platform; [debugging](https://careersatdoordash.com/blog/doordash-optimizing-in-house-search-engine-platform/)
7. [Embedding-Based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval) 👉 Snap's retrieval model for spotlight videos 
8. [Snap Ads Understanding](https://eng.snap.com/snap-ads-understanding) 👉 Snap's video ad content understanding model
9. [Machine Learning for Snapchat Ad Ranking](https://eng.snap.com/machine-learning-snap-ad-ranking) 👉 Snap's pCVR model
10. `ApplyingML` 👉 [blogposts](https://applyingml.com/resources/) on ML model and system designs 


## ML Infra Components
### Feature + Data Generation

11. [The Dataflow Model](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/) 👉 Google's seminal paper on large-scale data generation
12. [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) 👉 Jeff Dean's original MapReduce paper
13. [Point-in-Time Correctness in Real-Time Machine Learning](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) 👉 data leakage prevention
14. [Speed Up Feature Engineering for Recommendation Systems](https://eng.snap.com/speed-up-feature-engineering) 👉 Robusta, Snap's feature pipeline
15. [Michelangelo Palette: A Feature Engineering Platform at Uber](https://www.infoq.com/presentations/michelangelo-palette-uber/#:~:text=Michelangelo%20Palette%20is%20essentially%20a,models%20and%20why%20is%20that%3F) 👉 Palette, Uber's feature pipeline and feature store
16. [Zipline --- Airbnb's ML Data Management Framework](https://conferences.oreilly.com/strata/strata-ny-2018/cdn.oreillystatic.com/en/assets/1/event/278/Zipline_%20Airbnb_s%20data%20management%20platform%20for%20machine%20learning%20Presentation.pdf) 👉 Zipline, Airbnb's feature pipeline and feature store
- [Building a Spark-Powered Platform for ML Data Needs at Snap](https://eng.snap.com/prism) 👉 Prism, Snap's training data pipeline


### Distributed Training
17. [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 👉 train LLMs on GPU clusters
18. [How to Train a Model on 10k H100 GPUs?](https://soumith.ch/blog/2024-10-02-training-10k-scale.md.html) 👉 PyTorch author's short blogpost
19. [Training Large-Scale Recommendation Models with TPUs](https://eng.snap.com/training-models-with-tpus) 👉 Snap has been using Google's TPUs since 2022


### Real-Time Features
20. [How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8) 👉 real-time user aggregation features
21. [Large-scale User Sequences at Pinterest](https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9) 👉 real-time user sequence features

### GPU Serving
22. [Applying GPU to Snap](https://eng.snap.com/applying_gpu_to_snap) 👉 Snap's switch from CPU to GPU serving
23. [GPU-Accelerated ML Inference at Pinterest](https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d) 👉 Pinterest did the same a year later
24. [Introducing Triton: Open-Source GPU Programming for Neural Networks](https://openai.com/index/triton/) 👉 OpenAI's Triton kernels
25. [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 👉 CUDA graphs are often used in GPU serving