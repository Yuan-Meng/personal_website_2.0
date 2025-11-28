---
title: Preparing for ML Infra System Design Interviews
date: 2025-11-27
math: true
categories: ["career", "ml infra", "interview"]
toc: true
---

# The Knowledge Dilemma: Those Who Build Models for Scalable RecSys Don't Work on Scalable Infra
Only a handful of companies like Netflix, Snap, Reddit, Notion, and DoorDash have an ML infra system design round for MLE candidates, in addition to standard ML system design. Maybe you'll never have to interview with them. However, apart from the frontier AI {{< sidenote "labs" >}}In fact, even if you get an offer from a frontier lab as a Research Engineer rather than a Research Scientist, you don't necessarily get paid more than you would at Netflix or Snap at the same level.{{< /sidenote >}} (e.g., OpenAI, Anthropic, xAI, Google DeepMind, Reflection), the first two pay (far) more than most at the same level. Many solid MLEs are incentivized to pass their interviews at some point in their careers.

ML system design focuses on translating business objectives into ML objectives, choosing training data, labels, and model architectures, and evaluating models offline and online. By contrast, <span style="background-color: #D9CEFF">ML infra system design focuses on the offline + online pipelines that support an ML system</span>. One type of question asks you to walk through full online + offline pipelines; another asks you to design specific components, such as a feature store, real-time feature updates, or distributed training.

Here's the funny thing: If your company serves truly large-scale recommender systems (e.g., recommending billions of items to hundreds of millions of DAUs), you're likely working with dedicated infra teams that handle logging, training, and inference for you. Your job is to optimize a narrow set of North Star metrics your team is funded for (e.g., CTR, CVR, revenue, search relevance). If your company isn't making recommendations at scale, the knowledge of how to build scalable ML systems may be years away from your reality.

That said, I do think ML infra interviews are valuable: modern recommender system teams function like Formula One teams --- even if you ask Max Verstappen to build a car, he couldn't do it to save his life, but no driver on the grid doesn't have an intimate knowledge of car mechanics. The best drivers have a fantastic feel for which parts are or aren't working and collaborate with technicians to improve the car throughout a season. Similarly, the best ML engineers can make necessary, timely, and reasonable requests of their ML infra partners well ahead of major projects. Solid ML infra knowledge goes a long way in an impactful career. So even if you never take an ML infra interview, you should still spend time learning this knowledge.

# Interview Type 1: Pipeline Walk-Through

## A Bare-Bone ML System

A bare-bone ML system consists of the following components:

{{< figure src="https://www.dropbox.com/scl/fi/rdbqiv5wk1vs8967e0rt5/Screenshot-2025-11-27-at-4.30.41-PM.png?rlkey=kyjd9hf8ol4w6ekx4diqqzh3t&st=oocf1im6&raw=1" caption="A bare-bone ML system." width="1800">}}


1. **Data ingestion**: consume training data (features + labels + metadata) either all at once or in a streaming fashion 👉 preprocess the data so they're ready to be fed into the model
   - *Batching*: we usually can't load the entire training set at once 👉 split it into mini-batches and train on one batch at a time
   - *Sharding*: extremely large datasets may not fit on a single machine 👉 shard data across multiple machines and let each worker consume from some shards
   - *Caching*: if we read data from a remote source (e.g., a database or datalake) or have expensive preprocessing, we can cache preprocessed data in RAM for future epochs
2. **Model training**: initialize a model, train it on ingested data in a distributed way (data or model parallel), and write out checkpoints
   - How to distribute model training
      - *Distributed data parallel (DDP)*: copy the model to multiple workers 👉 each worker consumes data and computes gradients independently 👉 aggregate gradients (e.g., sum them) to update parameters
      - *Distributed model parallel (DMP)*: if a huge model can't fit in one worker's memory (e.g., hundreds of Transformer blocks), split the model across workers 👉 each worker consumes source data or upstream outputs to compute gradients and update the parameters it owns
   - How to ensure eventual consistency
      - *Use parameter severs*: a parameter server stores the authoritative model parameters 👉 workers push gradients to it, the server applies updates, and workers pull the latest parameters before training the next batch
      - *Collective communication via `allreduce` (`reduce + broadcast`)*: each worker computes gradients independently 👉 aggregate gradients across workers (`reduce`) 👉 send the aggregated gradients back to all workers so they can update parameters (`broadcast`)
3. **Model serving**: load a trained model and use it to make predictions for new inputs (realtime or batched)
      - *Replication*: to handle high many concurrent queries with low latency, replicate the model across multiple model servers and use a load balancer to distribute traffic evenly
      - *Sharding*: if a request is too large for a single worker and can be decomposed (e.g., frame-level video understanding) 👉 distribute sub-requests to shards 👉 aggregate results
      - *Even-driven processing*: in systems with strong surge patterns (e.g., Uber ride requests, Airbnb bookings), create a shared resource pool that processes can borrow from during peak hours (with a rate limiter to prevent resource exhaustion)

## Interview Answer Organization

### Top Principle: Design Interview == Leadership + Time Management + Domain Knowledge

A design interview is a perfect venue to showcase leadership, time management, and communication skills, on top of  domain knowledge. 

An ML system has many moving parts (like all distributed systems do) --- from generating and validating training data, scheduling training, to handling high QPS in a way that makes the most sense for your product. You need a coherent story to tie those little pieces together and sell your story-telling plan to your interviewer. You should be assertive when the interviewer doesn't have a strong preference, and flexible when they do. That's essentially what leadership is: influencing without authority and staying open-minded to different ideas.

Last but not least, painting a high-level picture isn't enough --- you must identify and deep dive into the most interesting parts of your system, rather than dwelling on the mundane or trivial parts. That's where your time management instincts and domain knowledge shine.

### Online Inference: Query Life Cycle

### Offline Processing: Get Data to Train Models

# Interview Type 2: Component Design

## Feature Stores

## Real-Time Feature Updates

## Distributed Training

# References

## End-to-End Systems
### Abstract ML Systems
To begin, get an abstract overview of end-to-end systems:

> Many books have been written on either machine learning or distributed systems. However, there is currently no book available that talks about the combination of both and bridges the gap between them. --- *Distributed Machine Learning Patterns*

1. [Distributed Machine Learning Patterns](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022) 👉 this is a *fantastic* book about ML infra. Some folks dismiss it because the author still uses (1) TensorFlow and (2) the Fashion-MINST dataset to illustrate the model component, but I bet they didn't read the book. 
   - (1) is understandable because the author was a main contributor of *Google*'s Kubeflow. Do we expect PyTorch? 🤣
   - (2) is the point --- the author wants readers to build an end-to-end ML infra system on their own machine and has provided probably the cleanest and most vividly explained instructions on how to do so. A toy dataset is what fits.
2. [Introducing Bento, Snap's ML Platform](https://eng.snap.com/introducing-bento) 👉 Snap's all-in-one ML platform for feature engineering, training data generation, model training, and online inference
3. [Scaling AI/ML Infrastructure at Uber](https://www.uber.com/blog/scaling-ai-ml-infrastructure-at-uber/?uclick_id=d2051111-296f-44e0-b45d-0a6bd4cc98b4) 👉 evolution of Uber's Michaelangelo platform
4. Metaflow 👉 Netflix's open-source framework for model training and inference: [Why Metaflow](https://docs.metaflow.org/introduction/why-metaflow), [paper](https://arxiv.org/abs/2303.11761), [blog](https://netflixtechblog.com/supercharging-the-ml-and-ai-development-experience-at-netflix-b2d5b95c63eb), [tech overview](https://docs.metaflow.org/internals/technical-overview)


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