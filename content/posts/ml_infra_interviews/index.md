---
title: Preparing for ML Infra System Design Interviews
date: 2025-11-29
math: true
categories: ["career", "ml infra", "interview"]
toc: true
---

# Dilemma: Model Builders != Infra Builders

Only a handful of companies like Netflix, Snap, Reddit, Notion, and DoorDash have an ML infra system design round for MLE candidates --- in addition to standard ML system design. Maybe you'll never have to interview with them. However, apart from the frontier AI {{< sidenote "labs" >}}In fact, even if you get an offer from a frontier lab, but as a Research Engineer rather than a Research Scientist, you don't necessarily get paid more than you would at Netflix or Snap at the same level.{{< /sidenote >}} (e.g., OpenAI, Anthropic, xAI, Google DeepMind, Reflection), the first two pay more than most at the same level. I guess many solid MLEs are incentivized to pass their interviews at some point in their careers.

ML system design focuses on translating business objectives into ML objectives, choosing training data, labels, and model architectures, and evaluating models offline and online. By contrast, <span style="background-color: #D9CEFF">ML infra system design focuses on the offline + online pipelines that support an ML system</span>. One type of question asks you to walk through full online + offline pipelines; another asks you to design specific components, such as a feature store, real-time feature updates, or distributed training.

Here's the funny thing: If your company serves truly large-scale recommender systems (e.g., recommending billions of items to hundreds of millions of DAUs), you're likely working with dedicated infra teams that handle logging, training, and inference for you. Your job is to optimize a narrow set of North Star metrics your team is funded for (e.g., CTR, CVR, revenue, search relevance). If your company isn't making recommendations at scale, the knowledge of how to build scalable ML systems may be years away from your reality.

That said, I do think ML infra interviews are valuable: Modern recommender system teams function like Formula One teams --- even if you ask Max Verstappen to build a car, he couldn't do it to save his life, but no driver on the grid doesn't have an intimate knowledge of car mechanics. The best drivers have a fantastic feel for which parts are or aren't working and collaborate with technicians to improve the car throughout a season. Similarly, the best ML engineers can make necessary, timely, and reasonable requests of their ML infra partners well ahead of major projects. So, solid ML infra knowledge goes a long way in an impactful career. Therefore, even if you never take an ML infra interview, you should still spend time learning this knowledge.

# Interview Type 1: Pipeline Walk-Through

## A Bare-Bone ML System

A bare-bone ML system consists of the following components:

{{< figure src="https://www.dropbox.com/scl/fi/uvptgcmfbhjs2o5y58i0e/Screenshot-2025-11-28-at-6.54.39-PM.png?rlkey=4oop20e6lbho45elw1ut46z2q&st=c3hwud96&raw=1" caption="A bare-bone ML system ([Distributed Machine Learning Patterns](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022), Chapters 2-6)." width="1800">}}

1. **Data ingestion**: Consume training data (features + labels + metadata) either all at once or in a streaming fashion 👉 preprocess the data so they're ready to be fed into the model
   - *Batching*: We usually can't load the entire training set at once 👉 split it into mini-batches and train on one batch at a time
   - *Sharding*: Extremely large datasets may not fit on a single machine 👉 shard data across multiple machines and let each worker consume from assigned data shards
   - *Caching*: If we read data from a remote source (e.g., a database or datalake) or have expensive preprocessing, we can cache preprocessed data in RAM for future epochs
2. **Model training**: Initialize a model, train it on ingested data in a distributed fashion (using DDP or DMP), and save checkpoints
   - How to distribute model training
      - *Distributed data parallel (DDP)*: copy the full model to multiple workers 👉 each worker consumes data and computes gradients independently 👉 aggregate gradients (e.g., sum them) to update parameters
      - *Distributed model parallel (DMP)*: if a huge model can't fit in one worker's memory (e.g., hundreds of Transformer blocks), split the model across workers 👉 each worker consumes source data or upstream outputs to compute gradients and update the parameters it owns
   - How to ensure eventual consistency
      - *Centralized parameter severs*: a parameter server stores the authoritative model parameters 👉 workers push gradients to it, the server applies updates, and workers pull the latest parameters before training the next batch
      - *Collective communication via `allreduce` (`reduce + broadcast`)*: each worker computes gradients independently 👉 aggregate gradients across workers (`reduce`) 👉 send the aggregated gradients back to all workers so they can update parameters (`broadcast`)
3. **Model serving**: Load a trained model and use it to make predictions for new inputs (realtime or batched fashion)
      - *Replication*: To handle high many concurrent queries with low latency, replicate the model across multiple model servers and use a load balancer to distribute traffic evenly
      - *Sharding*: If a request is too large for a single worker and can be decomposed (e.g., frame-level video understanding) 👉 distribute sub-requests to shards 👉 aggregate results
      - *Even-driven processing*: In systems with strong surge patterns (e.g., Uber ride requests, Airbnb bookings), create a shared resource pool that processes can borrow from during peak hours (with a rate limiter to prevent resource exhaustion)

We can use a *workflow* to connect components in an ML system, specifying their trigger logic and dependencies, typically via a directed acyclic graph (DAG). Some companies like to build in-house solutions (e.g., Meta, Pinterest), whereas others like to distribute or adopt open-source solutions such as Netflix's [Metaflow](https://docs.metaflow.org/) and Google's [Kubeflow](https://www.kubeflow.org/). 

Snap has a fantastic blogpost on [Bento](https://eng.snap.com/introducing-bento), a unified ML platform for feature and data generation, model training, and model serving. [Netflix](https://netflixtechblog.com/supercharging-the-ml-and-ai-development-experience-at-netflix-b2d5b95c63eb), [Uber](https://www.uber.com/blog/scaling-ai-ml-infrastructure-at-uber/?uclick_id=d2051111-296f-44e0-b45d-0a6bd4cc98b4), and [Pinterest](https://medium.com/pinterest-engineering/a-decade-of-ai-platform-at-pinterest-4e3b37c0f758) also wrote vividly about ML platform evolution. It's worth revisiting those posts to digest design choices.

## Interview Preparation

### Design Question Generator

In interviews, there's no way you'll be asked to sketch out the abstract system. The prompt is always grounded in a specific model use case:

- **Feed (organic + ads)**: Can you design XXX recommendations? 👉 choose from `{content, people, product}`
  - Content could be long videos (think YouTube), short videos (TikTok), posts (LinkedIn), music (Spotify), restaurants (Uber Eats), places (Google Maps), ads (CTR, CVR), to name a few
  - People could be people you may know (think LinkedIn or Facebook), artists (Spotify), colleagues (Glean), etc.
  - Product could be anything sold by the platform or sellers
- **Search**: Can you design XXX search? 👉 choose from `{consumer vs. enterprise}` × `{open-domain vs. closed-domain}` × `{conversational vs. one-off}` 
  - Feed vs. search: Many think the two are alike but their origins and goals are quite different 👉 feed makes educated guesses about what users might like, whereas search is *question answering* --- the system is strictly required to retrieve *relevant* documents to satisfy the user's *information need*.
  - Who's asking: Consumer search (e.g., Google, Perplexity, Amazon) handles huge amounts of traffic, whereas enterprise search (e.g., Glean) has to be extra permission-aware
  - Asking about what: Web search allows you to search anything (e.g., Google, Perplexity, ChatGPT), whereas e-commerce (e.g., Amazon, DoorDash, Uber Eats) or other specialty websites (e.g., Airbnb, LinkedIn) typically only allow you to search products or entities that exist on that platform
  - How to get answers: Traditional search engines returns a list of ranked documents --- if you're not happy, you have to reformulate your query (e.g., typos? too vague? too specific?) and search again; conversational search allows you to have back-and-forth chats with the system to clarify your intent or ask follow-up questions and get a good answer in a few turns
- **Other common topics**: Trust and safety (e.g., harmful content detection), `{user, content, query}` understanding, etc.
  - `{user, content, query}` understanding: Usually some sort of deep representation learning model trained with some sort of contrastive objectives (see Lilian Weng's amazing [post](https://lilianweng.github.io/posts/2021-05-31-contrastive/)) to embed single entities or entity pairs/triads/etc. 👉 from a system perspective, the interesting parts are how to do distributed training (especially if you're fine-tuning an LLM too large to to fit in worker memory), how to index trained embeddings with low storage costs without metric loss, how to version and roll back embeddings, etc. (see [Uber post](https://www.uber.com/blog/evolution-and-scale-of-ubers-delivery-search-platform/))
  - Harmful content detection: Usually some sort of multimodal, multi-task, multi-label classification model trained on a combination of `{user, annotator, LLM}`-generated labels to predict the probability of each type of harm, based on which we can take automatic or human-reviewed actions

**Note**: These aren't exactly interview questions I've personally faced. Rather, think of this list as a "generator" with a `.95` Recall of all potential questions for non-research track, general MLE/RE hires. In backend system design interviews, you could be asked to design a rate limiter, a KV store, a blob storage, or systems like Twitter, YouTube, Dropbox, or Slack --- even if you haven't worked on them. These seemingly random services tap into universal design patterns like realtime updates or scaling reads/writes. Likewise, in ML infra designs, you might be asked to design feed or trust and safety systems because they tap into common ML infra patterns like batch/online inferences or distributed training, not necessarily because you're interviewing with those teams or have related experience on your resume.

### Signals: Leadership + Time Management + Expertise

A design interview is a perfect venue to showcase leadership, time management, and communication skills --- not just domain knowledge. 

An ML infra system has many moving parts (like all distributed systems do) --- from generating and validating training data, scheduling training, to handling high QPS how it makes the most sense for your product, to name a few. You need a coherent story to tie those little pieces together and gotta sell your story telling to your interviewer. You must be assertive when the interviewer doesn't have a strong preference, but flexible when they do. That's what leadership is: influencing without authority and staying open-minded to different ideas.

Last but not least, painting a high-level picture is far from enough --- you must identify and deep dive into the most interesting parts of your system, rather than dwelling on mundane or trivial parts. That's where your time management instincts and domain knowledge shine.


## Case Study: Design Xiaohongshu Feed

> **Note**: This [blogpost](https://liuzhenglaichn.gitbook.io/system-design/news-feed/design-a-news-feed-system) is a good backend system design for (Follow) Feed. In ML interviews, however, it's a terrible idea to focus on things like how users publish posts. Instead, our focus should be on ML components: feature + data generation, model training, and model serving.

### Clarification Questions & Problem Statement

Nowadays it's rare to see a candidate jump straight into the design. More often than not, many candidates feel the urge to ask a dozen questions because they've been told that jumping straight into the design in is a red flag. I think one should ask as many questions as needed to align on a clear, reasonable scope for 45 min. Each question should either narrow the scope or clarify requirements. 

1. *Xiaohongshu has several feeds: Follow Feed, Explore Feed, and Nearby Feed. Should we focus on one or all of them?*
   - **Why ask**: A reasonable interviewer will pick one or let you choose. This spares you the pain of building 3 systems with very different candidates and optimization objectives.
2. *Should we consider both ads and organic content?*
   - **Why ask**: The real Xiaohongshu feed blends both, but it's cruel asking candidates to design ads ranking, organic ranking, and ads + organic blending in one interview. A {{< sidenote "reasonable" >}}Not all interviewers are reasonable. For instance, if your interviewer is obsessed with ads + organic blending and not great at time/scope management themself (now and at work), they might ask you to go in this direction. {{< /sidenote >}} interviewer should allow you to focus on one.
3. *How many active posts are in the corpus?*
   - **Why ask**: If the corpus is huge, you may need sharding, or must do vertical scaling in order to still co-locate the full corpus with models on the model server. And if you're doing nearest-neighbor retrieval, exhaustive KNN is unrealistic, so you need ANN.
4. *What are the average and peak QPS? What's the latency target?*
   - **Why ask**: This may be the least useful question here --- as the creator of Hello Interview [pointed out](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery), QPS is obviously high, latency obviously has to be low, and you waste time only signaling you can do division. Still, you can mention that high QPS and low latency requirements justify a multi-stage design and caching wherever possible.

**Problem statement**: We will build Xiaohongshu's Explore Feed, focusing on organic content. Explore Feed surfaces relevant notes from a corpus of billions. The system must handle high {{< sidenote "QPS" >}}Xiaohongshu has 100 million DAUs in the last known report. Suppose each DAU fetches Explore Feed 10 times a day, the average QPS is $\frac{1\mathrm{e}^9}{864000} \approx \frac{1\mathrm{e}^9}{1\mathrm{e}^5} = 10000.$  Peak QPS is usually 6x-10x. {{< /sidenote >}} (average: > 10k; peak: > 100k) while keeping end-to-end p99 latency < 500 ms. The system availability should also be high (e.g., > 99.9% uptime).

### High-Level Design

Below is a high-level design of Xiaohongshu Explore Feed (references: Instagram Explore Feed [model](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) and [infra](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/) designs).

{{< figure src="https://www.dropbox.com/scl/fi/adw63e6ubvsowmmsepo5j/Screenshot-2025-11-28-at-8.55.12-PM.png?rlkey=1uvtzqjk3p07vfqoxhs9q3av3&st=5jplk53m&raw=1" caption="A boilerplate design for (almost any) large-scale ranking systems." width="1800">}}

For me personally, it's most natural to follow the lifecycle of an online request and move into offline data ingestion and model training pipelines. I usually check in with the interviewer by saying something like: *"To structure the design, I'll start by walking through a user request and the online inference path. Then we can switch to the offline pipelines for feature and data generation and model training. Does it sound good?"*

### Online Inference: Life Cycle of a Ranking Request

1. **Client request**: The client fires a request whenever the user needs a new page of notes --- for example, when they land on the homepage, refresh it, or approach the end of the current page.
   - *Load balancing*: To handle high QPS, we use a load balancer to distribute traffic across multiple retrieval and ranking servers. We can use round-robin or least-connections algorithms for a stateless service. If we have heavy per-user cache on server memory, we can apply consistent-hashing on `userId` to maintain the state (`server = hash(userId) mod N`).

2. **Candidate generation**: We retrieve thousands of candidates (from billions) that may rank highly at later stages.
   - *Generators*: to ensure good recall and diversity, we fetch candidates from multiple generators. Some generators rely on heuristics (e.g., trending notes, notes from followed authors, the user's top interest topics). Others use similarity between the user and each item (user-to-item) or between previously engaged items and new items (item-to-item).
     - Heuristic candidates: No models needed. We can get candidates using note engagement counts (>= threshold for trending), recent notes from followed authors, and notes tagged with the user's interest topics, etc.
     - User–item or item–item similarity: perform approximate nearest neighbor (ANN) search. A common approach is training a two-tower model to learn user and item embeddings. Item embeddings are pre-computed and stored in an ANN index (e.g., HNSW or IVF-PQ). During inference, compute the user (or query-item) embedding using the embedding model hosted on the model server, and return the top-k items with highest dot products.
   - *Retrieval API*: A retrieval request consists of a `userId`, the `feedType`, and a retrieval `limit` per generators. We can fan out the request to multiple candidate generators. 
      ```typescript
      // Request
      interface GetRetrievedNotesRequest {
        userId: string;
        feedType: "FOLLOW" | "EXPLORE" | "NEARBY";
        limit?: number;           // limit per generator
      }

      // Response
      interface Note {
        noteId: string;
        authorId: string;
        recallScore?: number;   // optional
        source?: string;        // optional, e.g. "similar", "hot", "fresh", "ann"
        otherMetadata?: any;    // optional
      }

      type GetRetrievedNotesResponse = Note[];

      // API
      function getRetrievedNotes(
        req: GetRetrievedNotesRequest
      ): GetRetrievedNotesResponse {
        // implementation omitted
        return [];
      }
      ```
   - *Caching*: we can cache retrieved candidates with a short TTL to improve latency and reduce retrieval load. A TTL close to the average session length (e.g., 5–10 min) is reasonable; we shouldn't cache too long since Explore Feed changes quickly. The cache key can be `(userId, feedType)`.

3. **Lightweight ranking (L1)**: It's inefficient to rank all retrieved candidates with a heavy model, so we first use a lightweight ranker to select a few hundred items likely to rank high later.
   - *L1 ranking API*: An L1 ranking request consists of a `userId` and an unordered list of retrieved documents (thousands).
      ```typescript
      // Request
      interface L1RankNotesRequest {
        userId: string;
        candidates: Note[];        // retrieval output (e.g., 2k–10k items)
        topK?: number;             // default: 500
      }

      // Response item for L1 (low-cost features, lightweight model)
      interface L1RankedNote extends Note {
        l1Score: number;           // score from L1 model
        position: number;          // rank among L1 results
      }

      type L1RankNotesResponse = L1RankedNote[];

      // API
      function rankNotesL1(req: L1RankNotesRequest): L1RankNotesResponse {
        // implementation omitted
        return [];
      }

      ```
   - *Feature hydration*: Fetch features for `userId` and `noteId`
     - User features: For one user's request, only fetch user features once. If a batch request contains multiple users, some of which are the same, we can first look up features for unique users and then broadcast them to all requests ([Meta](https://arxiv.org/html/2508.05640v1) calls this "request-only optimization", and [Pinterest](https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d) calls it "request-level broadcasting").
     - Item features: We must fetch item features for thousands of items. To reduce latency, we can co-locate the document feature store with the model server so we don't need to fan out hydration requests to another service. If the model server doesn't have enough capacity, we can vertically scale it (add memory).
   - *Caching*: We can cache features (keyed by each feature's key) as well as L1 ranking results (keyed by `(userId, feedType, modelVersion)`). User features change quickly with in-session activities --- we can cache them for a short TTL, like a minute. Item features are more static --- we can cache them for longer, like an hour. We can cache L1 ranking results for one minute (same as user feature cache TTL).

4. **Reranking (L2)**: Finally, we use a heavy ranker to select the best couple dozen items from the hundreds of L1 candidates. The L2 ranking API and caching strategies are similar to L1.
   - *L2 ranking API*: An L2 ranking request consists of a `userId` and an unordered list returned by the L1 ranker (hundreds). 
      ```typescript
      // Request
      interface L2RankNotesRequest {
        userId: string;
        candidates: L1RankedNote[];   // output of L1 (e.g., 500 items)
        topK?: number;                // default: 20
      }

      // Response item for L2 (full features + heavier ML model)
      interface L2RankedNote extends L1RankedNote {
        finalScore: number;           // score from L2 model
        position: number;             // final rank (1..20)
      }

      type L2RankNotesResponse = L2RankedNote[];

      // API
      function rankNotesL2(req: L2RankNotesRequest): L2RankNotesResponse {
        // implementation omitted
        return [];
      }

      ````
   - *Optional final ranking*: After L2, we can apply a final layer of business or quality rules (e.g., diversity, trust & safety filters, promotions). I treat this as out of scope for the ranking service itself and part of a separate policy / blending layer.
5. **Logging and monitoring**: To monitor the current model's performance and train future models, we need to log feature values, model predictions, and user engagement for each request.
   - *Model quality monitoring*: For retrieval and L1 ranking, we can monitor the survival rate --- i.e., out of all returned candidates, how many made it to the next stage. For L2 ranking, we can compare model predictions with engagement labels and compute ranking metrics such as normalized cross entropy (NCE), AUC, PR-AUC, nDCG@k, MRR, MAP@k, etc.
   - *Data and feature monitoring*: We should monitor changes in data volume, feature and label distributions, and pipeline staleness. It's bad to train or serve with stale data, and it's dangerous to train or infer on incorrect or corrupted data.
   - *System performance monitoring*: We should also monitor the latency, throughput, CPU/CPU utilization of each service.
   - *Logging*: For each request, we can immediately log feature values and model predictions. Some user engagements may be delayed (e.g., I might read a hilarious note in the afternoon but only share it with a friend before bed). Engagements are usually logged separately, after they happen, and later joined with features on `impression_id` or some other keys that reliably link labels back to the original request.

### Offline Processing: Generate Data to Train Models

1. Offline feature pipeline
2. Realtime feature pipeline
3. Training data pipeline
4. Model training pipeline
5. Model deployment pipeline 

# Interview Type 2: Component Designs

Which one feels scarier: Spending an hour glancing over an end-to-end ML system, or spending 45 minutes digging into a single component? I imagine the main risk with the former is running out of time --- which is fixable with better practice and structure, whereas the main risk with the latter is running out of knowledge --- which you can't do much about in the short term. IMO, the latter is a good failure to have because it informs you what to learn more about in the future.

## Offline Feature + Data Generation

> [...] we will never know if or when we have seen all of our data, only that new data will arrive, old data
may be retracted, and the only way to make this problem tractable is via principled abstractions that allow the practitioner the choice of appropriate tradeoffs along the axes of interest: correctness, latency, and cost [...]. Since money is involved, correctness is paramount. --- [*The Data Flow Model (2015)*](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/)

### Context: What Data Do We Need?

In many ranking applications, the model maps a feature vector $\mathbf{x}$ to a binary label $y$ (i.e., $f: \mathbf{x} \mapsto y,; y \in {0,1}$). Features arrive first --- they're fetched from Feature Stores or computed on the fly and then passed into the model. Model predictions follow, typically tens of milliseconds later, and are used to produce the ranking. Labels arrive seconds, minutes, or even days or weeks later, after users have engaged (e.g., clicked an ad, watched a video, purchased a product, reported a post, or did nothing) with the ranked results. 

<!-- features 
item, user, context, cross

scalar: count, ratio
ID
embedding
timestamps
list of ID
list of scalars
list of embs -->

Where does the model fetch features from? Some features are pre-computed and stored in a Feature Store (usually a distributed KV store) and updated on a regular basis (e.g., daily), such as the average CTR or total clicks between this user and this advertiser in the past 24 hours, 3 days, 7 days, 30 days, and so son. Some features are computed on the fly, such as the BM25 score between a query and a document. Some features are pre-computed but updated in a near realtime fashion, such as user action sequences or action counts. 

For model training, we need to stitch together (features $\mathbf{x}$, label $y$) pairs for each impression, or at least impressions we sample. Labels come from event logging and features can be obtained in 2 ways:

- **Forward logging**: When we log an engagement event, we can also log its features. If we must log all events (e.g., for ads) but don't have space for all their features, we can use a separate logging job to record features for only a fraction of impressions. The advantage is that forward logging mitigates online-offline discrepancies in feature values. The downside is that when you have new features, you must wait X days to get X days of training data with all features. If X is large (e.g., 90 days), it hurts velocity.
- **Backfill**: As mentioned, some features are fetched from the Feature Store while others are computed on the fly. If we can find the correct feature values at {{< sidenote "impression time" >}}Prediction time is more accurate, but not usually logged and available for join.{{< /sidenote >}}, then we can join them with labels (e.g., on `impression_id`) to build training data. The pro is that we don't need to wait for forward logging. The downside is that even the most experienced engineers can suffer from data leakage by joining "future" feature values --- values computed after the impression time --- with that impression's label. See this [blogpost](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) for a cautionary example.

To debug model performance issues, we need the full (features $\mathbf{x}$, prediction $\hat{y}$, label $y$) triads in order to examine how much model evaluation metrics (computed on ($\hat{y}$, $y$) pairs) change if we tweak the current problematic model or use another model to make predictions.

### Problem Statement: Unified Feature Platform

Even just a few years ago, even at early ML adopters like Pinterest, it was common for ML engineers to write their own feature code for offline iterations, while backend engineers translated it into Java/C++/Go for online serving (see Pinterest's ML infra [blogpost](https://medium.com/pinterest-engineering/a-decade-of-ai-platform-at-pinterest-4e3b37c0f758)). Even today, ML engineers often write feature pipelines for backfill, while backend or infra engineers implement forward logging and realtime feature updates. These inconsistent implementations may lead to online–offline discrepancies and hinder iteration velocity.

Many modern ML teams have built unified feature platforms that support consistent and efficient feature generation across forward logging, backfill, and realtime updates (Snap [blog](https://eng.snap.com/speed-up-feature-engineering)). Let's aim for this.

- **Functional requirements**: The Feature Store needs to support multiple common feature generation methods
   - *Forward log*: Must be able to log inference feature values
   - *Batch backfill*: Must be able to generate new or expensive features offline and join them with other features to create training data with point-in-time correctness
   - *Real-time updates*: Must be able to update frequently changing features with near-realtime cadence
- **Non-functional requirements**: The Feature Store must allow easy feature specification as well as fast and correct feature generation
   - *Usability*: Feature users can define new features via a [declarative language](https://en.wikipedia.org/wiki/Declarative_programming) rather than manually deploying code
   - *Velocity*: Feature users shouldn't have to wait weeks to forward log new features for training
   - *Scale*: The system should be able to process thousands of aggregation features from billions of events per day

### High-Level Design: From Events to Features

> Users specify a *map* function that processes a key/value pair to generate a set of intermediate key/value pairs, and a *reduce* function that merges all intermediate values associated with the same intermediate key. Many real world tasks are expressible in this model [...]. --- [*MapReduce (2004)*](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/)

{{< figure src="https://www.dropbox.com/scl/fi/b691em8ao42vn4vvbgtot/Screenshot-2025-11-29-at-4.56.09-PM.png?rlkey=xgo0hxjns2h2ubz1hsls799jx&st=3sfjgsnc&raw=1" caption="Robusta, Snap's feature engineering platform (source: [Snap eng blog](https://eng.snap.com/speed-up-feature-engineering))." width="1800">}}

Say we want to know for each snap, how many views it got in the last 6 hours (`snap_view_count_last_6h`). Let's look at 3 scenarios:
- Realtime updates: What happens when a *view event* happens?
- Online path: What happens when we *serve a request online*?
- Offline path: What happens when we *build training data offline*?

1. **Declarative feature spec**: First, we need to allow MLE users to define this feature with online + offline paths
   - Config: User define features by specifying a few parameters 
      - Aggregation type: e.g., <span style="background-color: #FFC31B">`count`</span>, `sum`, `last_n`, `approx_quantile`
      - Keys to group by: e.g., <span style="background-color: #FFC31B">`DOCUMENT_ID`</span>, `USER_ID`, `HOUR_OF_DAY` --- can select a primary key
      - Window granularity: e.g., 5 min, 1h, <span style="background-color: #FFC31B">6h</span>, 30d, etc.
      - Filters: e.g., view duration, view day of week
   - Execution: The engine will compute aggregations you specify via associate + communicative operations
      - Fundamental assumptions: `sum` meets both
         - Associative: Grouping doesn't change results
         - Communicative: Reordering doesn't change results
      - The engine later computes the following aggregations
        - `map(event) -> rep`: map each view event to `rep = 1` (each view contributes a value of 1)
        - `combine(rep_a, rep_b) -> rep_ab`: combine reps by `sum` (associative + communicative)
        - `finalized(rep) -> feature_value`: identity --- just return the `sum` (total views in a window)
2. **Event stream**: Ingest, clean, and store engagement events
   - When this view happens, the online service logs an event --- 
      ```json
      {
        "event": "view",
        "user_id": "U123",
        "snap_id": "S456",
        "surface": "spotlight",
        "event_ts": "2025-11-29T16:03:10Z",
        "request_id": "R999"
      }

      ```
   - Data ingestion: The event goes to a Kafka topic (e.g., `snap_views`), which gets periodically written to an Iceberg table (e.g., `snap_views_raw`) partitioned by date/hour
   - Team-specific tables: Each team can define a Spark table with custom schema, filtering, dedup, transformation, etc. 👉 each row is a clean event `(snap_id, event_time, …)`
      ```sql
      CREATE VIEW spotlight_snap_view_data AS
      SELECT
        user_id,
        snap_id,
        ts AS event_time
      FROM snap_views_raw
      WHERE surface = 'spotlight';

      ```
3. **Aggregation engine**: To avoid duplicate work, the engine computes pre-aggregated blocks at various granularities and saves them to Iceberg, which are building blocks of all jobs
   - Streaming job: Computes pre-aggregated blocks at 5-min intervals --- the goal is to ensure freshness
      - For each incoming row `(snap_id, event_time)`
         - Get its 5-min bucket --- in this case, `16:00–16:05`
         - Increment view count in bucket: `rep_block(snap_id, 16:00–16:05) += 1`
      - Write the new or updated block to
         - Iceberg table (e.g., `snap_view_blocks_5m`)
         - Online features store (e.g., Aerospike)
   - Batch job: Computes pre-aggregated blocks at coarser intervals (e.g., 1h) --- the goal is to ensure completeness
      - Every hour/day, run job to 
         - Re-scan the raw event table `snap_views_raw`
         - Recompute block view counts
            - 5 min blocks for correction, if needed
            - 1h blocks by summing 12 x 5 min blocks
      - Write new or updated blocks to Iceberg + Feature Store
         - Each row is `(primary_key, window_start, window_size, feature_id, representation_blob)`
         - The table is partitioned by `(primary_key_shard, window_start_bucket)` so that
            - Online reads can fetch blocks for given key
            - Offline jobs can scan by shard + time range
4. **Online path**: Fetch latest feature values for the current request
      - Say a request comes in at 16:10 for `snap_id = S456`
      - Option 1: Assemble from intermediate blocks upon request 
         - We need to fetch blocks to cover the last 6 hours, dating back from the request time: `[10:10, 16:10)`
         - Select non-overlapping blocks that best cover this range
            - 1h blocks: `11:00–12:00`, `12:00–13:00`, `13:00–14:00`, `14:00–15:00`, `15:00–16:00`
            - 5 min blocks: `10:10–10:15` to `10:55–11:00`, `16:00–16:05`, `16:05–16:10`
         - Combine counts in those blocks to get the final answer
         - When to use: For large or rare features we don't want to pre-materialize, we can sacrifice some latency for space
      - Option 2: Pre-assemble in the feature store
         - Periodically scan `snap_view_blocks_5m` to compute the full feature value for each `snap_id`
         - Write `(snap_id, snap_view_count_last_6h, watermark_ts)` to a KV store or a document index file
         - When to use: If we score the same `snap_id` many times, we should pre-assemble feature values to reduce latency, at the expensive of slight staleness --- for most ranking candidate features, we should do this if possible
      - Forward logging: Regardless of how we get the feature, we can log it with other features and model predictions
         ```json
         {
           "impression_id": "IMP123",
           "user_id": "U999",
           "snap_id": "S456",
           "event_ts": "2025-11-29T16:10:05Z",
           "model_score": 0.78,
           "features": {
             "snap_id_view_count_last_6h": 137,
             ...
           },
           "feature_watermark": "2025-11-29T16:05:00Z"  // up to which blocks we trust
         }
         ```
5. **Offline path**: If this is a new feature we've yet to forward log or has wrong serving values we want to fix, we can generate feature values that the model would've seen online (point-in-time correctness) and join them with labels for training
   - Offline feature generation via point-in-time lookup
      - The impression's `feature_watermark` is 16:05 --- that means, the largest time bucket used for online prediction was 16:05 👉 to prevent data leakage, we should not use newer buckets than this 
      - Search for 1h + 5m pre-aggregated blocks from Iceberg to cover the range `[10:10, 16:05)`
      - Sum counts in these buckets to generate `snap_id_view_count_last_6h_offline`
   - Training data generation via bucketed join
      - We can join impression logs with offline features
         - LHS (left-hand side): impression logs bucketed by `snap_id`
         - RHS (right-hand side): offline-generate feature table `(snap_id, impression_id, snap_id_view_count_last_6h_offline, ...)`
      - Join on `(snap_id, impression_id)` within buckets
      - The final table has all features needed to train new models: `[user_id, snap_id, ts, label, snap_id_view_count_last_6h_offline, other features...]`


**Note**: The design above is based on Snap's blog post. It's clever and more complex than any feature platforms I've seen. In my last weeks at DoorDash, someone asked in the ML platform channel whether we could avoid reloading last 30 days' data every day to recompute a 30-day aggregations. People thought it was a good idea but couldn't be done. Snap's idea of pre-computing aggregation blocks and assembling them into arbitrary windows is ingenious. My colleague asked that question in late 2024, and Snap had been doing this before 2022.

### Deep Dive #1: Backfill

### Deep Dive #2: Co-Location

offline vs. online code

items: 
co-location with inference engine: mentioned in Snap's and Pinterest's blogs


forward fill
backward fill: pinterest blogpost
intrainer join

## Real-Time Features

Kafka, Flink

## Distributed Training

## GPU Serving
goal: keep GPU busy
Model Flops Utilization (MFU)
continuous batching from HF

# References

## End-to-End Systems
### Abstract ML Systems
To begin, get an abstract overview of end-to-end ML systems:

> Many books have been written on either machine learning or distributed systems. However, there is currently no book available that talks about the combination of both and bridges the gap between them. --- *Distributed Machine Learning Patterns*

1. [Distributed Machine Learning Patterns](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022) 👉 this is a *fantastic* book about ML infra. Some folks dismiss it because the author still uses (1) TensorFlow and (2) the Fashion-MINST dataset to illustrate the model component, but I bet they didn't read the book. 
   - (1) is understandable because the author was a main contributor of *Google*'s Kubeflow. Do we expect PyTorch? 🤣
   - (2) is the point --- the author wants readers to build an end-to-end ML infra system on their own machine and has provided probably the cleanest and most vividly explained instructions on how to do so. A toy dataset is what fits.
2. Metaflow 👉 Netflix's open-source framework for model training and inference: [Why Metaflow](https://docs.metaflow.org/introduction/why-metaflow), [paper](https://arxiv.org/abs/2303.11761), [blog](https://netflixtechblog.com/supercharging-the-ml-and-ai-development-experience-at-netflix-b2d5b95c63eb), [tech overview](https://docs.metaflow.org/internals/technical-overview)
3. [Introducing Bento, Snap's ML Platform](https://eng.snap.com/introducing-bento) 👉 Snap's all-in-one ML platform for feature engineering, training data generation, model training, and online inference
4. [Scaling AI/ML Infrastructure at Uber](https://www.uber.com/blog/scaling-ai-ml-infrastructure-at-uber/?uclick_id=d2051111-296f-44e0-b45d-0a6bd4cc98b4) 👉 evolution of Uber's Michaelangelo platform

### Model Use Cases
Then, dig into ML systems designed for specific models:

5. [Evolution and Scale of Uber's Delivery Search Platform](https://www.uber.com/blog/evolution-and-scale-of-ubers-delivery-search-platform/) 👉 Uber Eats' search platform
6. [Introducing DoorDash's In-House Search Engine](https://careersatdoordash.com/blog/introducing-doordashs-in-house-search-engine/) 👉 Argo, DoorDash's search platform; [debugging](https://careersatdoordash.com/blog/doordash-optimizing-in-house-search-engine-platform/)
7. [Embedding-Based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval) 👉 Snap's retrieval model for spotlight videos 
8. [Snap Ads Understanding](https://eng.snap.com/snap-ads-understanding) 👉 Snap's video ad content understanding model
9. [Machine Learning for Snapchat Ad Ranking](https://eng.snap.com/machine-learning-snap-ad-ranking) 👉 Snap's pCVR model
10. `ApplyingML` 👉 [blogposts](https://applyingml.com/resources/) on ML model and system designs 

## ML Infra Components
### Offline Feature + Data Generation
11. [The Dataflow Model](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/) 👉 Google's seminal paper on large-scale data generation
12. [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) 👉 Jeff Dean's original MapReduce paper
13. [Point-in-Time Correctness in Real-Time Machine Learning](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) 👉 data leakage prevention
14. [Speed Up Feature Engineering for Recommendation Systems](https://eng.snap.com/speed-up-feature-engineering) 👉 Robusta, Snap's feature pipeline
15. [Michelangelo Palette: A Feature Engineering Platform at Uber](https://www.infoq.com/presentations/michelangelo-palette-uber/#:~:text=Michelangelo%20Palette%20is%20essentially%20a,models%20and%20why%20is%20that%3F) 👉 Palette, Uber's feature pipeline and feature store
16. [Zipline --- Airbnb's ML Data Management Framework](https://conferences.oreilly.com/strata/strata-ny-2018/cdn.oreillystatic.com/en/assets/1/event/278/Zipline_%20Airbnb_s%20data%20management%20platform%20for%20machine%20learning%20Presentation.pdf) 👉 Zipline, Airbnb's feature pipeline and feature store
17. [Building a Spark-Powered Platform for ML Data Needs at Snap](https://eng.snap.com/prism) 👉 Prism, Snap's training data pipeline

### Real-Time Features
18. [Feature Infrastructure Engineering: A Comprehensive Guide](https://mlfrontiers.substack.com/p/feature-infrastructure-engineering) 👉 Samuel Flender's new blogpost on real-time signals
19. [How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8) 👉 real-time user aggregation features
20. [Large-scale User Sequences at Pinterest](https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9) 👉 real-time user sequence features

### Distributed Training
21. [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 👉 train LLMs on GPU clusters
22. [How to Train a Model on 10k H100 GPUs?](https://soumith.ch/blog/2024-10-02-training-10k-scale.md.html) 👉 PyTorch author's short blogpost
23. [Training Large-Scale Recommendation Models with TPUs](https://eng.snap.com/training-models-with-tpus) 👉 Snap has been using Google's TPUs since 2022

### GPU Serving
24. [Applying GPU to Snap](https://eng.snap.com/applying_gpu_to_snap) 👉 Snap's switch from CPU to GPU serving
25. [GPU-Accelerated ML Inference at Pinterest](https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d) 👉 Pinterest did the same a year later
26. [Introducing Triton: Open-Source GPU Programming for Neural Networks](https://openai.com/index/triton/) 👉 OpenAI's Triton kernels
27. [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 👉 CUDA graphs are often used in GPU serving