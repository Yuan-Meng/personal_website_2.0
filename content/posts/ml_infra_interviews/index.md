---
title: Preparing for ML Infra System Design Interviews
date: 2025-12-01
math: true
categories: ["career", "ml infra", "interview"]
toc: true
---

# Dilemma: Model Builders != Infra Builders

A handful of companies like Netflix, Snap, Reddit, Notion, and DoorDash have an ML infra system design round for MLE candidates --- in addition to standard ML system design. Maybe you'll never have to interview with them. However, apart from the frontier AI {{< sidenote "labs" >}}If you do get an offer from a frontier AI lab but outside of the research org, you don't necessarily get paid more than an MLE at Netflix or Snap at the same level (e.g., OpenAI L4 Research Engineer vs. Snap/Netflix L5 MLE).{{< /sidenote >}} (e.g., OpenAI, Anthropic, xAI, DeepMind, Meta TBD, Thinking Machines Lab, Reflection), the first two pay more than other companies are able to match at the same level. So I feel that many talented MLEs are incentivized to pass their interviews at some point in their careers.

ML system design focuses on translating business objectives into ML objectives, choosing training data, labels, and model architectures, and evaluating models offline and online. By contrast, <span style="background-color: #D9CEFF">ML infra system design focuses on the offline + online pipelines that support an ML system</span>. One type of question asks you to walk through full online + offline pipelines; another asks you to design specific components, such as a feature store, real-time feature updates, or distributed training.

Here's the funny thing: If your company serves truly large-scale recommender systems (e.g., recommending billions of items to hundreds of millions of DAUs), you're likely working with dedicated infra teams that handle logging, training, and inference for you. Your job is to optimize a narrow set of North Star metrics your team is funded for (e.g., CTR, CVR, revenue, search relevance). If your company isn't making recommendations at scale, the knowledge of how to build scalable ML systems may be years away from your reality.

That said, I do think ML infra interviews are valuable: Modern recommender system teams function like Formula One teams --- even if you ask Max Verstappen to build a car, he likely can't it to save his life, but no driver on the grid doesn't have an intimate knowledge of car mechanics. The best drivers have a fantastic feel for which parts are or aren't working and collaborate with technicians to improve the car throughout a season. Similarly, the best ML engineers can make necessary, timely, and reasonable asks to their ML infra partners well ahead of major projects. So, solid ML infra knowledge goes a long way in an impactful career. Therefore, even if you never take an ML infra interview, you should still spend time learning this knowledge.

# Interview Type 1: Pipeline Walk-Through

## A Bare-Bone ML System

A bare-bone ML system consists of the following components:

{{< figure src="https://www.dropbox.com/scl/fi/uvptgcmfbhjs2o5y58i0e/Screenshot-2025-11-28-at-6.54.39-PM.png?rlkey=4oop20e6lbho45elw1ut46z2q&st=c3hwud96&raw=1" caption="A bare-bone ML system ([Distributed Machine Learning Patterns](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022), Chapters 2-6)." width="1800">}}

1. **Data ingestion**: Consume training data (features + labels; metadata) either all at once or in a streaming fashion 👉 preprocess the data so they are ready to be fed into the model
   - *Batching*: We usually can't load the entire training set at once 👉 split it into mini-batches and train on one batch at a time
   - *Sharding*: Extremely large datasets may not fit on a single machine 👉 shard data across multiple machines and let each worker consume batches from assigned data shards
   - *Caching*: If we read data from a remote source (e.g., a database or datalake) or have expensive preprocessing, we can cache preprocessed data in RAM for future epochs
2. **Model training**: Initialize a model, train it on ingested data in a distributed fashion (using DDP or DMP), and save checkpoints
   - How to distribute model training
      - *Distributed data parallel (DDP)*: Copy the full model to multiple workers 👉 each worker consumes data and computes gradients independently 👉 aggregate gradients (e.g., sum them) to update parameters
      - *Distributed model parallel (DMP)*: If a huge model can't fit in one worker's memory (e.g., hundreds of Transformer blocks), split the model across workers 👉 each worker consumes source data or upstream outputs to compute gradients and update the parameters it owns
   - How to ensure eventual consistency
      - *Centralized parameter severs*: A parameter server stores the authoritative model parameters 👉 workers push gradients to it, the server applies updates, and workers pull the latest parameters before training the next batch (<span style="background-color: #FFC31B">not used in modern systems anymore!</span>)
      - *Collective communication via `allreduce` (`reduce + broadcast`)*: Each worker computes gradients independently and sends gradients to all other workers 👉 each worker aggregates gradients from all other workers (`reduce`) 👉 each worker sends aggregated gradients all other workers (`broadcast`)
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
  - Content could be long videos (think YouTube), short videos (TikTok), posts (LinkedIn), music (Spotify), restaurants (Uber Eats), places (Google Maps), ads (CTR, CVR), notifications, promotions, query suggestions, to name a few
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

> **Note**: These aren't exactly interview questions I've personally faced. Rather, think of this list as a "generator" with a `.95` Recall of all potential questions for non-research track, general MLE/RE hires. In backend system design interviews, you could be asked to design a rate limiter, a KV store, a blob storage, or systems like Twitter, YouTube, Dropbox, or Slack --- even if you haven't worked on them. These seemingly random services tap into universal design patterns like realtime updates or scaling reads/writes. Likewise, in ML infra designs, you might be asked to design feed or trust and safety systems because they tap into common ML infra patterns like batch/online inferences or distributed training, not necessarily because you're interviewing with those teams or have related experience on your resume.

### Signals: Leadership + Time Management + Expertise

A design interview is a perfect venue to showcase leadership, time management, and communication skills --- not just domain knowledge. 

An ML infra system has many moving parts (like all distributed systems do) --- from generating and validating training data, scheduling training, to handling high QPS how it makes the most sense for your product, to name a few. You need a coherent story to tie those little pieces together and gotta sell your story telling to your interviewer. You must be assertive when the interviewer doesn't have a strong preference, but flexible when they do. That's what leadership is: influencing without authority and staying open-minded to different ideas.

Last but not least, painting a high-level picture is far from enough --- you must identify and deep dive into the most interesting parts of your system, rather than dwelling on mundane or trivial parts. That's where your time management instincts and domain knowledge shine.

## Case Study: Design Xiaohongshu Feed

> **Note**: This [blogpost](https://liuzhenglaichn.gitbook.io/system-design/news-feed/design-a-news-feed-system) is a good backend system design for (Follow) Feed. In ML interviews, however, it's a terrible idea to focus on things like how users publish posts. Instead, our focus should be on ML components: feature + data generation, model training, and model serving.

### Clarification Questions & Problem Statement

Nowadays it's rare to see a candidate jump straight into the design. More often than not, many candidates feel the urge to ask a dozen questions because they've been told that jumping straight into the design in is a red flag. I think one should ask a handful of questions essential for aligning on a clear, reasonable scope for 45 min. Each question should either narrow the scope or clarify requirements. 

1. *Xiaohongshu has several feeds: Follow Feed, Explore Feed, and Nearby Feed. Should we focus on one or all of them?*
   - **Why ask**: A reasonable interviewer will pick one or let you choose. This spares you the pain of building 3 systems with very different candidates and optimization objectives.
2. *Should we consider both ads and organic content?*
   - **Why ask**: The real Xiaohongshu feed blends both, but it's cruel asking candidates to design ads ranking, organic ranking, and ads + organic blending in one interview. A {{< sidenote "reasonable" >}}Not all interviewers are reasonable. For instance, if your interviewer is obsessed with ads + organic blending and not great at time/scope management themself (now and at work), they might ask you to go in this direction. {{< /sidenote >}} interviewer should allow you to focus on one.
3. *How many active posts are in the corpus?*
   - **Why ask**: If the corpus is huge, you may need sharding, or must do vertical scaling in order to still co-locate the full corpus with models on the model server. And if you're doing nearest-neighbor retrieval, exhaustive KNN is unrealistic, so you need ANN.
4. *What are the average and peak QPS? What's the latency target?*
   - **Why ask**: This may be the least useful question here --- as the creator of Hello Interview [pointed out](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery), QPS is obviously high, latency obviously has to be low, and you waste time only signaling you can do division. Still, you can mention that high QPS and low latency requirements justify a multi-stage design and caching wherever possible.
5. *For feed, it's important that we remove harmful content and ensure diversity. Do we need to focus on these considerations today?*
   - **Why ask**: Feed is not just about ranking content by `pAction`, but providing a good experience by showing relevant, safe, and diverse contents to users. It can be important to show such awareness and know some re-ranking methods (e.g., Xiaohongshu published a [paper](https://arxiv.org/abs/2107.05204) on diversity). However, don't start with this part and only deep dive if you've already presented a complete ranking solution.

**Problem statement**: We will build Xiaohongshu's Explore Feed, focusing on organic content. Explore Feed surfaces relevant notes from a corpus of billions. The system must handle high {{< sidenote "QPS" >}}Xiaohongshu has 100 million DAUs in the last known report. Suppose each DAU fetches Explore Feed 10 times a day, the average QPS is $\frac{1\mathrm{e}^9}{864000} \approx \frac{1\mathrm{e}^9}{1\mathrm{e}^5} = 10000.$  Peak QPS is usually 6x-10x. {{< /sidenote >}} (average: > 10k; peak: > 100k) while keeping end-to-end p99 latency < 500 ms. The system availability should also be high (e.g., > 99.9% uptime).

### High-Level Design

For me personally, it feels natural to start with the online path by following a request lifecycle and move to the offline path including data ingestion and model training. I'll check in with the interviewer: *"To structure the design, I'll start by walking through a user request and the online inference path. Then we can switch to the offline pipelines for feature and data generation and model training. Does it sound good?"*

Below is a high-level design of Xiaohongshu Explore Feed (references: Instagram Explore Feed [model](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) and [infra](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/) designs).

{{< figure src="https://www.dropbox.com/scl/fi/adw63e6ubvsowmmsepo5j/Screenshot-2025-11-28-at-8.55.12-PM.png?rlkey=1uvtzqjk3p07vfqoxhs9q3av3&st=5jplk53m&raw=1" caption="A boilerplate design for (almost any) large-scale ranking systems." width="1800">}}

### Online Path: Life Cycle of a Feed Request

1. **Client request**: The client fires a request whenever the user needs a new page of notes --- for example, when they land on the homepage, refresh it, or approach the end of the current page.
   - *Load balancing*: To handle high QPS with low latency, we can use a load balancer to distribute traffic across multiple retrieval and ranking servers. We can use round-robin or least-connections algorithms, which are stateless and good for most ranking use cases. If we have heavy per-user cache on server memory, we can apply consistent-hashing on `userId` to route the same user to the same server (`server = hash(userId) mod N`).
   - *Hot key handling*: Consistent hashing may create hot keys (e.g., if a user is super active or a post is super popular). If this occurs, we can (1) re-consider using stateless algorithms, (2) switch to another key to hash, (3) use a compound key (e.g., `postId-countryId`), or (4) add a [random salt](https://en.wikipedia.org/wiki/Salt_(cryptography)) to the hash.

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

### Offline Path: Generate Data to Train Models

1. **Offline feature generation**: Some features are expensive to compute (e.g., outputs from a large model) and don't require realtime updates, such as user and content understanding embeddings. We can compute these embeddings on a daily basis (e.g., via a cron job) and upload versioned embeddings to the Feature Store (e.g., Cassandra). Aggregation features, such as counts or ratios for a user, a piece of content, or a user–content pair over the last N hours or N days, can be pre-computed offline on an hourly or a daily basis, but we should allow realtime updates to those features when new engagements occur.
2. **Realtime feature generation**: Some features come from the client (e.g., `userId`, `deviceId`, `deviceType`, `geohash`, `country`) --- these can only be logged at inference time. Other features may need frequent updates, such as engagement counts or ratios. To generate realtime features, we typically combine a distributed messaging system like Kafka with a stream processing engine like Flink. When an engagement occurs, the client or a backend service logs an event to a Kafka topic (e.g., `user_post_clicks`). Consumers read these topics, and Flink processes the event stream to compute realtime features (e.g., post clicks in the last 5 minutes). These features can then be written to the Feature Store for serving. We can also incrementally update longer-range features --- such as hourly or daily post-click counts --- using the same realtime stream.
3. **Training data pipeline**: For most features, we can log feature values together with user actions for each request in the engagement logs. This forward logging approach is cheap and guarantees that feature values match exactly what the model saw during inference. However, if we introduce new features, want to fix wrong features, or have features that are too expensive to log (e.g., a user's lifelong action sequences), then we need to backfill those features offline and join them with engagement labels. To prevent data leakage, we should only use the latest feature values *before* the impression time (or a feature watermark indicating the latest feature timestamp used, in the "fix wrong feature" case).
4. **Model training pipeline**: Once all required features are joined with labels, we can use the dataset to train models. How many days of data to use depends on how much compute and training time we can afford. Using a wider date range usually improves generalization, and therefore test metrics, but requires more GPU workers and longer training. For feed ranking models, it's common to use at least a few weeks of data or even a few months. We can use distributed training to increase throughput. For ranking models, we typically use Distributed Data Parallel (DDP), where multiple workers process different batches simultaneously, compute local gradients, and then aggregate them via `all-reduce` (each worker reduces gradients from peers and then receives the fully aggregated gradients) before applying updates. We should save model checkpoints and optimizer states frequently to ensure fault-tolerant training. It's also important to track which batches have already been consumed so that, when resuming from a checkpoint, we don't repeat data and overfit.
5. **Model deployment pipeline**: After training, raw PyTorch models are typically exported to TorchScript to optimize for inference. Before deploying a new model, we should verify that offline metrics such as normalized cross-entropy (1 − treatment CE / baseline CE), AUC, and PR-AUC are not degraded. For low latency inference, the model needs to remain in memory on the server. To avoid downtime, we can deploy the model to a fresh set of servers and gradually shift traffic to them. During rollout, we should closely monitor online metrics such as post CTR, share rate, hide rate, and other engagement signals to ensure model quality. If online metrics degrade, we should immediately roll back. For models with new architectures or features, we should run an A/B experiment and only roll out the new model design if we see statistically significant improvements with no or justifiable degradation in business or performance metrics.

# Interview Type 2: Component Designs

Which one feels scarier: Spending an hour glancing over an end-to-end ML system, or 45 minutes digging into a single component? I imagine the main risk with the former is running out of time --- which is fixable with more practice and better structure, whereas the main risk with the latter is running out of knowledge --- which you can't do much about in the short term. IMO, the latter is a good failure to have because it informs you what to learn more about in the future.

## Offline Feature + Data Generation

### Context: What Data Do We Need?

In many ranking applications, the model maps a feature vector $\mathbf{x}$ to a binary label $y$ (i.e., $f: \mathbf{x} \mapsto y; y \in {0,1}$). Features arrive first --- they're fetched from Feature Stores or computed on the fly and then passed into the model. Model predictions follow, typically tens of milliseconds later, and are used to produce the initial ranking. Labels arrive seconds, minutes, or even days or weeks later, after users have engaged (e.g., clicked an ad, watched a video, purchased a product, reported a post, or did nothing) with the ranked results. 

Where does the model fetch features from? Some features, such as the CTR or total clicks between this user and this advertiser in the past 24 hours, 3 days, 7 days, 30 days, etc., are pre-computed and stored in a Feature Store (usually a distributed KV store) updated in batch or streaming fashion. Some features are computed on the fly, such as the BM25 score between a query and a document. 

For model training, we need to stitch together (features $\mathbf{x}$, label $y$) pairs for each impression, or at least impressions we sample. Labels come from event logging and features can be obtained in 2 ways:

- **Forward logging**: When we log an engagement event, we can also log its features. If we must log all events (e.g., for ads) but don't have space for all their features, we can use a separate logging job to record features for only a fraction of impressions. The advantage is that forward logging mitigates online-offline discrepancies in feature values. The downside is that when you have new features, you must wait X days to get X days of training data with all features. If X is large (e.g., 90 days), it hurts velocity.
- **Backfill**: As mentioned, some features are fetched from the Feature Store while others are computed on the fly. If we can find the correct feature values at {{< sidenote "impression time" >}}Prediction time is more accurate, but not usually logged and available for join.{{< /sidenote >}}, then we can join them with labels (e.g., on `impression_id`) to build training data. The pro is that we don't need to wait for forward logging. The downside is that even the most experienced engineers can suffer from {{< sidenote "data leakage" >}}I've witnessed an entire team's scope gone because the engineer who was supposed to lead their "flagship" model joined features with labels incorrectly and the model performance flopped... Then the project was handed over to an engineer on the sister team, who shifted the join date and delivered the model in a few weeks. {{< /sidenote >}} by joining "future" feature values --- values computed after the impression time --- with that impression's label. See this [blogpost](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) for a cautionary example.

To debug model performance issues, we need the full (features $\mathbf{x}$, prediction $\hat{y}$, label $y$) triads in order to examine how much model evaluation metrics (computed on ($\hat{y}$, $y$) pairs) change if we tweak the current problematic model or use another model to make predictions.

### Problem Statement: Unified Feature Platform

Even just a few years ago, even at early ML adopters like Pinterest, it was common for ML engineers to write their own feature code for offline iterations, while backend engineers translated it into Java/C++/Go for online serving (see Pinterest's ML infra [blogpost](https://medium.com/pinterest-engineering/a-decade-of-ai-platform-at-pinterest-4e3b37c0f758)). Even today, ML engineers often write feature pipelines for backfill, while backend or infra engineers take care of forward logging and realtime feature updates. These inconsistent implementations may lead to online–offline discrepancies and hinder iteration velocity.

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

{{< figure src="https://www.dropbox.com/scl/fi/b691em8ao42vn4vvbgtot/Screenshot-2025-11-29-at-4.56.09-PM.png?rlkey=xgo0hxjns2h2ubz1hsls799jx&st=3sfjgsnc&raw=1" caption="Robusta, Snap's feature engineering platform (source: [Snap eng blog](https://eng.snap.com/speed-up-feature-engineering))." width="1800">}}

Say we want to know for each `snap_id`, how many views it got in the last 6 hours (`snap_view_count_last_6h`). We should ask:
- **Realtime updates**: What happens when a *view event* happens?
- **Online path**: What happens when we *serve a request online*?
- **Offline path**: What happens when we *create training data offline*?

1. **Declarative feature spec**: First, we need to allow users (ML engineers) to define such a feature with online + offline paths
   - *Config*: User defines features by specifying some parameters 
      - Aggregation type: e.g., <span style="background-color: #FFC31B">`count`</span>, `sum`, `last_n`, `approx_quantile`
      - Keys to group by: e.g., <span style="background-color: #FFC31B">`DOCUMENT_ID`</span>, `USER_ID`, `HOUR_OF_DAY` --- can select a primary key
      - Window granularity: e.g., 5 min, 1h, <span style="background-color: #FFC31B">6h</span>, 30d, etc.
      - Optional filters: e.g., view duration, view day of week
   - *Execution*: The engine will compute the aggregations you specify via associate + communicative operations
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
   - *Data ingestion*: The event goes to a Kafka topic (e.g., `snap_views`), which gets periodically written to an Iceberg table (e.g., `snap_views_raw`) partitioned by date/hour
   - *Team-specific tables*: Each team can define a Spark table with custom schema, filtering, dedup, transformation, etc. 👉 each row is a clean event `(snap_id, event_time, …)`
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
   - *Streaming job*: Computes pre-aggregated blocks at 5-min intervals --- the goal is to ensure freshness
      - For each incoming row `(snap_id, event_time)`
         - Get its 5-min bucket --- in this case, `16:00–16:05`
         - Increment view count in bucket: `rep_block(snap_id, 16:00–16:05) += 1`
      - Write the new or updated block to
         - Iceberg table (e.g., `snap_view_blocks_5m`)
         - Online features store (e.g., [Aerospike](https://en.wikipedia.org/wiki/Aerospike_(database)))
   - *Batch job*: Computes pre-aggregated blocks at coarser intervals (e.g., 1h) --- the goal is to ensure completeness
      - Every hour/day, run job to 
         - Re-scan the raw event table `snap_views_raw`
         - Recompute block view counts in 
            - 5 min blocks to repair gaps or errors if needed
            - 1h blocks by summing 12 x 5 min blocks
      - Write new or updated blocks to Iceberg + Feature Store
         - Each row is `(primary_key, window_start, window_size, feature_id, representation_blob)`
         - The table is partitioned by `(primary_key_shard, window_start_bucket)` so that
            - Online reads can fetch blocks for given key
            - Offline jobs can scan by shard + time range
4. **Online path**: Fetch latest feature values for the current request
      - Say a request comes in at 16:10 for `snap_id = S456`
      - *Option 1*: Assemble from intermediate blocks upon request 
         - We need to fetch blocks to cover the last 6 hours, dating back from the request time: `[10:10, 16:10)`
         - Select non-overlapping blocks that best cover this range
            - 1h blocks: `11:00–12:00`, `12:00–13:00`, `13:00–14:00`, `14:00–15:00`, `15:00–16:00`
            - 5 min blocks: `10:10–10:15` to `10:55–11:00`, `16:00–16:05`, `16:05–16:10`
         - Combine counts in those blocks to get the final answer
         - When to use: For large or rare features we don't want to pre-materialize, we can sacrifice some latency for space
      - *Option 2*: Pre-assemble in the feature store
         - Periodically scan `snap_view_blocks_5m` to compute the full feature value for each `snap_id`
         - Write `(snap_id, snap_view_count_last_6h, watermark_ts)` to a KV store or a document index file
         - When to use: If we score the same `snap_id` many times, we should pre-assemble feature values to reduce latency, at the expensive of slight staleness --- for most ranking candidate features, we should do this if possible
      - *Forward logging*: Regardless of how we get the feature, we can log it with other features and model predictions
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
5. **Offline path**: If this is a new feature we've yet to forward log or if it has wrong serving values we want to fix, we can generate feature values that the model would've seen online (point-in-time correctness) and join them with labels for training
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

### Core Abstractions: Blocks vs. Features

I think many can tell Snap's [Robusta](https://eng.snap.com/speed-up-feature-engineering) has an unmistakable Google flavor (think [MapReduce](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/), [FlumeJava](https://research.google/pubs/flumejava-easy-efficient-data-parallel-pipelines/), [Dataflow](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/)) --- like the rest of their ML stacks (e.g., Kubeflow, TensorFlow). I haven't checked, but Snap's ML team must have been founded by Google engineers. This aligned block design is incredibly complex and clever, and perhaps unusual. In my last weeks at DoorDash, someone asked in the ML platform channel whether we could avoid loading last 30 days' data every day to recompute 30-day aggregations, seeing how today's data has a 29-day overlap with that of yesterday. People thought it was a good idea but couldn't be done. Snap's idea of pre-computing small aggregation blocks and assembling them into arbitrary windows is ingenious. 

By contrast, Uber's Feature Store [Palette](https://www.uber.com/blog/palette-meta-store-journey/) uses a more general, flexible design, with the core abstraction being features, not blocks. Features can be generated in many ways --- e.g., via a remote procedure call (e.g. calling an external ETA-service to get store ETA), a Flink SQL job that transforms Kafka streams into realtime features, or an ad-hoc job dumping features to S3. Batch-computed feature outputs are written first to the offline Feature Store (Hive), then synced to the online Feature Store (Cassandra); near-realtime features are ingested directly into the online store and later logged into the offline store to keep training and serving consistent. Maybe it's less effective at heavy aggregations, but better at handling "weird" or "random" features. 

## Real-Time Features

### Kafka and Flink 101

> [...] we will never know if or when we have seen all of our data, only that new data will arrive, old data
may be retracted, and the only way to make this problem tractable is via principled abstractions that allow the practitioner the choice of appropriate tradeoffs along the axes of interest: correctness, latency, and cost [...]. Since money is involved, correctness is paramount. --- [*The Data Flow Model (2015)*](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/)

We saw that streaming jobs generate near realtime features. To dig deeper, read Chapter 11 of [Designing Data-Intensive Applications](https://dataintensive.net/) and Google's seminal Dataflow and MapReduce papers. For a quick read, I like Samuel Flender's [blogpost](https://mlfrontiers.substack.com/p/feature-infrastructure-engineering) on feature infra, and Hello Interview's articles on [Kafka](https://www.hellointerview.com/learn/system-design/deep-dives/kafka) and [Flink](https://www.hellointerview.com/learn/system-design/deep-dives/flink). There's even a [children's book](https://www.gentlydownthe.stream/) on Kafka 👶.

A backend design question closely related to realtime features is [Design an Ad Click Aggregator](https://medium.com/@bugfreeai/designing-an-ad-click-aggregation-system-meta-senior-engineer-system-design-interview-guide-18db8a974c3b). It's considered pretty challenging in a backend interview, but MLE candidates can usually treat click counts as given and avoid going into rate limiting or click-abuse handling.

Batch processing assumes the input is bounded --- at the end of the hour/day, (we think) we know all the data from that hour/day, stored on some files. In reality, data arrives continuously, as small, self-contained, immutable records carrying information about past events, and is never complete. Streaming processing processes events as they happen (realtime) or shortly afterward (near realtime). 

A *producer* generates an event (e.g., a user clicks a post) and consumers process those events. Related events are grouped into *topics* (e.g., `user_post_clicks`). To ingest massive amounts of events with low latency, we need a distributed messaging system like [Kafka](https://www.linkedin.com/pulse/kafkas-origin-story-linkedin-tanvir-ahmed/): The producer sends a message with event data (e.g., JSON or binary) to a message queue, from which consumers can process events asynchronously. Below is a toy example of a click event message:

   ```json
     {
        "topic": "post_clicks",
        "key": "post_98765",
        "value": {
          "user_id": "user_12345",
          "post_id": "post_98765",
          "event_type": "click"
        },
        "timestamp": 1732945692123,
        "headers": {
          "schema_version": "1"
        }
     }
   ``` 

Consumers can subscribe to specific topics, such as `post_clicks`. 

   - **Horizontal scaling**: To allow multiple consumer groups to listen to each topic, each topic has multiple partitions (append-only "logs") 
      - `topic` provides a logical grouping of related messages
      - `key` determines the partition number within a topic --- e.g., via consistent hashing, `hash(key) % num_paritions`
      - `value` contains detailed information about the event
      - `timestamp` orders messages within each partition
      - `headers` stores metadata as key-value pairs
   - **Fault tolerance**: Kafka has extremely high availability through mechanisms such as replication and redelivery
      - *Broker*: a sever to which producers send messages, which then pushes them to message queues for consumers to read; one broker can handle multiple topics
      - *Leader-follower replication*: Each partition is replicated across multiple brokers, with one acting as the leader and the rest as followers. A message from the producer is only acknowledged as received if all (or >= threshold) replicas have received it. If one broker dies, others still have the data.
      - *Redelivery*: Once a consumer processes a message, it will send an acknowledgement to the broker, which then can delete the message from the queue. If an acknowledgement is not received after a given time, the message is consider lost and the broker will redeliver the message

Upon receiving a message, a consumer can write it to data storage ("sink") such as a database, cache, or search index. To create features (e.g., post clicks in the last 5 min), we need to aggregate events from Kafka streams over some time window --- see the figure below for common window definitions. We can write our own processing logic, but it's more convenient to use a stream processing engine like [Flink](https://en.wikipedia.org/wiki/Apache_Flink).

{{< figure src="https://www.dropbox.com/scl/fi/cpsspd61cy0kzjfj1yqt1/Screenshot-2025-11-30-at-1.42.20-PM.png?rlkey=xyrvazpo7ongbdqscms4lafof&st=dk6w0zv0&raw=1" caption="The Dataflow paper defines 3 types of windows. There are more now." width="1800">}}

Composing Flink operators to process data is similar to writing SQL queries (there's plenty of database analogies in the DDIA book as well) --- below is a toy example for computing rolling 5-min click counts:

```java
DataStream<ClickEvent> clicks = // input stream

clicks
  .keyBy(event -> event.getPostId())   // group by post
  .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))
  .reduce(
      (a, b) -> new ClickEvent(
          a.getPostId(),
          a.getCount() + b.getCount()
      )
  );
```

Flink can write results to an online Feature Store (e.g., Cassandra), which is fetched for serving. Through forward logging or backfill (if we can reconstruct values), these features can be used for training.

<!-- > Users specify a *map* function that processes a key/value pair to generate a set of intermediate key/value pairs, and a *reduce* function that merges all intermediate values associated with the same intermediate key. Many real world tasks are expressible in this model [...]. --- [*MapReduce (2004)*](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) -->

### Case Study: Realtime Aggregation Features

Apps like Xiaohongshu and TikTok feel incredibly sticky not because they retrain models in real time, but because they update engagement features in near real time. The model has already learned that past clicks predict future clicks --- what's amazing is how a new click on a post (thereby its author, topic, etc.) is reflected almost immediately, often within the same session, to help uprank similar content.

{{< figure src="https://www.dropbox.com/scl/fi/gjlt3lnd96kuxauf267ca/Screenshot-2025-11-30-at-4.49.43-PM.png?rlkey=9mx7ysy3tknetizfwfq2gh6s7&st=qgwq8ckr&raw=1" caption="Pinterest's realtime user signal serving (from 2019 and like outdated)." width="1800">}}

This 2019 [blogpost](https://medium.com/pinterest-engineering/real-time-user-signal-serving-for-feature-engineering-ead9a01e5b) from Pinterest is likely dated, but it has some useful lessons. For instance, raw engagement events can be lightweight --- we don't need to record everything but only need to push information such as `userId`, `pinId`, and `actionType` to the Kafka message queue. Consumers can perform feature hydration asynchronously by looking up user and pin features from a Feature Store and write hydrated events to a time sequence event store. 

Another insight is that the aggregator performs [incremental computing](https://en.wikipedia.org/wiki/Incremental_computing) instead of reprocessing a user's entire history on each update. The blogpost is hand-wavy about the details, but a common approach is to build a dependency graph among data elements --- e.g., `clicks_last_5min` and `clicks_last_1hr` both depend on click events. When a new event arrives, the system only needs to update impacted data elements (e.g., increment the count by 1).

### Case Study: Realtime Sequence Features

If you flip through KDD and RecSys papers, you'll see if 2025 is the year of Generative Recommendation (see {{< backlink "generative_recommendation" "my post" >}}), then 2023 and 2024 were all about sequence modeling (see {{< backlink "seq_user_modeling" "my other post" >}}). Many companies realized lots of gains can come from modeling user history as a time-ordered sequence, rather than a bag of engaged items. 

This Pinterest [blogpost](https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9) talks about how to build realtime sequences consisting of a user's last 100 actions (this [post](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8) talks about the model). 

{{< figure src="https://www.dropbox.com/scl/fi/rc8pyrnudj5fkqs4kasds/Screenshot-2025-11-30-at-5.44.20-PM.png?rlkey=nfxjyti37l044mbu3i17tvsb7&st=57xfc8sd&raw=1" caption="Pinterest's online + offline paths for realtime sequence features." width="1800">}}

For aggregation features (e.g., `count`, `sum`, `avg`), it's not the end of the world if events are slightly out of order --- as long as they fall within the correct window, we'll get the same result for that window. For sequences, however, we must preserve the order. This blogpost, like most engineering blogs, doesn't explain how. Here are some ideas:

- **Maintain an append-only, time-ordered per-user event log**: If we distribute a user's events across multiple partitions, it becomes more complicated to reconstruct their event sequence. Instead, we can keep a user-specific log to ensure $O(1)$ in-order inserts.
- **Use a small in-memory buffer to handle out-of-order events**: Some events arrive slightly late. We can hold them briefly in a smaller buffer, insert them in the correct event time order, and periodically flush the bugger into the user log in sorted order. To minimize delay, we keep the buffer short (e.g., ~60 seconds).
- **If sharding is necessary, shard by `userId`**: If logs are stored on multiple paritions, we should apply consistent hashing on `userId` (e.g., `hash(userId) % num_paritions`). This ensures that all of a user's events stay on the same partition.

Another interesting point is that we should be able to create new attribute sequences offline. For instance, suppose we have a time-ordered sequence with content embedding A --- `[a_0, a_1, ..., a_99]` --- but later develop a better content embedding B. We should be able to backfill a new sequence using the logged timestamps and new embedding values at each timestamp, `[b_0, b_1, ..., b_99]`. This allows us iterate on models offline with richer attributes.

## Distributed Training

Ranking models are tiny compared to LLMs. Recently, Meta published a blogpost on [Generative Ads Recommendation Model](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) (GEM), a foundation model for ads ranking that rivals LLM scale. In North America, however, I don't think any companies besides Meta (and perhaps Google) can or want/need to train such large ranking models. So for training, distributed model parallel (DMP) is often unnecessary (exception: in [this paper](https://arxiv.org/html/2508.05700v1), Pinterest ads ranking uses embedding tables so large that they need to be sharded across multiple CPUs). 

However, engagement data for training ranking models can be massive (e.g., billions of records each day, even after downsampling). If the model reads one batch at a time, training will take forever. Distributed data parallel (DDP) is a must, where multiples workers read different batches and aggregate gradients to update model parameters.

### Old Paradigm: Parameter Servers

How do we aggregate gradients from different workers to update a single model? A classic solution is to use a parameter server to store authoritative model parameters. After processing a batch, each worker pushes its gradients to the server, the server applies the update, and the worker pulls the latest parameters before the next batch. 

{{< figure src="https://www.dropbox.com/scl/fi/of5w5mgdrakp7nd72msd1/Screenshot-2025-11-30-at-9.54.11-PM.png?rlkey=rsa22b9wpe3p1tj1u3q8hbuxz&st=dgv01ij4&raw=1" caption="A gross simplification of the parameter server pattern." width="1800">}}

However, this design is dangerous --- if some workers are slow and push gradients computed on old parameters (like "lap cars" in a race), they can push parameters in the wrong direction, which hinders learning. Modern distributed training rarely uses this architecture.

### New Standard: `all-reduce`

Modern distributed training systems commonly use *collective communication* between peer workers. A naive approach is all-to-all communication with $O(N^2)$ time complexity ($N$: number of workers).

{{< figure src="https://www.dropbox.com/scl/fi/jk8zbxrml9f3d2a3ekzei/Screenshot-2025-11-30-at-11.11.53-PM.png?rlkey=j313fk9b3r4u3llpsut5uqxk4&st=bfy5vjws&raw=1" caption="Collective communication between workers via the allreduce operation." width="1800">}}

When each worker finishes computing gradients, it sends them to all other workers in the group. Upon receiving gradients from all other workers, each worker aggregates the gradients (`reduce`) and sends the aggregated gradients back to all other workers (`broadcast`). The combination of `reduce` and `broadcast` is called `allreduce`. This design makes sure all workers are on the same page.

However, you might ask: Is it really necessary for each worker to send its aggregated gradients back to all other workers? Doesn't each worker already possess the full gradients after the `reduce`? Fair question. In reality, we can let each worker only communicate with its two neighboring workers so it would only hold three gradients (its own and those of its two neighbors). These partial gradients are then broadcast and aggregated across all workers. This algorithm is called `ring-allreduce` and has a $O(2 \cdot (N-1)$ time complexity.

In their [blogpost](https://eng.snap.com/training-models-with-tpus), Snap explains how they distribute training across TPU cores: Each core processes a different slice of input data and maintains a local copy of the model. Gradients from all cores are synchronized via `all-reduce` that we just discussed, and large embedding tables are automatically sharded across the TPU cluster. When comparing performance, they report that 4× A100 GPUs offer lower throughput than a TPU v3-32 pod --- but the GPU setup did not use embedding-lookup optimizations, while the TPU did.

For distributed training on GPUs, check out the PyTorch author's blogpost, [*How to train a model on 10k H100 GPUs?*](https://soumith.ch/blog/2024-10-02-training-10k-scale.md.html?utm_source=chatgpt.com). Key lessons:
- **Maximize parallelism & memory efficiency**: Distribute not just over batches (data parallel), but also across model layers (layer + pipeline parallelism); use memory-saving tricks (e.g., checkpointing, sharding weights) to scale model and batch sizes. 
- **Overlap communication and computation**: Start gradient communications (e.g., `all-reduce`) as soon as parts of the backward pass finish, so network transfer happens in parallel with ongoing compute rather than blocking it. 
- **Leverage network and hardware topology**: Use optimal collective-communication libraries (e.g., NCCL, RDMA) to exploit interconnect topology, minimize cross-node data movement, and reduce latency when synchronizing across many GPUs. 
- **Design for fault tolerance and checkpointing**: At the scale of thousands of GPUs, hardware failures, node dropouts, and even silent memory bit-flips become nontrivial. A robust system shards model state, checkpoints frequently (preferably asynchronously), and monitors for stragglers or unresponsive nodes.

## GPU Serving

### Debug Throughput by Profiling

As a wise person once said (I've forgotten who 😂), the key to efficient training and serving is keeping your machines busy, as measured by Model FLOPs Utilization (MFU). Most operations in deep learning models (e.g., projections, attention) boil down to General Matrix Multiplication (GEMM), which GPUs excel at. I learned from tenured colleagues that GPU serving was a game changer for Pinterest: Within a year of enabling it, ranking models grew larger and more expressive, delivering gain after gain. This [blogpost](https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d) recounts on how it all started. 

{{< figure src="https://www.dropbox.com/scl/fi/pfdctl0fahh24as9gi3ch/Screenshot-2025-12-01-at-12.41.42-AM.png?rlkey=rl457xfnoiwi71l5cdp0y6vvk&st=ukb405dv&raw=1" caption="Many small kernels inimitably led to lower-than-expected throughput." width="1800">}}

In the beginning, serving throughput didn't increase as much as expected. The infra team profiled the model during inference and discovered that many small kernels dominated the timeline. 

### Optimize by Eliminating Small Kernels

The infra team made improvements via several optimizations:
- **Fused embedding lookup**: Embedding lookups were slow --- for each raw ID, we must first look up its index in the embedding table, and then retrieve the embedding vector. A model may need to look up hundreds or even thousands of IDs (think sequence models). Using [cuCollections](https://github.com/NVIDIA/cuCollections), we could fuse all lookups into one.
- **One-time feature copy**: Copying each feature tensor from CPU to GPU individually incurred a large overhead. Instead, we can pre-allocate a contiguous host memory buffer, copy all feature tensors into it, and copy the buffer into GPU at once. On the GPU side, tensors were reconstructed to point into the buffer using offsets, reducing hundreds of `cudaMemcpy()` calls per request into just one. This cut data copy latency from ~10 ms down to sub-1 ms.
- **Using CUDA graph**: A [GPU graph](https://developer.nvidia.com/blog/cuda-graphs/) captures the full inference process as a static graph rather than a sequence of individual kernel launches. GPU executes the graph as one unit, eliminating much of the kernel launch overhead and idle gaps between ops. The trade-off: Tensors must be padded to fixed sizes, making tensor shapes less flexible and increasing memory footprint.
- **Using larger batches**: By merging multiple user requests into larger batches, GPU can run batch matrix operations efficiently. While we wait longer for requests, per-request latency drops. Conceptually, request batching is similar to what HuggingFace calls ["continuous batching"](https://huggingface.co/blog/continuous_batching), which aims to improve GPU utilization for LLM inference also by merging many small inference "requests". The LLM case is only more complex since each "request" (generating a sequence) requires multiple steps, whereas each ranking request can be done in one forward pass.

### Triton: Faster, Custom Kernels

We've seen some CUDA kernels, such as cuCollections that can be used to fuse embedding lookups. CUDA graph runs whatever kernels you already have, but just more efficiently. OpenAI's [Triton](https://openai.com/index/triton/) is a kernel compiler that allows you to write custom GPU kernels, which can beat PyTorch kernels and NVIDIA's CUDA kernels. You can even fuse ops from different libraries and across autograd boundaries.

For example, to optimize serving for lifelong sequence models, Pinterest ([blogpost](https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d3
)) built a custom Triton kernel to fuse QKV projection, attention, layer norm, and feed-forward into a single op. This eliminates the need to write intermediate tensors to GPU HBM (off-chip memory, large but slow), allowing all weights and data to stay in GPU SRAM (on-chip memory, tiny but extremely fast). Throughput improved by 6.6× as the fused Triton kernel removes kernel launch overhead and repeated data transfers between HBM and SRAM.

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
11. [The Dataflow Model](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/) 👉 using windows to tame unbounded data
12. [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) 👉 the original MapReduce paper
13. [Point-in-Time Correctness in Real-Time Machine Learning](https://towardsdatascience.com/point-in-time-correctness-in-real-time-machine-learning-32770f322fb1/) 👉 data leakage prevention for backfill
14. [Speed Up Feature Engineering for Recommendation Systems](https://eng.snap.com/speed-up-feature-engineering) 👉 Robusta, Snap's feature pipeline that unified online + offline feature and data generation
15. [Building a Spark-Powered Platform for ML Data Needs at Snap](https://eng.snap.com/prism) 👉 Prism, Snap's control-plane + workflow orchestration for Spark/ML data jobs
16. [Michelangelo Palette: A Feature Engineering Platform at Uber](https://www.infoq.com/presentations/michelangelo-palette-uber/#:~:text=Michelangelo%20Palette%20is%20essentially%20a,models%20and%20why%20is%20that%3F) 👉 Palette, Uber's feature pipeline and feature store
17. [Zipline --- Airbnb's ML Data Management Framework](https://conferences.oreilly.com/strata/strata-ny-2018/cdn.oreillystatic.com/en/assets/1/event/278/Zipline_%20Airbnb_s%20data%20management%20platform%20for%20machine%20learning%20Presentation.pdf) 👉 Zipline, Airbnb's feature pipeline and feature store
18. [How Pinterest Accelerates ML Feature Iterations via Effective Backfill](https://medium.com/pinterest-engineering/how-pinterest-accelerates-ml-feature-iterations-via-effective-backfill-d67ea125519c) 👉 Pinterest introduced two-stage backfill as well as training time join to speed up backfill 

### Real-Time Features
19. [Designing Data-Intensive Applications](https://dataintensive.net/) 👉 before you read anything else, read Chapter 11: Stream Processing
20. [Feature Infrastructure Engineering: A Comprehensive Guide](https://mlfrontiers.substack.com/p/feature-infrastructure-engineering) 👉 Samuel Flender's new blogpost on real-time signals
21. [Real-Time User Signal Serving for Feature Engineering](https://medium.com/pinterest-engineering/real-time-user-signal-serving-for-feature-engineering-ead9a01e5b) 👉 an old Pinterest blogpost on realtime aggregation features
22. Pinterest's Realtime User Sequences on Organic Homefeed 👉 blogposts written by [ML engineers](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8) and [ML infra engineers](https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9)

### Distributed Training
23. [Training Large-Scale Recommendation Models with TPUs](https://eng.snap.com/training-models-with-tpus) 👉 Snap has been using Google's TPUs since 2022
24. [How to Train a Model on 10k H100 GPUs?](https://soumith.ch/blog/2024-10-02-training-10k-scale.md.html) 👉 PyTorch author's short blogpost
25. [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 👉 train LLMs on GPU clusters

### GPU Serving
26. [Applying GPU to Snap](https://eng.snap.com/applying_gpu_to_snap) 👉 Snap's switch from CPU to GPU serving
27. [GPU-Accelerated ML Inference at Pinterest](https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d) 👉 Pinterest did the same a year later
28. [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 👉 CUDA graphs are often used in GPU serving
29. [Continuous Batching](https://huggingface.co/blog/continuous_batching) 👉 continuous batching speeds up language model inference, which may apply to recommender systems
30. [Introducing Triton: Open-Source GPU Programming for Neural Networks](https://openai.com/index/triton/) 👉 OpenAI's Triton kernels
