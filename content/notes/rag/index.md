---
title: "Fundamentals of Retrieval Augmented Generation (RAG)"
date: 2025-10-25
categories: ["llm", "search", "rag"]
math: true
toc: true
---

If you ask 100 ML engineers about their career goals, 90 of them will say they want to work on LLMs someday. If you ask which part of LLMs they want to work on, probably 80 out of those 90 will say pretraining, post-training, or whatever is perceived as "core modeling work." Quite likely they never could. The remaining 10 may or may not land jobs on <span style="background-color: #D9CEFF">applied research engineering</span> teams at `{OpenAI, Anthropic, xAI, GDM}`, Perplexity, Glean, Anysphere, or the likes, building AI products (e.g., chatbots, web/enterprise search, etc.) that people commonly use. [Retrieval-Augmented Generation](https://python.langchain.com/docs/concepts/rag/) (RAG) and, later, [AI agents](https://lilianweng.github.io/posts/2023-06-23-agent/) are top technologies behind popular AI applications.

In this post, I'll summarize RAG fundamentals from LangChain's RAG from Scratch ([YouTube](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x), tutorials [1](https://python.langchain.com/docs/tutorials/rag/) & [2](https://python.langchain.com/docs/tutorials/qa_chat_history/), [repo](https://github.com/langchain-ai/rag-from-scratch)), DeepLearning.AI's [RAG course](https://www.coursera.org/learn/retrieval-augmented-generation-rag), and some classic or recent RAG + Agentic Search papers.

# The Structure of a RAG

*Why RAG?* Out of the box, an LLM isn't well-suited to generate information it didn't have access to during training---such as recent events, a company's proprietary knowledge base, personal data, and so on. RAG retrieves missing information the LLM needs in order to generate accurate, up-to-date, and context-aware responses.

{{< figure src="https://www.dropbox.com/scl/fi/eruy5k5nbt3qsi41647pb/Screenshot-2025-10-24-at-6.25.52-PM.png?rlkey=ur5ljzfju8lmd4ds4jdec0wnm&st=0dg93uma&raw=1" caption="A basic RAG system (source: [DeepLearning.AI RAG](https://learn.deeplearning.ai/courses/retrieval-augmented-generation/information) course)." width="1800">}}

- **Retriever**: The query is first routed to the retriever, which retrieves relevant documents from the knowledge base and ranks them relative to the prompt to select the top-k.
- **Generation**: The retrieved documents are used to augment the prompt, which is then sent to an LLM to generate a response.

The system could be more complex---for instance, we can use a routing LLM to decide whether to skip the retriever and generate a direct LLM response. The first paper from the much hyped Meta Superintelligence Lab is a RAG paper called *REFRAG: Rethinking RAG based Decoding* ([Lin et al., 2025](https://arxiv.org/abs/2509.01092)), which aims to make RAG cheaper and faster. This shows how central RAG is to LLM applications.

# Before a Search Is Issued

In an {{< backlink "ltr" "old post">}}, I wrote about how traditional search engines work. Before we could issue any search, we need to index all documents so later they could be easily found. When a query comes in, we might need to transform it somehow (e.g., autocompletion, spell checking, intent classification, rewriting) to optimize search results. It turns out that in LLM-based search systems such as RAG, indexing and query analysis are still instrumental to search engine performance.  

## Indexing

From what kind of databases are we retrieving information? In case of natural language documents, we usually store them in two formats:
- **[Inverted index](https://www.geeksforgeeks.org/dbms/inverted-index/)** (for keyword-based search): Analyze documents into tokens (e.g., ElasticSearch [analyzer](https://www.elastic.co/docs/manage-data/data-store/text-analysis/anatomy-of-an-analyzer)) 👉 for each token, store a list of documents in which it appears (like the end of textbooks)
- **Vector database** (for embedding-based retrieval): Chunk documents into meaningful pieces 👉 for each chunk, use an LLM to create a dense embedding and we store its embedding

For structured data, we can store them in relational (e.g., Snowflake, MySQL) or graph (e.g., Neo4j) databases and use GQL or SQL to query. 

For traditional search engine indexing, I highly recommend the [Relevant Search](https://www.oreilly.com/library/view/relevant-search/9781617292774/) book. Amazing how nothing beats its comprehensiveness, when it still uses Python 2.7. Below we'll focus on more "fashionable" techniques designed for vector databases.

### Chunk Raw Documents
If a document is a book, for instance, it's too coarse to represent the entire book as a single embedding. 
- **Lack granularity**: It may be hard to retrieve specific topics, chapters, or pages from the book.
- **Context window explosion**: Even if we successful deem the whole book as relevant and successfully retrieve it, it will blow up the LLM context window when we plug it into the prompt.

We can chunk documents into smaller pieces. But how? How about we split the book into words? Good luck knowing what `dog` or `the` means without the context. Chunking is critical for a RAG system's performance. Below are common techniques:

- **Fixed-size chunking**: We can split the document into chunks containing the same number of tokens (e.g., 250). It's best to have token overlaps between adjacent chunks so the context flows.
- **Recursive character splitting**: We can split the document on one (".") or several special characters (e.g., all sorts of end-of-sentence symbols). Each chunk may end more naturally. 
- **Semantic chunking**: We can start a chunk with a sentence --- if the next sentence has high cosine similarity with the current sentence, include it in the chunk; otherwise, start a new chunk.
- **LLM-based chunking**: We can prompt an LLM to create chunks from a document. Nowaways, this method is becoming increasingly more cost-effective and widely used. 
- **Context-aware chunking**: This is orthogonal to above methods --- regardless of how we created a chunk, we can use an LLM to add context to it --- e.g., whether a chunk from a dissertation pertains to a title, an acknowledgement, a method, or something else.

### Store Embeddings in Vector Databases

We can use an embedding model (pretrained or in-house models trained using contrastive objectives) to generate embeddings for chunks we created and store them in a vector database. If storage cost is a concern, we can apply quantization to condense raw embeddings at the element level (e.g., map each float element into an int8 or even an int4 representation) or the vector level (e.g., map each dense vector into a binary vector via vector or product quantization).

In some cases, quantization can speed up top-k retrieval---e.g., if queries and documents are binary vectors, we can use Hamming distance instead of inner product to select the top k, reducing the complexity of each comparison from $O(d)$ to $O(1)$, where $d$ is the vector dimension. (I first read about this trick in an [Alibaba paper](https://arxiv.org/pdf/2108.04468), but I remember it was based on a LinkedIn paper whose title I forgot.)

See my EBR {{< backlink "ebr" "blogpost" >}} for more details on storage optimization.

## Query Analysis

Raw inputs are typically natural language queries from the user. Depending on where we query data from, we may need to **construct** natural language queries into SQL (for relational databases) or GQL (for graph databases) for structured databases, or **rewrite** the query to improve natural language documents that we could return. 

### Query Rewriting
I remember back in 2022 as a new grad, my onboarding project at DoorDash was on query expansion --- at that time, Starbucks wasn't on DoorDash so users searching for `starbucks` would see null results. However, if they were sleepy and needed coffee, wouldn't they want to see other coffee shops? Mapping a query to a type of stores to expand search results is a form of query expansion. LangChain has an awesome [post](https://python.langchain.com/docs/concepts/retrieval/) that summarizes many forms of query rewriting:
- **Query clarification**: Rephrase ambiguous, poorly worded, or otherwise confusing queries for clarity
  - E.g., a patient went on a tangent to describe how they hurt themself on a skiing trip starting from booking an Airbnb and getting to the resort; in the end, they asked whether they should worry about the symptoms 👉 you could just extract the symptoms and search them in a medical database
- **Semantic understanding**: Identify the query intent (e.g., shopping or finding information), so we could go beyond literal matching
- **Query expansion**: Generate related terms or concepts to broaden the search scope, such as from `starbucks` to `coffee shops` or `breakfast` --- useful when Recall is very low
- **Complex query handling**: Break down multi-part queries into simpler sub-queries --- examples techniques including
  - **Multi-query**: Rewrite a compound query into several simple sub-queries 👉 retrieve documents for each sub-query
  - **Decomposition**: Decompose a complex problem into subproblems (e.g., the user asks the LLM to debug an issue and there are 5 things to investigate) 👉 retrieve documents to help resolve each subproblem; combine the solution to the previous problem and retrieved information for the current problem to resolve the current problem
  - **Step-back** ([Zheng et al., 2024](https://arxiv.org/abs/2310.06117)): Ask the LLM to derive high-level concepts and first principles from detailed examples and use those concepts and principles to guide reasoning
  - **HyDE** ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)): Convert a hard-to-handle query into hypothetical documents and use hypothetical document embeddings (HyDE) to retrieve real documents

Which method to use depends on how the system underperforms (e.g., low recall? low precision? too much detail? too little detail?...). We can control the method by tweaking the system prompt.
 
### Query Construction
If we need to query from a relational or a graph database, we need to translate a natural language query into a structured query language. There are many [text-to-SQL](https://python.langchain.com/docs/tutorials/sql_qa/) and [text-to-Cypher](https://python.langchain.com/docs/tutorials/graph/) (Cypher is a graph query language) models at your disposal. You can also create metadata filters from the query (e.g., title, author, year, region, language).

# Retriever

## Hybrid Search

{{< figure src="https://www.dropbox.com/scl/fi/m9ts8k5sgebb7cacmvaec/Screenshot-2025-10-24-at-7.20.39-PM.png?rlkey=4mj913dxxuvn09jkufp8gcp4e&st=thy25rnj&raw=1" caption="An illustration of hybrid search (source: [DeepLearning.AI RAG](https://learn.deeplearning.ai/courses/retrieval-augmented-generation/information) course)." width="1800">}}

### Lexical Retrieval

Traditional search engines "know" a document is relevant if many query tokens appear in it. The two most famous algorithms for computing query-document lexical similarity are [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and [BM25](https://en.wikipedia.org/wiki/Okapi_BM25). 

Below is a toy TF-IDF retriever implementation based on this formula:

$${\displaystyle (1+\log f_{t,d})\cdot \log {\frac {N}{n_{t}}}}.$$

```python3
from collections import defaultdict
import heapq
import math

class TFIDFRetriver:

    def __init__(self, docs):
        """process doc corpus and precompute tf-idf of each doc"""

        self.docs = docs  # remember original docs
        self.N = len(docs)  # number of docs

        # document frequency: in how many DISTINCT docs does term appear?
        self.df = defaultdict(int)  # {term : # of unique docs}
        for doc in docs:
            seen = set()  # track if term has already been counted in this doc
            for term in doc.lower().split():
                if term not in seen:
                    self.df[term] += 1 
                    seen.add(term)

        # inverse document frequency: log(N / n_t)
        # n_t is df[term]
        self.idf = {term: math.log(self.N / df_val) for term, df_val in self.df.items()}

        # {term : index}
        self.stoi = {term: i for i, term in enumerate(self.idf.keys())}
        self.vocab_size = len(self.stoi)

        # precompute each document's tf-idf vector using (1 + log f_{t,d}) * idf
        self.doc_tfidf = {}
        for i, doc in enumerate(self.docs):
            vec = [0.0] * self.vocab_size

            # raw term frequency in this doc
            tf = defaultdict(int)
            for term in doc.lower().split():
                tf[term] += 1

            # fill tf-idf weights
            for term, count in tf.items():
                if term not in self.stoi:
                    continue
                idx = self.stoi[term]

                # (1 + log f_{t,d})
                tf_weight = 1.0 + math.log(count)

                vec[idx] = tf_weight * self.idf[term]

            self.doc_tfidf[i] = vec

    def _query_vector(self, query):
        """vectorize query into tf-idf vector with same formula"""

        vec = [0.0] * self.vocab_size

        # raw term frequency in query
        tf = defaultdict(int)
        for term in query.lower().split():
            tf[term] += 1

        for term, count in tf.items():
            if term not in self.stoi:
                continue
            idx = self.stoi[term]

            # (1 + log f_{t,d}) where d is "the query bag"
            tf_weight = 1.0 + math.log(count)

            vec[idx] = tf_weight * self.idf[term]

        return vec

    def _dot(self, a, b):
        """compute dot product between 2 vectors"""
        total = 0.0
        for x, y in zip(a, b):
            total += x * y
        return total

    def return_doc(self, query, topk=2):
        """return top-k documents with highest tf-idf scores"""
        query_vec = self._query_vector(query)

        heap = []  # min-heap of (score, doc_idx)
        for doc_idx, doc_vec in self.doc_tfidf.items():
            score = self._dot(query_vec, doc_vec)
            heapq.heappush(heap, (score, doc_idx))
            if len(heap) > topk:
                heapq.heappop(heap)

        # sort from best to worst
        heap.sort(reverse=True)

        results = []
        for score, doc_idx in heap:
            results.append((score, self.docs[doc_idx]))
        return results

# test case
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the mouse",
    "dogs and cats can be friends",
]
query = "cat sat on mat"

tfidf_retriver = TFIDFRetriver(docs)
results = tfidf_retriver.return_doc(query, topk=2)
print(results)
# printed results: [(3.3631710974274096, 'the cat sat on the mat'), (0.9609060278364028, 'the dog sat on the log')]
```

TF-IDF suffers from a lack of document length normalization (e.g., longer documents tend to include the same query token more times) and term frequency saturation (e.g., a document with 20 "pizza" matches is seen as 2x as relevant as one with 10 "pizza" matches). A convoluted algorithm, BM25, was created to fix these issues:

$${\displaystyle {\text{score}}(D,Q)=\sum\_{i=1}^{n}{\text{IDF}}(q_{i})\cdot {\frac {f(q_{i},D)\cdot (k_{1}+1)}{f(q_{i},D)+k_{1}\cdot \left(1-b+b\cdot {\frac {|D|}{\text{avgdl}}}\right)}}}.$$

The issue with lexical retrieval is, if I search `Taylor Swift husband`, I may not see Travis Kelce 🌲 results, since there are no term matches. Embedding-based retrieval comes to our rescue.

### Embedding-Based Retrieval
We could use a pretrained LLM to embed query and documents, or we could train our own query and document embeddings. Below are common architectures for learning query and document embeddings:

{{< figure src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/Bi_vs_Cross-Encoder.png" caption="Bi-Encoder vs. Cross-Encoder (source: Sentence Transformers [doc](https://sbert.net/examples/cross_encoder/applications/README.html)." width="1800">}}

- **Bi-encoder**: Embed query and doc separately using two identical towers 👉 compute cosine similarity between outputs.
- **Cross-encoder**: Concatenate query and document and embed the concatenated input 👉 feed the output to a binary classification head to predict a `[0, 1]` relevance score.
- **ColBERT**: Bi-encoder doesn't have any query-document interactions whereas cross-encoder performs an early fusion from the start; as a result, bi-encoder is fast but has low quality while cross-encoder has good quality but is slow 👉 to enjoy the best of both worlds, [ColBERT](https://arxiv.org/abs/2004.12832) performs a late fusion via MaxSum: for each query token, compute cosine similarity with each document, only keep the maximum score, and sum the scores over all query tokens to get the final relevance score.

{{< figure src="https://www.dropbox.com/scl/fi/w5ydcd7zfm4d0tl0s4ryu/Screenshot-2025-10-25-at-5.02.19-PM.png?rlkey=uig147bizg5wvb6bz5aukqq2u&st=vk9p8zx7&raw=1" caption="Architecture of ColBERT with late query-document fusion." width="1800">}}


Due to its inefficiency, rarely would we use the cross-encoder for retrieval. Moreover, since cross-encoders concatenate queries and documents before embedding, we can't precompute document embeddings offline, which is a hard requirement for low latency. Bi-encoders and ColBERT are therefore good choices for retrieval.

The brute-force way is, for a query, compute its similarity (e.g., inner product, cosine) with each document and return top k. This approach scales poorly with a large document corpus. In practice, people usually use approximate nearest neighbor (ANN) search to speed up search. To learn more, you can read the [ANN part](https://www.yuan-meng.com/posts/ebr/#approximate-retrieval-algorithms) in my blogpost, and a much more comprehensive summary in Lilian Weng's [blogpost](https://lilianweng.github.io/posts/2023-06-23-agent/#maximum-inner-product-search-mips). TL;DR is we need to reduce search space by:
- **Locality sensitive hash**: Hash each vector into multiple buckets 👉 only search in buckets where the query is assigned to
- **Graph traversal** (e.g., HNSW): Performs random walks on the document graph, hoping to minimize query-document distances
- **Clustering** (e.g., FAISS): Cluster documents based on their embeddings and only search in clusters where the query is in

### Metadata Filtering

We can further filter retrieved results based on metadata, such as title, author, creation data, region, etc.. This is not a true retrieval method, but an approach to purge unwanted or irrelevant retrieved results.

## Result Fusion
We can combine results retrieved from different sources using methods such as the [reciprocal rank fusion (RRF)](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/).

There are many knobs we can tune when combing different resources.
- **The "k" parameter in RRF**: RRF rewards documents highly ranked in each list---$\frac{1}{k + \mathrm{rank_1}} + \frac{1}{k + \mathrm{rank_2}} + \frac{1}{k + \ldots + \mathrm{rank_n}}$, where $\mathrm{rank_i}$ is a document's ranking in the $i$th list. The greater the k, the less a single high ranking matters. RRF cares about rankings, not scores.
- **Weight of keyword vs. semantic search**: Based on your understanding of the "ideal" system, you can make keyword matches more important than EBR, or the reverse. You can also tune the weights based on downstream evaluation results.

## Reranking

After selecting documents with highest combined scores, we need to rerank them by relevance and further cap the result length. At this stage, we can use the more expensive but performance cross-encoder.

## Retriever Evaluation

Results from the retriever can be evaluated end to end (e.g., whether the final generation is "good", whatever that means) or on their own. We need the latter for debugging and logging. When evaluating the retriever alone, please read the [evaluation section](https://www.yuan-meng.com/posts/ltr/#what-is-the-right-order) in my LTR post:
- **Rank-unaware metrics**: Recall@k, Precision@k
- **Rank-aware metrics**: nDCG@k (how much top k differs from "ideal ranking"), MRR (how early the first relevant result appears), MAP@k (how much relevant results are concentrated at the top)

# Generation
In traditional search, top k documents are returned to the user on a search result page. In RAG, however, they are plugged into a prompt to an LLM and it's the LLM that returns the user-facing response. 


## Prompt Template
A prompt typically consists of the following components:

- **System prompt**: Instructions on how an LLM should behave
  - **Role**: It controls the desired tone and personality of the LLM as well as what procedures it should follow
    - For RAG, we can ask the LLM to (1) only use retrieved documents to answer, (2) judge whether a document is relevant, and (3) cite sources in responses
  - **Be careful about the token usage!** System prompts are added to every prompt --- think twice before using in-context learning (i.e., providing one or few examples); in-context learning can even be detrimental to reasoning models
- **Conversation history**: Multi-turn conversations require the LLM need to remember previous conversations (i.e., past user messages and assistant responses); we can trim older messages no longer needed, a practice called "context pruning"
- **Retrieved information**: The content of retrieved documents (clean up or summarize, if need be) and their sources and metadata
- **User prompt**: The user's query with the question they want to ask

## Pick an LLM

Which LLM to use depends on your task. The LLM Arena [leaderboard](https://lmarena.ai/leaderboard) ranks models by performance in each domain (e.g., coding, writing, multimodal). When choosing an LLM, compare these key specs:
- **Context window**: How much text the model can process at once. Bigger is better for long documents or multi-turn conversations.
- **Training cutoff**: The most recent date of data the model was trained on. Newer is better for up-to-date knowledge.
- **Time to first token** (TTFT): Delay before the model starts responding. Lower is better for responsiveness.
- **Tokens per second** (TPS): How fast the model generates text. Higher is better for throughput.

## Decoding Strategies
Whichever LLM you choose, you can usually adjust its decoding strategy to control how it generates responses ---
- **Greedy decoding**: Always pick the most likely next token. Best for deterministic tasks (e.g., math, code), but gets repetitive or dull.
- **Top-k sampling**: Sample from top k tokens. Good for creative writing or brainstorming where you want controlled randomness.
- **Top-p (nucleus) sampling**: Sample from the smallest set of tokens whose cumulative probability exceeds p. Smoother and more adaptive than top-k; ideal for natural dialogue or storytelling.

You can also tweak token logits to change their probabilities:
- **Repetition penalty**: Reduces the probability of tokens already generated. Prevents loops and redundancy in long outputs.
- **Logit bias**: Manually adjust the probability of specific tokens. Useful for steering style or enforcing constraints (e.g., suppressing profanity, forcing certain keywords).
- **Temperature**: Lower = more deterministic, higher = more diverse. Use low temperature for precision, high for creativity.

## Generation Evaluation
At a high level, the responsibility of a RAG system is to incorporate relevant information found by the retriever and discard irrelevant information in order to generate **relevant**, **faithful** responses. 

In the early days, the RAG triad was a popular evaluation framework:

- **Answer relevance**: Whether the response is relevant to the query
- **Context relevance**: Whether context is relevant to the query
- **Groundedness**: Whether the response is supported by context

{{< figure src="https://truera.com/wp-content/uploads/2024/03/TruEra-The-Rag-Triad-1.png" caption="The RAG triad for evaluating relevance (source: [TruEra](https://truera.com/ai-quality-education/generative-ai-rags/what-is-the-rag-triad/))." width="600">}}

Nowadays, RAG evaluation has become more comprehensive. For instance, in addition to the triad, LlamaIndex also [outlined](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/) several other metrics:

- **Correctness**: Whether the generated answer matches that of the reference answer given the query (requires labels).
- **Semantic similarity**: Whether the predicted answer is semantically similar to the reference answer (requires labels).
- **Faithfulness**: Evaluates if the answer is faithful to the retrieved contexts (in other words, whether if there's hallucination).
  - *Tip*: To reduce hallucination, prompt the LLM to cite sources. To detect hallucination automatically, compare multiple generations — factual claims remain stable, while hallucinations tend to vary.
- **Guideline adherence**: Whether the predicted answer adheres to specific guidelines.

Judging correctness and semantic similarity typically requires human evaluation, since a human needs to provide a reference answer to the query and compare it with the LLM-generate response.

For other metrics, LLM-as-a-judge is a scalable alternative: We can use another LLM to evaluate the relevance (answer + context), groundedness, and faithfulness of LLM-generated responses. Be aware of biases: A model usually prefers other models in its family! 

# Fine-Tuning

## Embeddings for Retrieval

<!-- Metric learning 

https://www.yuan-meng.com/posts/negative_sampling/ -->

## LLMs for Generation

# Put RAG in Production
<!-- ## Performance
latency, throughput, memory, compute usage 
## Quality
Truthfulness Handling Unexpected Queries 
## Safety
Security and Privacy
## Logging and Monitoring
per component
- init prompt
- query sent to retriever
- chunks returned
- processing by reranker
- final prompt
- response 
- latency
end to end 

LLM human rule -->

<!-- ## Experimentation -->

# Learn More
## Agentic Search
<!-- create a workflow => use a different LLM for each task
workflow types
- sequential 
- conditional e.g., router
- iterative: writer-eval
- parallel: orchestrator-synthesizer
 -->
## LLamaIndex
## RAG Book