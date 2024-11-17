---
title: "Down the Rabbit Hole: Sequential User Modeling"
date: 2024-11-17
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

A Markov-chain recommender predicts the transition probability from one action sequence ($S = \\{\mathrm{printer}, \mathrm{ink}\\}$) to another ($S = \\{\mathrm{printer}, \mathrm{ink}, \mathrm{paper}\\}$), either directly (e.g., [Garcin et al., 2013](https://arxiv.org/abs/1303.0665)) or by embedding Markov chains into a latent space to compute transitional probabilities based on Euclidean distances (e.g., [Feng et al., 2015](https://www.ijcai.org/Proceedings/15/Papers/293.pdf)). 

{{< figure src="https://www.dropbox.com/scl/fi/pgjlb1nee84lihensin30/Screenshot-2024-11-12-at-12.59.47-PM.png?rlkey=a43d16ssyuhjjtxmzyldqawpu&st=nfyg4npx&raw=1" caption="Markov chains recommend items by transition probabilities ([Feng et al., 2015](https://www.ijcai.org/Proceedings/15/Papers/293.pdf))." width="600">}}

A fatal shortcoming of Markov chains lies in the ["memorylessness"](https://en.wikipedia.org/wiki/Memorylessness) assumption --- that future states depend only on the current state, ignoring preceding states. For example, if I bought cat food (because I suddenly remembered 😂) after buying ink, the system won't be more likely to recommend paper to me than if I didn't buy ink at all. Real shoppers jump between diverse interests and engage with unrelated items, which are nuances that Markov chains cannot capture.

### Recurrent Neural Networks

Instead of relying solely on today's input to predict tomorrow, a Recurrent Neural Network (RNN) maintains a "hidden state" that serves as a memory of yesterday's activation and that of all days prior. The hidden state is "hidden" because it doesn't produce any observable output, such as a character or an action. Today's input, combined with the previous hidden state, is used to predict tomorrow. The last hidden state can be used to represent the sequence so far. 

{{< figure src="https://www.dropbox.com/scl/fi/r05faxbpxpevzgtjywz2m/Screenshot-2024-11-12-at-7.18.01-PM.png?rlkey=1p88m4grpfp1imww83d9lf8vx&st=nbyecz33&raw=1" caption="The vanilla RNN architecture that almost nobody uses directly ([Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network))." width="1800">}}


Generally speaking, a RNN maps the input and the hidden state at step $t$, $x_t$ and $h_t$, to an output $y_t$ and an updated hidden state $h_{t+1}$,  

$$
f_{\theta}(x_t, h_t) \rightarrow (y_t, h_{t+1}),
$$

where:

- $x_t$: input vector;  
- $h_t$: hidden vector;  
- $y_t$: output vector; 
- $\theta$: neural network parameters.

Vanilla RNNs are difficult to train because they are susceptible to the vanishing or exploding gradient problem, where gradients become increasingly small or large when we are backpropagating errors from the last output to the first input. More advanced versions RNNs, such as [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Units (GRUs)](https://en.wikipedia.org/wiki/Gated_recurrent_unit), were invented to address these issues. From the early 2010s until the rise of Transformers around 2017-2018, these RNN variants were state-of-the-art in Natural Language Processing (NLP).

[Wu et al. (2017)](https://research.google/pubs/recurrent-recommender-networks/) introduced the Recurrent Recommender Network (RRN), using LSTMs to encode user and movie hidden states, which then help predict how a user might rate a movie at a given time. RRN outperformed [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) methods such SVD++, which won the Netflix Prize but ignored the temporal dynamics of users and movies. The idea is based on a few clever observations:

- **Users evolve with movies and over time**: After watching a great detective movie like *Knives Out*, I want to see more like it; as I grow older, I appreciate non-action movies more...
- **Movies evolve with audiences and over time**: Every Christmas, *Elf* sees a resurgence in popularity; winning an award often leads to a sudden spike in appreciation for a film; some "sleeper hits" take years to build a reputation..

{{< figure src="https://www.dropbox.com/scl/fi/qn3dzi4d7ajmmvbc9xgru/Screenshot-2024-11-12-at-9.15.11-PM.png?rlkey=mrud07o647yakehuf7cxw87rq&st=syacksj8&raw=1" caption="RRN uses 2 LTSMs to capture user and movie hidden states ([Wu et al., 2017](https://research.google/pubs/recurrent-recommender-networks/))." width="1800">}}

The authors used one LSTM to model user hidden states and another to model movie hidden states. A time $t$, the predicted rating of user $i$ on movie $j$, $\hat{r}_{ij|t}$, is a combination the user's and the movie's dynamic latent factors (the last hidden states of the LTSMs) and their stationary latent factors (representing stable user and movie traits), 

$$
\hat{r}\_{ij|t} = f(u\_{it}, m\_{jt}, u_i, m_j) := \langle \tilde{u}\_{it}, \tilde{m}\_{jt} \rangle + \langle u_i, m_j \rangle,
$$

where $\langle \cdot, \cdot \rangle$ denotes an inner product, and:

- $\tilde{u}\_{it}$ and $\tilde{m}\_{jt}$ are time-varying states $u\_{it}$ (for the user) and $m\_{jt}$ (for the movie) generated by the LSTM networks;
- $u_i$ and $m_j$ are stationary latent factors for the user (e,.g., profile, long-term interests) and the movie (e.g., genre), respectively.

The model is trained to minimize the error between the predicted rating $\hat{r}\_{ij|t}$ and the actual rating $r\_{ij|t}$. It captures exogenous dynamics (e.g., winning an award) and endogenous dynamics (e.g., seasonality) of movies more effectively than models that don't take account of temporal information. 

LTSMs in RRN can be replaced with other RNN variants, such as GRUs (e.g., [Hidasi et al., 2016](https://arxiv.org/abs/1511.06939)) and hierarchical RNNs (e.g., [Quadrana et al., 2017](https://dl.acm.org/doi/abs/10.1145/3109859.3109896)). Regardless of the specific architecture, RNN-based recommenders share some common drawbacks:

- **False dependencies**: Just because I bought cat litter after buying books doesn't mean the two are related. User action sequences contain lots of noise, which RNNs cannot sift through;
- **Ignoring union-level dependencies**: Multiple past actions can collectively predict a future action. For example, after buying rum and mint, it makes sense to guess I'm making a Mojito and recommend lime juice, but not if I just bought one of them. It's hard for RNNs to capture such "union-level dependencies".

### Convolutional Neural Networks

Convolutional Neural Networks (CNNs) were initially developed for Computer Vision (CV) and are known for their ability to automatically extract image features, eliminating the need to hand-engineer features. In a [previous post](https://www.yuan-meng.com/posts/human_vision/#convolutional-neural-network-cnn), I reviewed the CNN architecture in detail.

CNNs apply to recommendation because we can look up item embeddings in a sequence and stack them into a 2D matrix, which we can treat as a 2D image ([Tang & Kang, 2018](https://arxiv.org/pdf/1809.07426)) or a 1D image whose "color channel" is the embedding dimension ([Yuan et al., 2019](https://arxiv.org/abs/1808.05163)). Unlike RNNs that process inputs sequentially, CNNs process the entire image in parallel, greatly improving training speed. Moreover, CNNs are capable of capturing union-level dependencies, where multiple actions together determine a future action --- this is hard for RNNs which best capture point-level dependencies from one action to another. 

{{< figure src="https://www.dropbox.com/scl/fi/za225afaopif9mpd6wc23/Screenshot-2024-11-13-at-9.13.07-PM.png?rlkey=ca284r83tx8z6zv4xkswin58d&st=ub0lhkdv&raw=1" caption="The first CNN recommender [*Caser*](https://arxiv.org/pdf/1809.07426) treats the embedding matrix as a 2D 'image'." width="1800">}}

*Caser* was the first CNN recommender to beat the RNN SOTA in sequential recommendation. Given a user sequence of length $L$, we perform an embedding lookup and stack item embeddings into matrix $\mathbf{E}$ of dimension $L \times d$, where $d$ is the embedding dimension. Horizontal filters of dimension $h \times d$ slide over the rows of $\mathbf{E}$ to capture union-level dependencies (e.g., $(\mathrm{rum}, \mathrm{mint}) \rightarrow \mathrm{lime})$, where $h$ is the height of the filter --- we can think of it as the size of the union. We use max pooling to extract the max value from each horizontal filter. Vertical filters of dimension $L \times 1$ slides over the columns of $\mathbf{E}$ to capture point-level dependencies. No max pooling is needed since we want to keep every item latent dimension. 

Outputs from the convolutional layers are concatenated with the user embedding (generated by a Latent Factor Model) and fed into an MLP layer to predict the probability of each possible next action. We can recommend the top N items with the highest predicted probabilities.

{{< figure src="https://www.dropbox.com/scl/fi/5tvwhafskm9awdch582kd/Screenshot-2024-11-13-at-8.48.28-PM.png?rlkey=7bp27lrq17dm7b4ho5lzjfktr&st=b2cxqikr&raw=1" caption="Yuan et al., 2019 transformed the 2D image into a 1D representation and applied 1D dilated convolution without max pooling." width="1800">}}

In Caser, max pooling leads to information loss, and small filter sizes make it challenging to capture long-range dependencies. Yuan et al. (2019) addressed these issues by (1) removing max pooling and (2) using dilated convolutions, where some positions are filled with 0's. In general, larger filters increase the numbers of parameters to learn, making CNNs harder to train --- dilated convolution strikes a good balance between the receptive field size and training efficiency.

Despite these optimizations, capturing super long-range dependencies and looking beyond local patterns is inherently difficult for CNNs.

### Graph Neural Networks

A common challenge for sequential recommendation is the noise in user actions. None of the methods above (Markov chains, RNNs, CNNs) has a mechanism to distinguish noise (e.g., an accidentally clicked item) from signal (i.e., an item of interest). The solution proposed in the SURGE paper ([Chang et al. (2021)](https://arxiv.org/pdf/2106.14226) is converting loose item sequences into tight item-item interest graphs $\mathcal{G} = \\{\mathcal{V}, \mathcal{E}, A\\}$, where each node $v \in \mathcal{V}$ is an interacted item, $A$ is the adjacency matrix, and $\mathcal{E}$ are edges learned via {{< sidenote "node similarity metric learning" >}}Here, the metric function between nodes $h_i$ and $h_j$ is a weighted cosine similarity of their embeddings, $M_{ij} = \cos(\vec{\textbf{w}} \odot \vec{h}_i, \vec{\textbf{w}} \odot \vec{h}_j)$, where $\odot$ denotes the Hadamard product (an element-wise operation that takes 2 vectors of the same dimension and multiples the corresponding elements together) and $\vec{\textbf{w}}$ is a trainable weight learned end-to-end in the downstream recommendation task.{{< /sidenote >}}. Noise is dealt with through graph sparsification, where edges with low ranking in learned metrics are pruned, since they are likely accidental.  

{{< figure src="https://www.dropbox.com/scl/fi/629z5iedbo3yv1lrlpoiq/Screenshot-2024-11-13-at-11.59.24-PM.png?rlkey=gfl4zzrrs9tgu2w6m09bzn1fg&st=gfsuhnmz&raw=1" caption="GNNs aggregate item embedding via the interest graph. A sparse graph that represents the user's strongest interest is used for downstream prediction." width="1800">}}

Once the interest graphs are constructed, the Interest-Fusion Graph Convolutional layer uses an attention mechanism to weight the importance of each neighbor and aggregates their information to update the node embeddings accordingly. Items that reflect the user's core interests (typically closer to the cluster center) or are related to the query item (i.e., the target item to be scored) receive higher weights. This further reduces noise in the item embeddings and emphasizes the user's long-term and immediate interests. Graph pooling is then used to downsample the graphs, preserving the strongest interest signals from the user. At this point, the noisy user sequence has been converted into a compact representation of user interests, which can be flattened into a 1D sequence for prediction. 

## Target Attention

The GNN "alchemy" is one way to extract interest signals from noisy sequences. A more straightforward approach is via *target attention*. 


### One-Stage: DIN (2018) and DIEN (2019)

{{< figure src="https://www.dropbox.com/scl/fi/7pw0aj09hnr4zi8q7lh36/Screenshot-2024-11-15-at-9.27.07-PM.png?rlkey=ofrz5862p3sel38c9tewe9i9c&st=0qi0yzq3&raw=1" caption="DIN calculates the attention score between each item and the target item and uses the scores for weighted sum pooling. It is not itself a sequential model." width="1800">}}

Should you show me an ad for a MacBook keyboard cover? Knowing that I bought a MacBook, it'd be a great suggestion. By contrast, other items I've bought, such as cat food or fitness accessories, have no bearing on this particular interest. <span style="background-color: #abe0bb">User interests are diverse, and only parts of user sequences shed light on their interest in the target item</span>. This observation motivated the Deep Interest Network (DIN, [Zhou et al., 2018](https://arxiv.org/pdf/1706.06978)) and many target attention models that followed. Rather than doing a simple average or sum pooling over engaged items, this family of models uses target attention to weigh each item by its relevance to the target item and perform weighted sum pooling afterward.

Note that DIN is not a sequential model, because the attention score between an engaged item and the target item is the same regardless of the position of the engaged item in the sequence. The Deep Interest Evolution Network (DIEN, [Zhou et al., 2019](https://arxiv.org/abs/1809.03672)) introduced a year later employs GRUs to capture dependencies between interactions. DIEN was motivated by two key observations: (1) user interests are diverse (e.g., I like electronics and stationery, while my cats enjoy toys and cat food) and (2) each interest evolves independently over time (e.g., I need accessories for newer iPhone/MacBook/Apple Watch models, and my cats are only bothered by cooler toys).

{{< figure src="https://www.dropbox.com/scl/fi/e7c9rbkvy3wwmjvfiw8zp/Screenshot-2024-11-15-at-9.38.27-PM.png?rlkey=bzxdzlhutqn7xbhfxweb2ixyz&st=za8nlqsb&raw=1" caption="DIEN uses a variant of RNN --- GRU -- to capture dependencies between interactions. Each interaction is represented by its hidden state." width="1800">}}

The DIEN architecture has two components --- an Interest Extractor Layer to extract interest states from user sequences and an Interest Evolution Layer to model interest evolution w.r.t. the target item.

- **Interest Extractor Layer**: Each hidden state $\mathbf{h}_t$ represents the user interest state after action $i_t$. The output is an interest sequence concatenated from all hidden states, $[\mathbf{h}_1, \ldots, \mathbf{h}_L]$.
    - **Auxiliary loss**: Only after the final action can we predict the target item. To provide extra supervision in earlier steps, the authors introduced an auxiliary task, predicting the $(t+1)$-th action based on $\\{i_1, \ldots, i_t\\}$. The actual action $i_{t+1}$ serves as the positive instance, while a randomly sampled action serves as the negative instance. The task is to predict which is positive using the binary cross-entropy loss --- 
    $$L_{aux} = -\frac{1}{N} \left( \sum_{i=1}^{N} \sum_{t} \log \sigma(\mathbf{h}_t^i, \mathbf{e}_b^i[t+1]) + \log (1 - \sigma(\mathbf{h}_t^i, \hat{\mathbf{e}}_b^i[t+1])) \right),$$
    where $\mathbf{e}_b^i[t+1])$ and $\hat{\mathbf{e}}_b^i[t+1]$ are positive and negative item embeddings, respectively, and $\sigma(\mathbf{x_1}, \mathbf{x_2}) = \frac{1}{1 + \exp(-[\mathbf{x_1}, \mathbf{x_2}])}$ denotes the {{< sidenote "predicted score" >}}The notation is a bit unconventional, but inferring from the context, $-[\mathbf{x_1}, \mathbf{x_2}]$ is not a concatenation but an operation such as a dot product that computes the similarity between two vectors and returns a scalar.{{< /sidenote >}}.
    - **Global loss**: The global loss for the target item CTR prediction task is given by $L = L_{target} + \alpha \cdot L_{aux}$, where $\alpha$ is a hyperparameter to balance interest representation (from the auxiliary task) and the final CTR prediction.

- **Interest Evolution Layer**: The previous Interest Extractor Layer uses one GRU to learn each interest state up until time step $t$; the Interest Extractor Layer uses a second GRU to generate the final sequence representation. The update to each hidden state is controlled by its attention score with the target item.
    - **Attention function**: The relation between interest state $\mathbb{h}\_t$ and the target item $a$ is given by the attention function ---
    $$a_t = \frac{\exp(\mathbb{h_t}We_a)}{\sum_{j=1}^T \exp(\mathbb{h}_jWe_a)},$$
    where $e_a$ represents the target item embedding. A higher attention score indicates a stronger connection between the given interest state and the target item being considered. 

    - **GRU with attentional update gate (AUGRU)**: The hidden state update is controlled by attention score $a_t$, reducing the impact of interests less related to the target item on the hidden state. This weakens the disturbance from noise.
    $$\tilde{\mathbb{u}}\_t^{\prime} = a_t \cdot \mathbb{u}\_t^{\prime},$$
    $$\mathbb{h}\_t^{\prime} = (1 - \tilde{\mathbb{u}}\_t^{\prime}) \circ \mathbb{h}\_{t-1}^{\prime} + \tilde{\mathbb{u}}\_t^{\prime} \circ \tilde{\mathbb{h}}\_t^{\prime},$$
    where $\mathbb{u}^{\prime}_t$ is the original {{< sidenote "update gate" >}}In GRUs, the update gate determines how much of the previous hidden state should be carried forward into the current state.{{< /sidenote >}} and $\tilde{\mathbb{u}}\_t^{\prime}$ is the attention update gate. $\mathbb{h}\_t^{\prime}$ (soley based on sequential dependencies) and $\tilde{\mathbb{h}}\_t^{\prime}$ (also incorporating target attention) are hidden states in the Interest Extractor Layer and the Interest Evolution Layer, respectively. 

### Two-Stage: General Search Unit + Exact Search Unit

The original DIEN models up to 50 most recent actions, which is fine for capturing short-term interests. As the cash cow for e-commerce and social media companies, <span style="background-color: #abe0bb">deep CTR models are getting increasingly more competitive ('卷' 🤑) --- a new hope for performance gain lies in modeling long-term user sequences</span>. Target attention has a time complexity of $O(L \cdot B \cdot d)$, where $L$ is the sequence length, $B$ is the number of target items to score, and $d$ is the hidden dimension of item embeddings. A user's lifelong history can contain $10^3$ to $10^5$ interactions, making training inefficient for ultra-long sequences. An emerging paradigm for long-term sequential user modeling is to cascade sequential modeling into two stages ---

- **General Search Unit (GSU)**: Retrieve top $k$ items from the long-term user sequence that are most similar to the target item;
- **Exact Search Unit (ESU)**: Only compute target attention between each of the top $k$ items and the target item.

In the ESU step, we can pick a model from the DIN/DIEN family. Two-stage target attention models mainly differ in the GSU step --- i.e., how they retrieve the top $k$ items to balance performance and speed. 

#### SIM (Alibaba, 2020)

{{< figure src="https://www.dropbox.com/scl/fi/ogkh0f6g20va4kujl0hea/Screenshot-2024-11-16-at-12.31.56-PM.png?rlkey=4z8p7evygvml1bzsyjbuqvcvw&st=cft0v0gh&raw=1" caption="SIM retrieves top $k$ items most similar to the target using Maximum Inner Product Search (soft search) based on item embeddings or from an inverted index based on item categories (hard search). The sum pooling of top $k$ item embeddings and the target item embedding are used in CTR predictions." width="1800">}}

Alibaba's Search-based Interest Model ([SIM, 2020](https://arxiv.org/abs/2006.05639)) is the first two-stage target attention model. The GSU step in SIM finds the top $k$ items most relevant to the target item in one of two ways --- 

- **Hard search** (based on a category-based inverted index): Only retrieve items in the same category as the target --- $\mathrm{Sign}(C\_t = C\_a)$, where $C_t$ and $C_a$ denote categories of the $t$-th item in the sequence and the target item $a$, respectively;
- **Soft search** (based on item embeddings): The relevance score between the item interacted at $t$ and the target item $a$ is defined as $(\mathbf{W}\_i \mathbb{e}\_t) \odot (\mathbf{W}\_a \mathbb{e}\_a)^T$, where $\mathbb{e}_t$ and $\mathbb{e}\_a$ are embeddings of the two items, and $\mathbf{W}\_i$ and $\mathbf{W}\_a$ are weight matrices used to transform embeddings before the similarity calculation. We can conduct Maximum Inner Product Search (MIPS) over weighted item embeddings to find top $k$ items most relevant to the target.

In soft search, the user sequence representation generated in the GSU step is a weighted sum pooling of item embeddings, $\sum_{t=1}^L r\_t \mathbb{e}_t$, which is then concatenated with the target item embedding $\mathbb{e}_a$ before being passed into the MLP layer. The GSU and ESU steps are trained jointly, with a total loss given by $L = \alpha \cdot L\_{GSU} + \beta \cdot L\_{ESU}$.

In hard search, items are stored in an inverted index keyed by `user_id` and `category_id`. This index is built and updated separately from the model. Hard search is the deployed model because it offers only slightly worse evaluation performance compared to soft search but is much easier to train and serve than the latter.

#### ETA (Alibaba, 2021)

Due to training and serving challenges, Alibaba opted for hard search in SIM. However, it is slightly jarring to plug offline-generated top $k$ items into an online CTR model. For instance, the pre-built index can become outdated, leading to model degradation. The follow-up End-to-End Target Attention ([ETA, 2021](https://arxiv.org/pdf/2108.04468)) paper used a clever trick to accelerate MISP, enabling end-to-end GSU and ESU in online serving.

{{< figure src="https://www.dropbox.com/scl/fi/mzupzfwvu7s7o1ea187k7/Screenshot-2024-11-16-at-12.33.00-PM.png?rlkey=1t8nbpnm5x62cvn650rtuzffd&st=7628vptt&raw=1" caption="ETA hashes embeddings into binary vectors, reducing each $O(d)$ dot product operation into an $O(1)$ Hamming distance computation, accelerating top-$k$ retrieval, which allows for end-to-end online serving of GSU + ESU." width="1800">}}

The trick is to hash real-valued embeddings into binary vectors using [SimHash](https://en.wikipedia.org/wiki/SimHash), reducing vector similarity scoring from a dot product with time complexity $O(L \cdot B \cdot d)$, to a [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) {{< sidenote "calculation" >}}It takes $O(1)$ time to find differing bits between two binary numbers using bitwise XOR (`diffs = x ^ y`) and count 1's in the result (`bin(diffs).count('1')`).{{< /sidenote >}}with time complexity $O(L \cdot B)$. This speeds up top-$k$ retrieval, allowing efficient end-to-end GSU + ESU in both training and serving.

{{< figure src="https://www.dropbox.com/scl/fi/zuvbjq3ucy43n8pymw179/Screenshot-2024-11-16-at-4.27.27-PM.png?rlkey=91kk6sfm1ceoydjnxjxraswyo&st=l6qoa2xm&raw=1" caption="SimHash is locality-sensitive, generating similar outputs from similar inputs." width="600">}}

{{< figure src="https://www.dropbox.com/scl/fi/c881roy6bx3lc14mvgquz/Screenshot-2024-11-16-at-4.51.16-PM.png?rlkey=ueh3vfaztbg0a9tnjc51gqrvr&st=k6g338nd&raw=1" width="600">}}

#### The Kuaishou TWINs (2023, 2024)

Alibaba is an e-commerce platform: there are only so many products one wants to browse and has the money to buy. In contrast, short video users watch hundreds of thousands of videos over their lifetime, making it crucial for the GSU to retrieve the top $k$ items from an ultra-long sequence riddled with noise that the ESU will consider relevant. . The Chinese short video company Kuaishou is an industry leader in ultra-long sequence modeling, publishing the SOTA TWo-stage Interest Network ([TWIN, 2023](https://arxiv.org/abs/2302.02352)) and its "twin" ([TWIN-V2, 2024](https://arxiv.org/abs/2407.16357)).

Below are the key observations + innovations behind TWIN ---

{{< figure src="https://www.dropbox.com/scl/fi/z5m19lma0po88rxqxao1i/Screenshot-2024-11-16-at-12.33.36-PM.png?rlkey=dn9ezoosfr8f9j2al3ki5oga9&st=qal3uxr2&raw=1" caption="TWIN optimizes target attention by splitting features into inherent vs. user-item cross features, caching inherent features and simplifying cross-feature projections. Attention scores are used as the relevance metric in GSU + ESU." width="1800">}}

- **Feature splits**: A sequence of length $L$ results in a feature matrix $K \in \mathbb{R}^{L \times M}$, where $M$ is the dimension of each feature vector. When computing target attention between $K$ and the query (i.e., the target item), the linear projection of $K$ is the bottleneck. <span style="background-color: #abe0bb">User and item features are typically split into two parts --- inherent features $K_h \in \mathbb{R}^{L \times H}$ (e.g., video id, author, duration) and user-item cross features $K_c \in \mathbb{R}^{L \times C}$ (e.g., click timestamp, play time)</span>,
    $$K \triangleq \left[ K_h \quad K_c \right] \in \mathbb{R}^{L \times (H + C)}.$$
    - **Cache inherent features**: Inherent features are often projected into large hidden dimensions (e.g., $H = 64$). Once projected, they can be shared across sequences (e.g., Bob is always Bob, no matter which video he watches). We can cache the $K_h$ projection to reduce computational costs. 
    - **Simplify cross features projections**: Cross features cannot be cached since most only appear once (e.g., users {{< sidenote "rarely" >}}A 大佬 once said, the greatest feat of an ML engineer is to tailor a solution to the real business/user problem. The key assumption in TWIN only stands because Kushaisho users don't re-watch videos, but it'd be stupid to assume DoorDash consumers don't reorder dishes.{{< /sidenote >}} watch the same video again), but we can simplify their linear projection. Say we have $J$ cross features ($C = 8J$), each feature $K\_{c, j} \in \mathbb{R}^{L \times 8}$ can be projected into an embedding dimension of 8. Multiplying each cross feature by a weight vector $\mathbf{w}\_j^c \in \mathbb{R}^8$ returns a 1D vector for each feature $K\_{c, 1}\mathbf{w}\_j^c \in \mathbb{R}^L$. This reduces the linear projection to
    $$K_c W^c \triangleq \left[ K\_{c,1} \mathbf{w}\_1^c, \ldots, K\_{c,J} \mathbf{w}\_J^c \right].$$
- **Consistency-Preserved GSU (CP-GSU)**: A major issue with previous two-stage models is that GSU and ESU use different relevance metrics, hurting both recall (not all relevant items are returned) and precision (some top $k$ items are irrelevant). TWIN addresses this by using attention scores as a consistent relevance metric in both stages. After feature splitting and simplified linear projections, target attention in TWIN is defined as
    $$\mathbf{\alpha} = \frac{(K_h W^h)(\mathbf{q}^\top W^q)^\top}{\sqrt{d_k}} + (K_c W^c) \mathbf{\beta},$$
    where $d_k$ is the dimension of the projected query and key. The cross features serve as the bias term $\mathbf{\beta}$. In GSU, top 10 items with highest attention scores are returned. The same scores are used as weights when performing a weighted average pooling in ESU.

Feature splits make target attention computations faster for longer sequences, while CP-GSU increases the likelihood that the top $k$ items retrieved in GSU will be the final items of interest in ESU. 

TWIN-V2 enhances the ability to model ultra-long ($> 10^6$) sequences by aggregating similar items into clusters and using a cluster to represent many a similar items in it, thereby reducing the sequence from $S = \\{i_1, \ldots, i_L\\}$ to $\hat{S} = \\{c_1, \ldots, c_{\hat{L}}\\}$, where $\hat{L} \ll L$. Each cluster is represented by a "virtual item" --- for numerical features, we take the average across all items in the cluster; for categorical features, we take feature values of the item closest to the centroid.

{{< figure src="https://www.dropbox.com/scl/fi/99csxcencja4m8wtv3k34/Screenshot-2024-11-16-at-12.34.18-PM.png?rlkey=rx9wm0j01xcb2c1tzhkmojxdp&st=6xj90rki&raw=1" caption="TWIN-V2 clusters similar items and uses a virtual item to represent each cluster. GSU retrieves the top 100 clusters and ESU computes attention scores between the virtual item in each retrieved cluster and the target item." width="1800">}}

{{< figure src="https://www.dropbox.com/scl/fi/vvpy0juljindvyhr7oxie/Screenshot-2024-11-17-at-9.26.31-AM.png?rlkey=gcnmh012h719hgxjq6rxzsq92&st=y1muupbn&raw=1" width="600">}}

- **GSU**: Retrieve top $k=100$ clusters whose "virtual items" have the highest attention scores with the target item --- TWIN-V2 adjusts attention scores by cluster sizes $\mathbf{n} \in \mathbb{N}^{\hat{L}}$, $\mathbf{a}^{\prime} = \mathbf{a} + \ln \mathbf{n}$;
- **ESU**: Compute attention between virtual items and the target.

As far as *ultra-long* sequential user modeling goes, TWIN-V2 may be the "craziest" yet. **Bonus question**: How would you make it {{< sidenote "crazier" >}}As a product ML engineer, I often sigh at the "chasm" between research and reality. The more SOTA papers I read, however, the more I started to realize good ideas often come naturally. For instance, TWIN-V2 is clearly motivated by the need to compress a list vectors and clustering is the path taken. Why not ask ourselves, what else is there?{{< /sidenote >}}?

#### Temporal Interest Module (TIM, 2024)

> As far as *ultra-long* sequential user modeling goes, TWIN-V2 may be the "craziest" yet.

But in terms of information to encode from a user sequence, it is not 🤯. The Temporal Interest Module (TIM) in a new Tencent paper ([Pan et al., 2024](https://arxiv.org/abs/2403.00793)) crammed everything possible into sequential features. 

{{< figure src="https://www.dropbox.com/scl/fi/mddhbpsmnt4qc2ind8dt0/Screenshot-2024-11-17-at-9.52.10-AM.png?rlkey=zms5sred20ofe12ifrr4eabj6&st=lj0fkzkd&raw=1" caption="TIM is part of a gigantic model used for ads prediction across Tencent products." width="1800">}}

Inspired by positional encoding in Transformer models, TIM introduces a target-aware temporal encoding, $\boldsymbol{p_f}(X_t)$, to capture the relative position or discretized time interval of each interaction in a user sequence. This encoding is added to each interaction’s embedding, producing a temporally encoded embedding: $\boldsymbol{\tilde{e}_t} = \boldsymbol{e_t} \oplus \boldsymbol{p_f}(X_t)$. The encoded user sequence $S = \{i_1, \ldots, i_L\}$ is summarized as:

$$
\boldsymbol{u_{\text{TIM}}} = \sum_{X_t \in S} \alpha(\boldsymbol{\tilde{e}_t}, \boldsymbol{\tilde{v}_a}) \cdot (\boldsymbol{\tilde{e}_t} \odot \boldsymbol{\tilde{v}_a}),
$$

where $\alpha(\boldsymbol{\tilde{e}_t}, \boldsymbol{\tilde{v}_a})$ denotes the target-aware attention between the interaction at time $t$ and the target item $a$, and $(\boldsymbol{\tilde{e}_t} \odot \boldsymbol{\tilde{v}_a})$ denotes the target-aware representation that captures feature interactions.

## "Language Modeling"

In our profession as ML engineers in search/rec/ads ("搜广推"), there is a common saying: <span style="background-color: #abe0bb">"A user sequence is just like a sentence in a language."</span> But I've come to feel that's not quite the case. In natural language, syntactic constraints typically prevent speakers from expressing multiple, entangled thoughts in parallel. In contrast, users often jump between interests --- starting with a search, moving to a stream of random discoveries, and then clicking on an ad along the way. If user sequences were sentences, they might read like this:

> <span style="color: #002676;">I</span>, <span style="color: #002676;">plan</span>, <span style="color: #002676;">to</span>, <span style="color: #002676;">start</span>, <span style="color: #FDB515;">I</span>, <span style="color: #002676;">model</span>, <span style="color: #FDB515;">need</span>, <span style="color: #002676;">training</span>, <span style="color: #FDB515;">to</span>, <span style="color: #002676;">work</span>, <span style="color: #FDB515;">do</span>, <span style="color: #002676;">before</span>, <span style="color: #FDB515;">grocery</span>, <span style="color: #002676;">Monday</span>, <span style="color: #FDB515;">shopping</span>, <span style="color: #FDB515;">on</span>, <span style="color: #FDB515;">Sunday</span>...

While language modeling objectives have been adapted for sequential user modeling, careful domain adaptions are likely a good idea.

### Masked Action Modeling: BERT4Rec (2019)

The observation that user interests do not strictly evolve from left to right, but are instead intertwined, inspired Alibaba's [BERT4Rec (2019)](https://arxiv.org/abs/1904.06690). Unlike RNN-based sequential models such as DIEN, BERT4Rec uses bidirectional self-attention to model dependencies in user sequences. 

{{< figure src="https://www.dropbox.com/scl/fi/k95vxf9rbub3rmvfpi395/Screenshot-2024-11-17-at-11.12.04-AM.png?rlkey=vlcbge9e4t20qkw9hvy9wewuy&st=d8o2jt2w&raw=1" caption="BERT4Rec applies the masked language modeling objective, randomly selecting items to mask and predicting ids of masked items based on bidirectional contexts. The last hidden state of the target item represents the user sequence." width="1800">}}

BERT4Rec is trained with the [masked language modeling](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) (MLM) objective: at each training step, $k$ items are randomly chosen in the sequence of length $L$ and replaced with a special `[mask]` token. The model predicts the original ids of the masked items based on their left and right contexts. A bonus point of this setup is data efficiency: random masking gives us $\binom{L}{k}$ training samples over multiple epochs.

The target item is always masked, and the last hidden state is used to represent the sequence in downstream recommendation tasks.

### Dense All Action Prediction: PinnerFormer (2022)

<!-- TransAct (Pinterest, 2023) [repo](https://github.com/pinterest/transformer_user_action) -->

## Is Attention What You Need?

### Sequential Transduction: HSTU (2024)

# What Else Is There?

## Embedding&MLP Paradigm 

## Up the Ante in the Ranking Game

{{< backlink "negative_sampling" >}}
{{< backlink "human_vision" >}}
{{< backlink "ebr" >}}
{{< backlink "attention_as_dict" >}}

<!-- #### TIM (Tencent, 2024)

{{< figure src="https://www.dropbox.com/scl/fi/pb0wk5pm0mahuui0si6hp/Screenshot-2024-11-16-at-12.35.04-PM.png?rlkey=nrpy54qvo2wjmwzxetx7lecjf&st=1b4llezs&raw=1" caption="TIM" width="1800">}}

TIM is part of the agglomeration of all tricks. 
 -->

---
# References

## Overview & Collections
1. "Old" but popular lit review 👉 [*Sequential Recommender Systems: Challenges, Progress and Prospects*](https://arxiv.org/abs/2001.04830) (2019) by Wang et al., *IJCAI*.
2. A Meta MLE's awesome post 👉 [*User Action Sequence Modeling: From Attention to Transformers and Beyond*](https://mlfrontiers.substack.com/p/user-action-sequence-modeling-from) 
3. GitHub repos of sequential user modeling 👉 papers ([Awesome-Sequence-Modeling-for-Recommendation](https://github.com/HqWu-HITCS/Awesome-Sequence-Modeling-for-Recommendation)) + code ([FuxiCTR](https://github.com/reczoo/FuxiCTR))

## Approach: Pre-Transformer
4. Markov chains kick-started sequential recs 👉 [*Personalized News Recommendation with Context Trees*](https://arxiv.org/abs/1303.0665) (2013) by Garcin et al., *RecSys*.
5. RNNs once ruled sequential modeling 👉 LTSM: [*Recurrent Recommender Networks*](https://research.google/pubs/recurrent-recommender-networks/) (2017) by Wu et al., *WSDM*.
    - Other RNN variants: [GRUs](https://arxiv.org/abs/1511.06939), [hierarchical RNNs](https://dl.acm.org/doi/abs/10.1145/3109859.3109896)
6. CNNs capture union-level dependencies 👉 [*Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding*](https://arxiv.org/abs/1809.07426) (2018) by Tang and Wang, *WSDM*.
    - No max pooling: [*A Simple Convolutional Generative Network for Next Item Recommendation*](https://arxiv.org/abs/1808.05163) (2018) by Yuan et al., *WSDM*.
7. GNNs extract interests from noise 👉 [*Sequential Recommendation with Graph Neural Networks*](https://arxiv.org/abs/2106.14226) (2021) by Chang et al., *SIGIR*.

## Approach: Target Attention
8. Overview: [*Target Attention Is All You Need: Modeling Extremely Long User Action Sequences in Recommender Systems*](https://mlfrontiers.substack.com/p/target-attention-is-all-you-need) by Samuel Flender.
9. The OG architecture 👉 DIN: [*Deep Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/1706.06978) (2018) by Zhou et al., *KDD*.
    - And its many a Alibaba siblings: [DIEN (2019)](https://arxiv.org/abs/1809.03672), [DSIN (2019)](https://arxiv.org/abs/1905.06482), [DHAN (2020)](https://arxiv.org/abs/2005.12981), [DMIN (2020)](https://dl.acm.org/doi/abs/10.1145/3340531.3412092), [DAIN (2024)](https://arxiv.org/abs/2409.02425), ...
10. Go crazy on sequence length 👉 SIM: [*Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction*](https://arxiv.org/abs/2006.05639) (2020) by Qi et al., *CIKM*.
    - Ultra long: [ETA (2021)](https://arxiv.org/abs/2108.04468), [TWIN (2023)](https://arxiv.org/abs/2302.02352), [TWIN-V2 (2024)](https://arxiv.org/html/2407.16357v1), ...
    - Review post: [*Towards Life-Long User History Modeling in Recommender Systems*](https://mlfrontiers.substack.com/p/towards-life-long-user-history-modeling) by Samuel Flender.
11. Squeeze every ounce of sequences 👉 TIM: [*Ads Recommendation in a Collapsed and Entangled World*](https://arxiv.org/abs/2403.00793) (2024) by Pan et al, *KDD*.
    - Paper summary: [*Breaking down Tencent's Recommendation Algorithm*](https://mlfrontiers.substack.com/p/breaking-down-tencents-recommendation) by Samuel Flender.

## Approach: Language Modeling

12. The OG 👉 [*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*](https://arxiv.org/abs/1904.06690) (2019) by Sun et al., *CIKM*.
13. Play with objectives 👉 [*PinnerFormer: Sequence Modeling for User Representation at Pinterest*](https://arxiv.org/abs/2205.04507) (2022) by Pancha et al., *KDD*.
14. Capture short-term interests 👉 [*TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest*](https://arxiv.org/abs/2306.00248) (2024) by Xia et al., *KDD*.
15. Applications at Pinterest 👉 organic ranking ([*Large-scale User Sequences at Pinterest*](https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9)) + ads ranking ([*User Action Sequence Modeling for Pinterest Ads Engagement Modeling*](https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e))

## Approach: Beyond Attention
16. Meta AI 👉 [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152) (2024) by Zhai et al., *ICML*.

