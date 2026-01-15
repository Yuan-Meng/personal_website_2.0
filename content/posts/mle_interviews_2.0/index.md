---
title: "MLE Interview 2.0: Research Engineering and Scary Rounds"
date: 2026-01-15
math: true
categories: ["career", "machine learning", "interview"]
toc: true
---

# Scary New Rounds

> "Here goes nothing." --- Dustin Henderson, *Stranger Things*

{{< figure src="https://i.ebayimg.com/00/s/NjAyWDgxOA==/z/irgAAOSwkz1lo2l~/$_57.PNG?set_id=880000500F" width="400">}}

I have always loved how Dustin from *Stranger Things* says "here goes nothing" before executing an impossible plan. According to [Reddit](https://www.reddit.com/r/etymology/comments/3iwni1/here_goes_nothing/), this phrase means "you have nothing left to lose" or that "even if it fails, you don't lose anything of value". That's probably the mindset to have when interviewing with companies everybody wants to join today.

For ML engineers, companies like Google, Meta, Pinterest, and LinkedIn have established ranking teams, good compensations, and standard "ML design + LeetCode + behavior" interview rounds. I wrote about how to tackle standard MLE interviews in my previous {{< backlink "mle_interviews" "post" >}}. 

For applied/non-research ML or research engineers, only a few places pay more than the above: research engineers at {xAI, OpenAI, Anthropic, DeepMind} and ML engineers at {Netflix, Roblox, Snap, Databricks}. Some have interview rounds I find genuinely scary, e.g., 

- **ML infra design**: I'm **not** talking about ML system design focusing on modeling, where you can get by with briefly mentioning sharding, caching, etc. to check the "scaling" box. That would be easy by now. I'm talking about ML **infra** design that doesn't focus on modeling at all, but instead asks in great detail about the infra around ML systems, such as feature stores, distributed training, and online serving. Such interviews are rare, but they happen to be a required round at companies like Netflix, Snap, and Reddit. I recently wrote a {{< backlink "ml_infra_interviews" "blogpost" >}} on how to prepare for this rare round.
- **Multi-level object-oriented programming**: You'll implement a toy system that mimics a real system you see in life, such as a database, a KV store, a game, to name a few. You'll start with basic functionalities and gradually add more or optimize.
- **LLM coding**: A few years ago, some companies might ask you to implement a simple model (e.g., KNN, K-means, decision trees, logistic regression, linear regression, MLP) from scratch and fit it on toy data. Today, frontier labs will ask you to debug or implement LLM training or inference code --- Transformer encoders/decoders, KV cache, LoRA, or even [autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). 
  - *Why scary*: Even people who use PyTorch daily rarely know low-level details like how computation graphs are built or how autograd works under the hood. When it comes to model architectures, we may know kNN and attention well in isolation, but don't immediately realize how the latter is a softer version of the former. These are just a few examples. Deep down, the real signal is understanding modern deep learning "primitives" inside out. Perhaps the ideal candidate is an ML PhD student who has just passed their qualifying exams. The rest of us can try our best but still miss the mark.
- **Research presentation**: In traditional project deep dives, you verbally walk through one or two representative projects. Some Research Engineer roles require a job talk style presentation on your past work. You make ~10 slides, go over technical details, and "defend" your body of work like a PhD candidate would.

If you think these rounds are daunting, so does everyone else. Mayuko explained in an awesome [YouTube video](https://www.youtube.com/watch?v=e2Y-rhTlHHI) that tech interviews once focused on domain knowledge; then Microsoft popularized domain-agnostic data structures and algorithms questions to give raw talents from all backgrounds a fair chance. For a while, SWE candidates found those interviews scarier than domain knowledge --- until LeetCode came along and gave everyone an efficient way to prepare.

Today, new interview formats feel daunting because they don't have a LeetCode-equivalent yet. We're the pioneers. When doing "standard" ML interviews back in 2024, my goal was converting every onsite into an offer. For research engineering roles with scarier rounds, I've come to realize that we may not get what we want on the first try, but it's always worth learning what we need to get what we want next time.

# Why Do You Wanna Leave?

To do well at MLE interviews 2.0, we inevitably lose some productivity at work, so think clearly about why you wanna leave (every recruiter and hiring manager will ask you anyways). And once you start, commit to finishing within 2 months with full determination. Don't drag it out.

Under normal circumstances, new hires don't work on high-impact projects immediately, and many don't survive long enough to ever take on one. If you're already in a rare position to lead high-impact projects, you'll 99% find yourself in a worse situation at a new company. Don't leave such a job easily, not without a compelling reason.

I was lucky to take on high-visibility, top-down projects shortly after joining Pinterest. So for me to leave, I thought through my goals: 

- **Career development**: I want to advance my career by meaningfully (+~50%) bumping my compensation or seniority.
- **Scope**: I want to own an area tightly linked to the company's mission and bets, so scopes expand and projects don't dry up. 
- **Domain**: I want to work on future-oriented domains and stacks.
  - *Type 1*: foundation ranking models ("LLM-as-rec", "LLM4rec") at companies that need and can afford to train them
  - *Type 2*: applied AI engineering at a frontier lab
- **WLB**: I accept some degradation in WLB as our industry's new reality. 55–65 hrs/week is OK, but 80+ hrs/week is too much.
- **Stability**: Hopefully the company still exists in 3–4 years. That rules out many early-stage startups for me, even if they pay well.

If you don't have compelling reasons, staying put may be the best.

# Up the Ante on Interview Prep

## MLE Interview 1.0 Recap

Most companies still have "standard" MLE interview rounds, typically coding, ML system design, ML fundamentals, and behavior. Some companies have made slight changes. For instance, Meta added an AI-assisted coding round. Companies like Google, LinkedIn, and many startups (e.g., xAI, Perplexity) now conduct in-person onsite. 

For standard MLE interviews, check out my {{< backlink "mle_interviews" "old post" >}}. Below is a recap.

1. **Coding**: Medium–to-Hard LC problems, object-oriented programming (often multi-level CodeSignal), and ML coding (e.g., debugging or building and training a PyTorch/NumPy model)
2. **ML system design**: design a {ranking, query understanding, content understanding, NLP, trust & safety} ML system, focusing on data, features, labels, and model architectures
3. **Project deep dive**: walk through a project you're most proud of
4. **ML fundamentals**: rapid-fire questions on machine learning foundations, such as common network architectures, optimization routines, loss functions, activation functions, regularization, etc.
5. **Behavior and leadership**: a time when you demonstrated problem solving, team work, conflict resolution, time management, leadership, adaptability, growth, etc. beyond your level
6. **Domain knowledge**: woven into all rounds, explaining how you shipped past projects or tackle current problems using expertise


## 2.0 vs.1.0: What Has Changed?

One big change is the additional rounds mentioned in the beginning. Even for the same rounds, the bar is stranger and the process is longer.

### Stranger (Personalized) Bar

I used to have an accurate feel for when an offer was coming. Now I don't know anymore. When I last interviewed in mid-2024, I rigorously followed people's (and my own) advice --- solving coding problems cleanly and quickly, organizing ML system design answers in a “perfect” structure (e.g., clarifications 👉 business goals 👉 ML objectives 👉 high-level ranking funnel 👉 blah blah), and telling polished behavioral stories in impeccable [STAR](https://en.wikipedia.org/wiki/Situation,_task,_action,_result) or [SAIL](https://phyllisnjoroge.medium.com/the-sail-framework-for-behavioral-interview-questions-f66e56eee91a) frameworks.

At the end of 2025, I interviewed with 5 companies and had lots of imperfections here and there (e.g., didn't solve some coding problems, gave long-winded answers in behavioral interviews, focused too much on certain parts of a design and ran out of time). Yet I got offers from 4 of them (the only onsite I failed was for a backend-heavy role).

Is the bar getting lower? Definitely not. Interestingly, the only onsite I failed was one where I sort of did perfectly --- I solved 3 hard coding problems, delivered structured ML designs, and told well-rehearsed behavioral stories. But my experience didn't match (I'm an ML engineer with an understanding of backend systems, and they were looking for a backend engineer with ranking knowledge). I doubted if I could do the job, and indeed I didn't get it. I think <span style="background-color: #D9CEFF">the bar is getting personalized</span> --- headcount is so tight in this market that hiring teams must answer this question: <span style="background-color: #D9CEFF">“Why you? Why not anybody else?”</span>

I think my unique strength lies in the fact that I haven't spent a single day since 2022 not thinking about or reading papers on recommender systems. I'm fascinated by every aspect of RecSys. It's what I do and what I love. For instance, how exactly do we build value models? If we use a single label source (e.g., user engagement), we may end up optimizing for just one objective. But if we combine multiple label sources, then the label weights themselves are exactly what we try to tune when combining predictions. And how do we train early-stage rankers when most instances are not impressed and therefore unlabeled? If we distill from a late-stage ranker, is that late-stage ranker itself even good enough to make predictions on unranked examples (i.e., those never returned by the early-stage ranker)? How do we handle such data drift? And so on --- just to name a few. Even when I managed time badly in a design interview, what I did say out loud may still show what I've thought deeply about in my day-to-day job. I don't know how, but the interviewers must have spotted that.

My honest advice: Find a domain you love from the bottom of your heart and apply to roles in that domain. This way, you worry less about "the bar" and focus more on finding a place that shares your passion.

Nevertheless, below is what to aim for if you want to do "perfectly".

1. **Coding**: As an MLE/RE candidate, you may go through a funny experience of writing an elaborate [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) API yesterday, implementing a Transformer decoder today, and solving an LC hard tomorrow --- all with the expectation to be impeccable. Some friends complain about such expectations, saying (almost) no one in real life is simultaneously a backend genius, a SOTA model author, and a competitive programmer. I just accept it as reality, swallow the pain, and try my best to get ready for all three.
   - **LC**: If a LC-style interview is 45 minutes, you will solve 1 Easy + 1 Medium or 2 Medium (Meta), or 1 Hard with follow-ups (e.g., Google, Snap, Pinterest, Databricks). In the former case, the ideal outcome is to come up with the solution in 2-3 min, code it up in 5-8 min, and do a dry run on in 1 min. In the latter, the ideal outcome is to write a bug-free solution in 15–20 minutes, write and pass essential test cases, and discuss optimizations. For follow-ups, you're expected to provide extendable solutions in code or words. If no follow-ups, it means you've spent too long on the main problem. 
   - **OOP**: In a 1-hour OOP-style interview, you'll be writing a toy API to play a game, manage a database or a process, etc.. The problem is usually organized by levels. The first level is a warmup. Then you gradually expand the API in subsequent levels. Most people can't type fast enough to finish all levels, but an ideal candidate should finish early and leave time to discuss how to scale the toy system in production. 
   - **ML**: In a 1-hour ML coding interview, you're either given some model code (e.g., model class, training and inference loops) and asked to debug bad performance, or you'll implement a model from scratch using NumPy or PyTorch and overfit it on toy samples. Many candidates spend too long reading the code or Googling NumPy/PyTorch syntax. An ideal candidate is fluent in PyTorch and NumPy and can code up common model architectures (e.g., linear layers, residual connections, Transformer blocks, non-linear activation functions) without thinking. Googling many times leaves you no time to debug or implement the model.
2. **ML system design**: By now, everyone knows how to sketch a standard ranking pipeline --- L1 (candidate generation) 👉 L1 (first-stage ranking) 👉 L2 (second-stage ranking) 👉 L3 (call it re-ranking, value models, or "special effects"). If you sound like everyone else, that won't cut it anymore. You need full control of the conversation: use the ranking funnel as scaffolding, but show insights that only a RecSys expert would have. Bring ideas and solutions from your work (e.g., you've worked on lifelong sequence models and they are actually a good choice here) and industry frontiers (e.g., this year's standouts are perhaps "One" series from Kuaishou or {{< backlink "generative_recommendation" "generative recommendation" >}} in general).
3. **Project deep dive**: I've got offers because the team needed someone to work on X, and I happened to have worked on X. In one case, my first interviewer was so interested in X that they skipped the coding question and interviewers in the next few rounds ditched the agenda to ask me more about X. If you work on models with broad industry applications and your team is ahead of the industry, you have a natural advantage over very well-prepared candidates who don't have such experience. I think this is why you need to choose your current projects or next moves carefully --- as an ML engineer, try to choose a company that has the same or a more advanced ML stack as your current one; only move to a company with a {{< sidenote "dated" >}}It doesn't matter at all, for instance, if a ranking team currently uses DCNv2, RankMixer, or some other architectures for feature interaction, because details can change in a few projects, including yours. However, if a company doesn't use DL, has no plan to support DL, and won't benefit from DL, then think twice before joining, however much you like their product. In a few years when you interview again, perhaps for a frontier AI lab or a more mature company, you can speak to scope but won't capture sufficient interests in deep dives, research presentations, or team matches. Your future career may be limited to designer-driven, pre-IPO companies.{{< /sidenote >}} ML stack if you get a once-in-a-lifetime level bump (e.g., +2) or a package that keeps you happy for five years. Otherwise, you're cashing out too early and losing your advantage in 2-3 years. That said, you can still fail if you talk about a complex project as if it were trivial. I wrote a [post](https://www.yuan-meng.com/notes/project_complexity/) on how to present complex projects that do them justice.
4. **ML fundamentals**: Most of the time, you'll get 3–5 rapid-fire ML fundamentals questions in a phone screen, often before coding. Many candidates are confused when they fail, remembering they've solved the coding question perfectly, only to forget that they made fundamental mistakes on ML fundamentals. One or two wrong answers can be enough for rejection. And if those 5 minutes already feels exposing, know that some companies have one or two 45-minute onsite rounds that keep asking ML fundamentals to test the limit of your deep learning knowledge. 
5. **Behavior and leadership**: It's hard to reflect on your career while you're still living it. If you claim to know perfectly well how to expand scope, manage priorities, collaborate with difficult teammates, and right all the wrongs, you're essentially saying "everyone is drunk while I alone am sober" (众人皆醉我独醒). That can't be true. You need repeated, honest conversations with friends and mentors to see where you can grow. Beyond that, you need to be that mentor and friend --- support junior engineers & peers and lead teams through difficult moments --- to show leadership across the board. Just like internal promotions, getting a Senior+ offer often requires showing Staff-level qualities.

### Longer Process

Increasingly more companies now ask for 2–3 references from recent managers, teammates, or internal referrals. I remember this was already the case when I was interviewing for my first job in grad school --- companies like Figma and Roblox have long required pre-offer reference checks. Nowadays, all frontier AI labs ask for them.

Most companies I interviewed with have a team match stage. Last year, it was pretty quick on my end: I was either matched within a day, or the hiring manager who I met during onsite extended a direct offer. This year, however, passing an onsite is still far from an offer: dozens of candidates who have cleared the interview need to talk to a handful of managers with openings. Your years of experience, project complexity, and interview performance are evaluated all over again.

# Preparing for Each Round

## 1. LC-Style Coding

### Don't Be Obsessed with Company Tags

Company tags on LC are added by candidates after seeing these problems in interviews. Some companies rarely update their question bank so their tags cover most of what you'll see in actual interviews. However, you can't plan on that, since occasional updates do happen at any company. Moreover, companies like Google purposefully avoid asking leaked problems and you can't prepare for Apple or Netflix where each team has the freedom to write their own questions. 

As hard as it may sound, you gotta let go of the idea that `solving company tags == solving coding interviews` and actually get good at problem solving and coding up the solution in your mind. 

First thing first, don't theorize problem solving until you've solved enough problems. I always recommend taking the [beginner](https://neetcode.io/courses/dsa-for-beginners/0) and [advanced](https://neetcode.io/courses/advanced-algorithms/0) NeetCode courses first and practicing [problems](https://neetcode.io/practice/practice) curated by NeetCode. I think [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) strikes a good balance between quality and quantity. There's no shortcut. If you have time, you can participate in LC [Weekly Contest](https://leetcode.com/contest/) to rehearse the pressure of solving a random Hard problem under pressure (kinda necessary for Google). 

Then, don't memorize concrete data structures you've used for a problem --- figure that out on the fly. For instance, if a problem gives you some dependencies and asks you to find a valid ordering that satisfies them, it's likely a topological sort problem. Usually, LC problems provide dependencies in the form of edge lists or adjacency lists. But what if a new problem provides dependencies as links or pointers? Can you still recognize the problem? Maybe. Can you still construct the adjacency list and the indegree map correctly? Maybe not. You might struggle between recalling old solutions and reasoning about the problem at hand, run out of time, or keep writing bugs.

The implication: when practicing LC, focus on the problem *at hand*. Given the task (e.g., finding an order under dependency constraints), what's a good/likely algorithm? Then use what you're given right here, right now to implement that algorithm, rather than trying to recall concrete bits and pieces from past solutions. If you do write a bug, refrain from asking GPT or looking at the solution right away. Because in a real interview, this is exactly when you should debug on your own --- by printing suspicious parts, or by thinking from first principles.

That said, don't practice randomly. For instance, while you don't need to (and probably never can) go over all Google tags, you can still research what types of problems are most popular at Google (e.g., [Reddit](https://www.reddit.com/r/leetcode/comments/1izv4ln/how_to_actually_prepare_for_google_via_leetcode/) says DP, graphs, and "fancy string problems") and focus on those. Track which algorithms trip you up the most and practice accordingly. For me, it's definitely advanced graphs, linked lists and trees (I hate pointers), bottom-up DP, greedy, and string evaluation (e.g., Basic Calculator I, II, III). I also accept that as an ML engineer, there are so many other rounds that I can't be perfect at coding. If I run into a red-black tree problem, I'll accept today is not my day.

{{< figure src="https://www.dropbox.com/scl/fi/d6bchbb2xewuycaynckkv/Screenshot-2026-01-11-at-10.31.08-AM.png?rlkey=a3fi4vfnhqvy55da0t2nf5612&st=hb43kyx4&raw=1" width="1800" caption="My practice plan for Google coding interviews.">}}

### Connect LC Problems to Scalable Systems

I was reading the classic [MapReduce](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) paper when it dawned on me that many --- or dare I say, most? --- LC problems stem from web-scale data processing. Google had massive amounts of web search event logs before it had MapReduce; engineers wrote custom scripts to answer questions like: what are the top k queries? how to build inverted indexes? how do we aggregate unbounded data streams?

Many LC problems answer these exact questions and tap into your ability and intuition to process loads of data efficiently. Examples:

- Streaming data & online aggregation 👉 sliding window, two pointers, monotonic queues (e.g., [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/), [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/))
- Counting, frequency, and heavy hitters 👉 hash maps, heaps, bucket sort (e.g., [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/), [Majority Element](https://leetcode.com/problems/majority-element/description/))
- Inverted indices & lookup tables 👉 hash maps, tries (e.g., [Word Pattern](https://leetcode.com/problems/word-pattern/description/), [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/description/), [Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/description/))
- Event scanning and interval logic 👉 prefix sums, line sweep, sorting (e.g., [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/description/), [Car Pooling](https://leetcode.com/problems/car-pooling/), [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/description/))
- Scheduling and resource allocation 👉 greedy + heap (e.g., [Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/), [Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/))
- Data compaction and normalization 👉 union find, graph traversal (e.g., [Accounts Merge](https://leetcode.com/problems/accounts-merge/description/), Sentence Similarity [I](https://leetcode.com/problems/sentence-similarity/description/) & [II](https://leetcode.com/problems/sentence-similarity-ii/description/))

Many complain that LC problems are detached from real work. But once I see them as toy versions of prod data processing and systems tasks, I solve LC problems faster, with more motivation and interest. 

## 2. Objective-Oriented Programming
It's hypocritical of me to say that I like connecting coding interviews to real-world systems when I have a terrible track record of passing object-oriented programming (OOP) interviews, where you do need to build a toy version of a real-world system quickly. (The *quickly* part gets me 💀.) I somehow always get offers from LC-first places first, so I never have to pass OOP interviews. But to get offers from the likes of OpenAI, Databricks, Netflix, and Reddit, OOP is what you must nail.

Below are some typical examples of OOP questions:
- Time-Based Key-Value Store
- In-Memory Database / Data Store
- C-Like Memory Allocator
- Type Inference Engine
- Bank Transaction System
- Employee Management System
- Circuit Breaker Implementation
- API Gateway with Rate Limiting
- LRU / LFU Cache
- Thread Pool / Task Scheduler
- ... 

To be honest, I enjoy LC 10x more than OOP. With OOP problems, the algorithms themselves often aren't hard, but you need enough domain knowledge to immediately know what to code. Writing the classes, testing them (sometimes coming up with representative test cases like a sharp QA engineer), and extending the design is a hell lot of work. After you finally finish everything asked by the prompt, the interviewer often asks how you'd optimize and scale the system in production. I've only coded fast enough to even hear those questions twice 😂.

Would a backend engineer have done it better? Perhaps not. In practice, they rarely need to build a KV store from scratch or implement memory allocation themself --- those low-level details are usually abstracted away. If I were preparing again, I'd probably review NeetCode’s tutorials on [Object-Oriented Design Interview](https://neetcode.io/courses/ood-interview/0) and  [Object-Oriented Design Patterns](https://neetcode.io/courses/design-patterns/0). Then I'd ask GPT to reverse-engineer CodeSignal-style prompts for each of the toy systems above:


```markdown
You are an interview question designer.  

For each system listed below, generate a **CodeSignal-style 4-level coding prompt** that incrementally builds an object-oriented system.

### Systems
- Time-Based Key-Value Store  
- In-Memory Database  
- C-Like Memory Allocator  
- Type Inference Engine  
- Bank Transaction System  
- Employee Management System  
- Circuit Breaker  
- API Gateway with Rate Limiting  
- LRU / LFU Cache  
- Thread Pool / Task Scheduler  

### Requirements
- Each system should have **4 levels**, where:
  - **Level 1**: Basic functionality and core classes (single-threaded, minimal features)
  - **Level 2**: Adds constraints, edge cases, or additional APIs
  - **Level 3**: Introduces concurrency, performance constraints, or correctness guarantees
  - **Level 4**: Extensibility or production-like requirements (pluggability, configuration, failure handling)

- Each level should include:
  - A clear problem statement  
  - Required public methods and their signatures  
  - Explicit constraints and assumptions  
  - Example inputs/outputs or representative test cases  

- Keep the scope realistic for a **45–60 minute interview** per system.
- Focus on **class design, APIs, and invariants**, not UI or persistence.
- Do **not** include the solution—only the prompt.

### Output Format
- Use clear section headers for each system.
- Clearly label **Level 1–4** for each system.
- Keep descriptions concise and interview-realistic.
```

## 3. ML Coding

Okay, so ML coding will be right in our alley, right? That'd be the funniest thing I've heard 🤣🤣🤣. I like OOP 10x more over ML coding.

If I ask my ML engineer friends, 9 out 10 will say they love machine learning, but it will scare the hell out of all to write PyTorch code without inheriting from your team's model class or using Cursor. In ML coding interviews, however, you're expected to write PyTorch as fluently as regular Python. Even if you can Google, you have no time. 

First, be really fluent in PyTorch. Read Raschka's [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) and play with [TensorGym](https://tensorgym.com/) to learn syntax. For a more comprehensive tutorial, go over [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/). 

Next, be really familiar with Transformer architectures as well as training and inference loops. Read Sebastian Raschka's [Build a Large Language Model](https://github.com/rasbt/LLMs-from-scratch) and reproduce all code on your own. Watch Andrej Karpathy's [GPT-2 video](https://www.youtube.com/watch?v=kCc8FmEb1nY). If you have enough time on your hands, follow along with Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and read some of his repos (e.g., [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)). Also be familiar with common optimization techniques for training and inference (e.g., [Flash Attention](https://lubits.ch/flash/), [LoRA](https://lightning.ai/lightning-ai/environments/code-lora-from-scratch?section=featured), [KV cache](https://huggingface.co/blog/kv-cache)) and how to implement them from scratch. 

To test your understanding, solve LC-style ML problems on [Deep-ML](https://www.deep-ml.com/problems). You probably have no time to practice everything, but do go over common model architectures (e.g., KNN, K-Means, linear regression, logistic regression, MLP, CNN, RNN, causal self-attention, bidirectional self-attention, etc.), activation functions, optimizers (you may need to implement backpropagation and autograd from scratch instead of doing `loss.backward()`), evaluation metrics (e.g., nDCG, AUC), as well as training + inference techniques such as LoRA, beam search, KV cache, etc.. If you encounter problems outside of this list, maybe today isn't your day. You can also go over the [notebooks](https://udlbook.github.io/udlbook/) accompanying the *Understanding Deep Learning* book to cover all common ML concepts.

Last but not least, not all companies ask you to write PyTorch models. Some ask you to fit classical Scikit-learn models or use NumPy to implement from scratch. Don't be caught off the guard --- brush up with Educative's [Scikit-learn cheat sheet](https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods). To practice NumPy itself, go over the [numpy-100](https://github.com/rougier/numpy-100) repo and the [*From Python to Numpy*](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) tutorial.


## 4. ML Fundamentals

I find it deeply unsatisfying to collect a checklist of “common questions” and prepare only for those. For example:
- What are common optimizers?
- What are common activation functions?
- What are common loss functions?
- What are common learning rate schedules?
- What to do if the loss doesn't converge?
- What is overfitting? How to prevent it?
- What are vanishing/exploding gradients? How to prevent them?
- What is a Transformer? How do encoders and decoders differ?
- … and so on.

I need to understand how concepts connect at a deeper level. My philosophy is that every minute I spend on interview prep should make me a better engineer. After all, I'm not a professional interviewee; I'm a professional ML engineer. Memorizing answers doesn't do that for me.

I treat ML fundamental prep as an opportunity to revisit the foundations of deep learning and uncover knowledge gaps I didn't know I had. While I like Kevin Murphy's [PML series](https://probml.github.io/pml-book/) as always, the clever bits are often scattered across chapters. I've had frequent epiphanies reading his writing, but it's not always phrased in the "standard language" you or your interviewers instinctively think in.

More recently, I like [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/). It's less mathy and more straight to the point. At least read Chapters 1–9, 11, and 12.

## 5. ML **Model** Design

As I wrote in a recent {{< backlink "ml_infra_interviews" "post" >}}, there is a finite space of ML design problems, 95% generated by the following combinations:

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


Below is how I usually approach ML model design problems: 
1. **Problem framing**: 5-10 min, understand what to build and how
   - **Clarify what system to build**: What's the UI and model entry point? How do users engage with recommendations? How do you want them to? What's the corpus size (i.e., how many items to recommend from)? What's the latency target? So on.
     - I never have a fixed list of questions. I just close my eyes for a second and let my intuitions speak about what's important for this system that could also impact the design. I sometimes play with the team or the company's product and imagine how I'd build it as a PM/EM/MLE.
   - **Understand why build it (business objectives)**: What are the desired actions you want your users to take immediately? What long term benefits do you want to bring to all parties in the system (e.g., users, the platform, and perhaps creators, advertisers, sells, etc.)? What don't you wanna see? These tell us the {short-term vs. long-term, positive vs. negative} metrics you can use to measure the success of your system.
   - **Propose how to build it with ML (ML objectives)**: If you predict one outcome, what would it be? If you build a multi-task model to predict multiple outcomes, what are they?
2. **High-level design**: lay out key components in your system 
    {{< figure src="https://www.dropbox.com/scl/fi/nuhob7pp2qpdjfk7eoc45/Screenshot-2026-01-14-at-11.30.19-PM.png?rlkey=mg4rjqhqcipuxholtqm3s02z5&st=zdot60a9&raw=1" width="800">}}
   - **Online path**: request (e.g., `POST /v1/feed:fetch`) 👉 API gateway (load balancing, authentication, rate limiting) 👉 retrieval 👉 L1 ranking 👉 L2 ranking 👉 logging & monitoring
   - **Offline path**: feature engineering (lambda architecture: batch + streaming processing), training data generation, model training, indexing (ANN, inverted index, forward index)
3. **Step-by-step design**: flesh out ML details in each component 
   - **Retrieval**: billions 👉 thousands; optimize Recall
     - **Model-based**: e.g., embedding-based retrieval (user-to-item, item-to-item), semantic ID match
     - **Heuristic-based**: e.g., popular/trending (time-decayed counts), content from people/topics you follow, etc.
   - **L1 ranking**: thousands 👉 hundreds; also optimize Recall
     - **Objective**: select items that will likely end up in L2 top k
     - **Training data**: impression data (L2 results returned to users + user engagement labels) + early funnel logs (L1 results + synthetic labels from an LLM or an L2 teacher; logs are sampled to keep storage cost reasonable)
     - **Features**: in theory, we can use all L2 features, including request features (user + context), item features, and cross features; in practice, select essential features and avoid high-cardinality cross features to reduce latency 
     - **Model architectures**: e.g., GBDT, two-tower (no feature interaction), simple DNN (e.g., fully connected)
     - **Objectives**: depends on the architecture 👉 e.g., if training two-tower, can use contrastive loss (if we use negative sampling) or just binary cross entropy (predict for each actual example); if training a simple DNN with an L2 teacher, use binary cross entropy + distillation loss
   - **L2 ranking**: hundreds 👉 dozens; optimize ranking metrics
        - **Objective**: predict `pAction` and rank items by it
        - **Training data**: impression data (L2 engagements)
        - **Features**: now we can use a rich set of request features (user + context), item features, and cross features; many companies also include expensive but effective features such as user engagement history for sequence models
        - **Model architectures**: most companies still use DLRM-style models with feature processing (normalization numeric features and pass them through MLP; look up embedding features and project them; pass sequence features to a Transformer module; concatenate outputs), feature interaction (based on MLP/CNN/RNN, attention, explicit cross networks, or a combination of them), and feature transformation layers (e.g., expert sub-networks each processing features in its own way and gating networks leveraging expert outputs for each task)
        - **Objectives**: most modern L2 ranking models predict multiple `pAction` that matter to the system; for each prediction, we can use pointwise loss, pairwise loss, or listwise loss for training 👉 for ads ranking, pointwise loss is most common since it's easy to calibrate; search often uses listwise loss to optimize for the list order
4. **Deep dives**: discuss glossed-over key details, such as how to scale the system (e.g., vertical scaling, sharding, caching, replication + load balancing), how to handle positional bias, cold start, data drifts, and so on. Usually the interviewer will prompt you.

Above is just a skeleton. I find ML model design interview outcomes to be highly dependent on the interviewer. If the interviewer doesn't work on RecSys or is inexperienced, the experience can be excruciating --- the interviewer may over-index on structure and question your choices or explanations when they come from actual practice that the interviewer isn't familiar with. An experienced interviewer, by contrast, will let you skip unimportant parts and dive into the interesting components, asking how you handle tricky situations in day-to-day work or how the industry typically approaches them. Those conversations are lovely. I've only run into the former type once or twice; in those cases, I know I won't join the team.

## 6. ML **Infra** Design

I have colleagues who are both strong ML engineers and ML infra engineers. They have one thing in common: 15+ years of experience, having started as backend/infra engineers before ML was cool, and then growing into ML leaders. Most of my peers and I, however, started our careers as baby ML engineers trying only to move model metrics, without deep low-level ML infra knowledge. Most of the time, you can avoid ML infra interviews --- but a handful of places (e.g., DoorDash, Reddit, Netflix, Snap) specifically ask MLE candidates to design the *infra* behind ML systems. It's funny that I've interviewed with them all and somehow never managed to escape this round.

I've summarized my ML infra interview preparation in this {{< backlink "ml_infra_interviews" "blogpost" >}}. It's pretty long, so I won't repeat its content here in the interest of space. Do check it out! Below are key lessons I've learned:

1. The best way to understand ML infra is to start from the infra teams you collaborate with and learn what they do (e.g., via design docs, [books](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022), and engineering blogs; see [resources](https://www.yuan-meng.com/posts/ml_infra_interviews/#references)) 
   - **Data infra**: e.g., how features are defined, computed, stored, and updated (batch vs. streaming); how engagement events are logged; how training data (features + labels) is generated via forward logging or backfill.
   - **Training infra**: e.g., how training is distributed across GPUs; checkpointing; failure recovery; continuous retraining + validation; model publishing and rollout.
   - **Serving infra**: e.g., how to fetch request (user + context) vs. document vs. cross features; when and how to batch requests; reducing end-to-end latency (caching, sharding, load balancing); real-time feature updates; pagination for large responses; request logging.
2. Brush up on distributed system knowledge, but don't dwell on it. ML infra is a special case of distributed systems, but you usually don't need to go deep on things like rate limiters or post creation/updates. The focus is the infra around **ML**.
   - **NeetCode**: watch [System Design for Beginners](https://neetcode.io/courses/system-design-for-beginners/0) and checkout sections in [System Design Interview](https://neetcode.io/courses/system-design-interview/9) relevant to ML systems, such as KV stores and distributed message queues. 
   - **Hello Interview**: go through [System Design in a Hurry](https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction) as well as core concepts, patterns, key technologies, and advanced topics like time-series databases (think feature stores).
   - [**DDIA book**](https://dataintensive.net/): skim Chapters 1-11 if you have time.
3. To practice, design feature stores, training data generation, distributed training, and retrieval & ranking (organic + ads), etc.. Spend 2–3 days per design to think through all details. Below is the table of contents in a toy ads ranking system design I wrote.
    ```
    ## Table of Contents

    1. **Problem Framing**
       - Clarification: What to Build  
         - Charging Events  
         - Attribution  
         - Candidate Size  
         - Scalability
       - Business Objectives: Why Build It
       - ML Objectives: How to Build with ML

    2. **Problem Statement**

    3. **High-Level Design**
       - Online Path
       - Offline Path

    4. **Online Path**
       - Core Entities
       - API and Data Flow
         - Ads Serving API
         - CandidateGenerationService
         - L1RankingService
         - L2RankingService
         - AuctionPolicyService
         - Logging

    5. **Offline Path**
       - Feature Engineering
       - Data Generation
       - Model Training
       - Item Indexing

    6. **Step-by-Step Design**
       - Candidate Generation
         - Ads Targeting
         - Budget Checks
         - Other Checks
       - L1 Ranking
         - Objectives
         - Training Data
         - Features
         - Model Architectures
       - L2 Ranking
         - Objectives
         - Training Data
         - Features
         - Model Architectures

    7. **Production**
       - Monitoring
       - Experimentation
       - Rollout
       - Rollback

    8. **Deep Dives**
       - Delayed Conversions
       - CVR Calibration

    9. **References**
       - Posts
       - CTR Papers
       - CVR Papers
    ```

## 7. Behavior Interview
<!-- Show the list and prep guide. -->

## 8. Project Deep Dive
<!-- Coming soon... -->
