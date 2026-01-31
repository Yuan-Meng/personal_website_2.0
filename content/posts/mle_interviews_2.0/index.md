---
title: "MLE Interview 2.0: Research Engineering and Scary Rounds"
date: 2026-01-31
math: true
categories: ["career", "machine learning", "interview"]
toc: true
---

# Scary New Rounds

> "Here goes nothing." --- Dustin Henderson, *Stranger Things*

{{< figure src="https://i.ebayimg.com/00/s/NjAyWDgxOA==/z/irgAAOSwkz1lo2l~/$_57.PNG?set_id=880000500F" width="400">}}

I have always loved how Dustin from *Stranger Things* says "here goes nothing" before executing an impossible plan. According to [Reddit](https://www.reddit.com/r/etymology/comments/3iwni1/here_goes_nothing/), this phrase means "you have nothing left to lose" or that "even if it fails, you don't lose anything of value". That's probably the mindset to have when interviewing with companies everybody wants to join.

For ML engineers, companies like Google, {{< sidenote "Meta" >}}Later I learned that some non-MSL organizations at Meta can match applied/non-research offers from OpenAI, Databricks, and Snap, so this list is just a rough reference.{{< /sidenote >}}, Pinterest, and LinkedIn have established ranking teams, good compensation, and standard "ML design + LeetCode + behavior" interview rounds. I wrote about how to tackle standard MLE interviews in my previous {{< backlink "mle_interviews" "post" >}}.

For **applied/non-research** ML or research engineers, only a few places can pay more than the above at the same level: research engineers at {xAI, OpenAI, Anthropic, Google DeepMind, Microsoft AI} and ML engineers at {Snap, Databricks, Netflix, Roblox}. Most of these companies have interview rounds I find genuinely scary --- for instance: 

- **ML infra design**: I'm **not** talking about ML system design focusing on modeling, where you can briefly mention sharding, caching, etc. to show your "scaling" awareness. That would be easy by now. I'm talking about ML **infra** design that doesn't focus on modeling at all, but instead asks in great detail about the infra around ML systems, such as feature stores, distributed training, and online serving. Such interviews are rare, but they happen to be a required round at companies like Netflix, Snap, Reddit, DoorDash, Notion, and so on. I recently wrote a {{< backlink "ml_infra_interviews" "blogpost" >}} on how to prepare for this rare but critical round (rather than giving up).
- **Multi-level object-oriented programming**: You'll implement a toy system that mimics a real-world backend system, such as a database, a KV store, a chat room, a game, to name a few. You'll start with basic methods and gradually add more or scale up.
- **LLM coding**: A few years ago, many companies liked to ask candidates to implement a simple model (e.g., KNN, K-means, decision trees, logistic regression, linear regression, MLP) from scratch or load a scikit-learn model and fit it on toy data. Today, frontier labs might ask you to debug or implement language model training or inference code --- e.g., Transformer encoders/decoders, KV cache, LoRA, or even [autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). 
  - *Why scary*: Even people who use PyTorch daily rarely know low-level details like how computation graphs are built or how autograd works under the hood. When it comes to model architectures, we may know KNN and attention well in isolation, but don't immediately realize how the latter is a softer version of the former. These are just a few examples. Deep down, the real signal is understanding modern deep learning "primitives" inside out. Perhaps the ideal candidate is an ML PhD student who has just passed their qualifying exams. The rest of us can try our best but still miss the mark.
- **Research presentation**: In traditional project deep dives, you verbally walk through one or two representative projects. Some Research Engineer roles require a job talk style presentation on your past work. You make ~10 slides, go over technical details, and "defend" your body of work like a PhD candidate would.

If you think these rounds are daunting, so does everyone else. Mayuko explained in an awesome [YouTube video](https://www.youtube.com/watch?v=e2Y-rhTlHHI) that tech interviews once focused on domain knowledge; then Microsoft popularized domain-agnostic data structures and algorithms problems to give raw talents from all backgrounds a fair chance. For a while, SWE candidates found those interviews scarier than domain knowledge --- until LeetCode came along and gave everyone an efficient way to prepare.

Today, new interview formats feel daunting because they don't have a LeetCode-equivalent yet. We're the pioneers. When doing "standard" ML interviews back in 2024, my goal was converting every onsite interview into an offer. For research engineering roles with scarier rounds, I've come to realize that we may not get what we want on the first try, but it's worth learning what we need to there next time :)

# Why Do You Wanna Leave?

To do well at MLE interviews 2.0, we inevitably lose some productivity at work, so think clearly about why you wanna leave (every recruiter and hiring manager will ask you anyways). Once you start, commit to finishing in 2 months with full determination. Dragging it out hurts your sleep and work performance. You're more ready than you think. 

Under normal circumstances, new hires don't work on high-impact projects right away, and many don't survive long enough to ever take on one. If you're already in a rare position to lead high-impact projects continuously, you'll find yourself in a worse situation at 99% new companies. Don't leave such a job without a compelling reason.

I was lucky to take on high-visibility, top-down projects shortly after joining Pinterest. So for me to leave, I {{< sidenote "thought through" >}}I started writing this blogpost towards the end of last year, back when leaving Pinterest required a strong upside. Now I feel 懂的都懂 😂. No need to think through anything now to see it's high time to leave.{{< /sidenote >}} my goals: 

- **Career development**: I want to advance my career by meaningfully bumping my compensation (2x) or seniority.
- **Scope**: I want to own an area tightly linked to the company's mission and bets, so scopes expand and projects don't dry up. 
- **Domain**: I want to work on future-oriented domains and stacks.
  - *Type 1*: foundation ranking models ("LLM-as-rec", "LLM4rec") at companies that need and can afford to train them
  - *Type 2*: applied AI engineering at a frontier lab
- **WLB**: I accept some degradation in WLB as our industry's new reality. 55–65 hrs/week is OK, but 80+ hrs/week is too much.
- **Stability**: Hopefully the company still exists in 3–4 years. That rules out many {{< sidenote "early-stage" >}}Or can OpenAI or Perplexity? There's no way to know.{{< /sidenote >}} startups for me, even if they pay well.

If you don't have compelling reasons, staying put may be the safest.

# Up the Ante on Interview Prep

## MLE Interview 1.0 Recap

Most companies still have "standard" MLE interview rounds, typically coding, ML system design, ML fundamentals, and behavior. Some companies have made slight changes. For instance, Meta added an AI-assisted coding round. Companies like {{< sidenote "Google" >}}I think the policy changes quite often. At the end of 2025, Google was doing in-person onsite interviews but as of early 2026, it might have gone back to virtual onsites.{{< /sidenote >}}, LinkedIn, and many startups (e.g., xAI, Perplexity) now conduct in-person onsites. 

For standard MLE interviews, check out my {{< backlink "mle_interviews" "old post" >}}. Below is a recap.

1. **Coding**: Medium–to-Hard LC problems, object-oriented programming (multi-level CodeSignal exercises), and ML coding (e.g., debug or build a PyTorch/NumPy model 👉 train it)
2. **ML system design**: design a {recommendation, search, trust & safety, content understanding, user understanding, NLP, CV} ML system, focusing on data, features, labels, and architectures
3. **Project deep dive**: walk through a project you're most proud of
4. **ML fundamentals**: rapid-fire questions on machine learning foundations, such as common network architectures, optimization routines, loss functions, activation functions, regularization, etc.
5. **Behavior and leadership**: a time when you demonstrated problem solving, team work, conflict resolution, time management, leadership, adaptability, growth, etc. beyond your level
6. **Domain knowledge**: woven into all rounds, explaining how you ship projects and tackle problems using your domain expertise

## 2.0 vs.1.0: What Has Changed?

One big change is the additional rounds mentioned in the beginning. Even for the same rounds, the bar is stranger and the process is longer.

### Match-Oriented Bar

I used to have an accurate hunch for when an offer was coming. Now I don't know anymore. When I last interviewed in mid-2024, I rigorously followed people's (and my own) advice --- solving coding problems cleanly and quickly, organizing ML system design answers in a "perfect" structure (e.g., clarifications 👉 business goals 👉 ML objectives 👉 high-level ranking funnel 👉 blah blah), and telling polished behavior stories in impeccable [STAR](https://en.wikipedia.org/wiki/Situation,_task,_action,_result) or [SAIL](https://phyllisnjoroge.medium.com/the-sail-framework-for-behavioral-interview-questions-f66e56eee91a) frameworks.

At the end of 2025 and the start of 2026, I did onsite interviews with $N \in [5, 10)$ companies and had imperfections here and there (e.g., bombed some coding questions, gave long-winded answers in behavior interviews, focused too much on certain parts of a design and ran out of time). Yet I received offers from $(N-1)$ of them. The only onsite I failed was for a backend role (shouldn't have got that far).

Is the bar getting lower? Definitely not. The only onsite I failed was one where I sort of did "perfectly" --- I solved all hard coding problems, delivered structured system designs, and told well-rehearsed behavior stories. But my experience didn't match --- I'm an ML engineer with an understanding of backend systems, and the team were looking for a backend engineer with ranking knowledge. I doubted if I could do the job, and indeed I didn't get the job. I think <span style="background-color: #D9CEFF">the bar is getting more based on match</span> --- headcount is so tight in this market that hiring teams must answer this question: <span style="background-color: #D9CEFF">*"Why you? Why not anyone else?"*</span>

I think my strength lies in the fact that I haven't spent a single day since 2022 not thinking about or reading papers on recommender systems. I'm fascinated by every aspect of RecSys. It's what I do and what I love. For instance, how do we build value models? If we use a single label source (e.g., user engagement), we may end up optimizing for that one objective. But if we combine multiple label sources, then label weights themselves are exactly what we try to tune when combining predictions. And how do we train early-stage rankers when most responses are not impressed and therefore unlabeled? If we distill from a late-stage ranker, is that late-stage ranker itself good enough to make predictions for unranked examples (i.e., those never returned by the early-stage ranker)? How do we handle such data drifts? Just to name a few. Even when I managed time badly in a design interview (i.e., not finishing up the ranking funnel), what I did say might've still shown what I've thought deeply about in my day-to-day job. I don't know how, but interviewers must have spotted that.

My honest advice: Find a domain you love from the bottom of your heart and apply to roles in that domain. This way, you worry less about "the bar" and focus more on finding a place that shares your passion.

Nevertheless, below is what to aim for if you want to do "perfectly".

1. **Coding**: As an MLE/RE candidate, you may get a funny experience of writing a toy backend API yesterday, implementing a Transformer decoder today, and solving an LC hard tomorrow --- all with the expectation to be fast, optimal, and well-designed. Some friends complain about such expectations, saying (almost) no one in real life is at once a backend genius, a SOTA model creator, and a competitive programmer. I just accept these requirements as reality, swallow the pain, and try my best.
   - **LC**: If an LC-style interview is 45 minutes, you will likely solve 1 Easy + 1 Medium or 2 Medium problems in a Meta interview, or 1 Hard problem with multiple follow-ups in a Google/Snap/Pinterest/Databricks/etc. interview. In the Meta case, the ideal outcome is to instantly think of the solution upon seeing the prompt, explain the solution in 2-3 min, code it up in 5-8 min, and do a dry run in 1 min. In the latter cases, the ideal outcome is to write a working solution within 20 minutes, create and pass sharp test cases, and discuss optimizations. For follow-ups, you're expected to provide extendable solutions in code or words. If no follow-ups, it means you've spent too long on the main problem. 
   - **OOP**: In a 1-hour OOP-style interview, you'll implement a toy API to play a game, manage a database or a process, etc.. Problems are typically organized by levels. The first level is a warmup. Then you gradually expand the API in subsequent levels. The ideal candidate not just has a good pace to finish all levels, but also make reasonable design choices for a real-world backend system. It's good to leave some time in the end to discuss how to scale this toy system in production.
   - **ML**: In a 1-hour ML coding interview, you're given some model code (e.g., model classes, training and inference loops) and asked to debug bad performance, or you implement a model from scratch using NumPy/PyTorch and overfit it on toy samples. Many candidates spend too long reading the code provided or Googling NumPy/PyTorch syntax. An ideal candidate is fluent in PyTorch or NumPy and can code up common architectures (e.g., MLP, CNN, RNN, Transformer encoders/decoders) and building blocks (e.g., linear layers, projections, residual connections, layer norm, batch norm, causal self-attention, bidirectional self-attention, activation functions, optimizers) from memory. Excessive Googling leaves you no time to debug or implement the model.
2. **ML system design**: By now, everyone knows how to sketch a multi-stage ranking pipeline --- L0 (candidate generation) 👉 L1 (first-stage ranking) 👉 L2 (second-stage ranking) 👉 L3 (call it re-ranking, value model, or "special effects"). If you sound like everyone else, that won't cut it anymore. You need full control of the conversation: use the ranking funnel as scaffolding, but show insights that only a RecSys expert would have. Bring ideas and solutions from your work (e.g., you've worked on lifelong sequence models and they are actually a good choice here) and industry frontiers (e.g., this year's standouts are perhaps Kuaishou's "One" series or {{< backlink "generative_recommendation" "generative recommendation" >}} in general). That said, only bring up a solution if it solves a real problem (e.g., scaling up, cold start), not because you know it.
3. **Project deep dive**: I've got one offer because the team needed someone to work on X, and I happened to have worked on X. In this case, my first interviewer was so interested in X that they skipped the coding question and interviewers in the next few rounds ditched the agenda to ask me more about X. If you work on models with broad industry applications and your team is ahead of the industry, you have a natural advantage over very well-prepared candidates who don't have such experience. I think this is why you need to choose your current projects or next moves carefully --- as an ML engineer, try to choose a company that has the same or a more advanced ML stack as your current one; only move to a company with a {{< sidenote "dated" >}}It doesn't matter at all, for instance, if a ranking team currently uses DCNv2, RankMixer, or some other architectures for feature interaction, because details can change in a few projects, including yours. However, if a company doesn't use DL, has no plan to support DL, and won't benefit from DL, then think twice before joining, however much you like their product. In a few years when you interview again, perhaps for a frontier AI lab or a more mature company, you can speak to scope but won't capture sufficient interests in deep dives, research presentations, or team matches. Your future career may be limited to designer-driven, pre-IPO companies.{{< /sidenote >}} ML stack if you get a once-in-a-lifetime level bump (e.g., +2) or a package that keeps you happy for the rest of your career. Otherwise, you're cashing out too early and losing your edge in 2-3 years. That said, you can still fail if you talk about a complex project in a trivial manner. I wrote a [post](https://www.yuan-meng.com/notes/project_complexity/) on how to do complex projects justice in deep dives.
4. **ML fundamentals**: Most of the time, you'll get 3–5 rapid-fire ML fundamentals questions in a phone screen, often before coding. Many candidates are confused when they fail, remembering they've solved the coding question perfectly, only to forget that they made fundamental mistakes on ML fundamentals. One or two wrong answers can be enough for rejection. And if those 5 minutes already feels exposing, know that some companies have one or two 45-minute onsite rounds that keep asking ML fundamentals to test the limit of your deep learning knowledge. 
5. **Behavior and leadership**: It's hard to reflect on your career while you're still living it. If you claim to know perfectly well how to expand scope, manage priorities, collaborate with difficult teammates, and right all the wrongs, you're essentially saying "everyone is drunk while I alone am sober" (众人皆醉我独醒). That can't be true. You need repeated, honest conversations with friends and mentors to see where you can grow. Beyond that, you need to be that mentor and friend --- support junior engineers & peers and lead teams through difficult moments --- to show leadership across the board. Just like internal promotions, getting a Senior+ offer often requires showing Staff-level qualities.

### Longer Process

More companies now ask for 2–3 references from recent managers, teammates, or internal referrals. I remember this was already the case when I was interviewing for my first job in grad school --- companies like Figma, Notion, Roblox, and Databricks have long required pre-offer reference checks. Nowadays, all frontier AI labs ask for them. So, it's important to be a kind and competent engineer while on a team and keep in touch with former colleagues and managers afterward.

Most companies have a team match stage. Back in 2024, I either received direct offers from hiring managers I met during onsites or a list of 5+ teams with openings. Finding a good fit was quick. This year, I don't know if it's due to leveling (>= senior) or the market, most companies I interviewed with only had a couple of openings. If you just want to get matched to *a* team, the process can still move fast. But if you want to find a high-visibility, high-technical-complexity team that aligns well with your experience, you may need to wait for candidates ahead of you to turn down offers, or for another such team to open a headcount (this can take weeks). If there is a good team, dozens of candidates who have cleared the interview may want to speak with the hiring manager. Your years of experience, project complexity, and interview performance are evaluated all over again.

# Preparing for Each Round

## 1. LC-Style Coding

### Don't Be Obsessed with Company Tags

As hard as it may sound, you gotta let go of the idea that `solving company tags == passing coding interviews` and actually get good at problem solving and coding up the solution you're thinking of.

LeetCode company tags came from candidates who encountered these problems in interviews. Some companies rarely update their question bank so their tags cover most of what you'll see in yours. However, you can't bank on that, since any company updates their questions occasionally. Moreover, companies like Google purposefully avoid asking leaked problems and you simply can't prepare for places like Apple or Netflix where each team writes their own questions. 

So how do you get good at solving (most) random LC problems?

First thing first: don't theorize about solving LC problems until you've solved enough of them. I always recommend taking the [beginner](https://neetcode.io/courses/dsa-for-beginners/0) and [advanced](https://neetcode.io/courses/advanced-algorithms/0) NeetCode courses first, and then practicing the [problems](https://neetcode.io/practice/practice) curated by NeetCode. [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) strikes a good balance between quality and quantity. There's no shortcut --- you either put in the hard work now and do what I mentioned, or did competitive programming as a kid; no other way to be great at LC. If you have time, I recommend participating in LC [Weekly Contest](https://leetcode.com/contest/) to rehearse solving random Hard problems under time constraints (kinda necessary for Google). 

Then, don't memorize concrete data structures you've used for a specific problem --- figure things out on the fly. After sufficient practice above, whenever I'm given a new problem, I stare at the prompt and test cases for about 30 seconds, after which an intuition about the solution usually jumps out. I jot down my fleeting thought on paper before it escapes me, then verbalize the solution step by step to the interviewer. If I don't get a flash of insight, I never get a "strong yes". With prompting, I may stumble toward a solution and get a "weak yes". In a tennis analogy, I'm good at "3-setters" but fall short in "5-set marathons". You might be better at taking hints, pivoting, and clutching out a solution after a long struggle. Unless you're a competitive programmer, accept that we can't solve all arbitrary coding problems. 

An example of "figuring it out on the fly": if you're given some dependencies and asked to find a valid ordering that satisfies them, it's 99% a topological sort problem. At an abstract level, you need to construct an adjacency list and an indegree map for all nodes. That's all you have to remember from past solutions. You don't need to remember, for instance, that LC problems often provide dependencies as edge lists (weighted or unweighted, directed or bidirectional). A new problem may provide dependencies as links or pointers instead. Can you still recognize the problem? Maybe. Can you still construct the adjacency list and indegree map correctly? Maybe not. You might find yourself oscillating between trying to recall old solutions and reasoning about the current problem, only to write bugs and run out of time. The Reddit [thread](https://www.reddit.com/r/leetcode/comments/1h886z6/most_frequent_google_interview_questions_on/) below mirrors my preparation process:

{{< figure src="https://www.dropbox.com/scl/fi/gr3tknplbn96loxapztvg/Screenshot-2026-01-31-at-12.16.10-AM.png?rlkey=ctun8ztfnb027wmxvrwgwvlhc&st=i9pn9th5&raw=1" width="800">}}

The implication: when practicing LC, focus on the problem *at hand*. Given the task (e.g., finding a valid order under dependency constraints), what's a good algorithm? Then use the input and starter code you're given *right here, right now* to implement that algorithm. Don't try to recall concrete bits from past solutions. If you do write a bug during practice, refrain from asking GPT or looking at the solution right away. Because in a real interview, now is the exact moment that makes or breaks this interview. If you're able to remain calm and debug --- by printing suspicious parts or thinking from first principles --- you send strong signals as a resourceful problem solver. If you panic or can't think of what to look into, it's a strong signal for rejection. 

That said, don't practice randomly. For instance, while you don't need to (and probably never can) go over all Google tags, you can still research what types of problems are most popular at Google (e.g., [Reddit](https://www.reddit.com/r/leetcode/comments/1izv4ln/how_to_actually_prepare_for_google_via_leetcode/) says DP, graphs, and "fancy string problems") and focus on those. Track which algorithms trip you up the most and practice accordingly. For me, it's definitely advanced graphs, linked lists and trees (I hate pointers), bottom-up DP, greedy, and string evaluation (e.g., Basic Calculator I, II, III). I also accept that as an ML engineer, there are so many other rounds that I can't be perfect at coding. If I run into a red-black tree problem, I just move on ("人固有一死" 😂).

{{< figure src="https://www.dropbox.com/scl/fi/d6bchbb2xewuycaynckkv/Screenshot-2026-01-11-at-10.31.08-AM.png?rlkey=a3fi4vfnhqvy55da0t2nf5612&st=hb43kyx4&raw=1" width="1800" caption="My practice plan for Google coding interviews.">}}

### Connect LC Problems to Scalable Systems

I was reading Google's classic [MapReduce](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) paper when it dawned on me that many --- or dare I say, most? --- LC problems stem from web-scale data processing. Google had massive amounts of web search event logs before it had MapReduce; engineers wrote custom scripts to answer questions like: what are the top k queries? how to build inverted indexes? how do we aggregate unbounded data streams?

Many LC problems answer these exact questions and tap into your ability and intuition to process loads of data efficiently. Examples:

- Streaming data & online aggregation 👉 sliding window, two pointers, monotonic queues (e.g., [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/), [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/))
- Counting, frequency, and heavy hitters 👉 hash maps, heaps, bucket sort (e.g., [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/), [Majority Element](https://leetcode.com/problems/majority-element/description/))
- Inverted indices & lookup tables 👉 hash maps, tries (e.g., [Word Pattern](https://leetcode.com/problems/word-pattern/description/), [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/description/), [Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/description/))
- Event scanning and interval logic 👉 prefix sums, line sweep, sorting (e.g., [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/description/), [Car Pooling](https://leetcode.com/problems/car-pooling/), [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/description/))
- Scheduling and resource allocation 👉 greedy + heap (e.g., [Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/), [Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/))
- Data compaction and normalization 👉 union find, graph traversal (e.g., [Accounts Merge](https://leetcode.com/problems/accounts-merge/description/), Sentence Similarity [I](https://leetcode.com/problems/sentence-similarity/description/) & [II](https://leetcode.com/problems/sentence-similarity-ii/description/))

Many complain that LC problems are detached from real work. But once I see them as toy versions of prod data processing and systems tasks, I solve LC problems faster, with more motivation and interest. 

## 2. Objective-Oriented Programming
It's hypocritical of me to say that I like connecting coding interviews to real-world systems when I have a terrible {{< sidenote "track record" >}}I said I have a high onsite conversion rate. That's because I don't often move to onsite after encountering OOP phone screens 😂.{{< /sidenote >}} of passing object-oriented programming (OOP) interviews, where you need to build a toy version of a real-world system quickly. Example problems:

- Time-Based Key-Value Store  
- In-Memory Database  
- C-Like Memory Allocator  
- Type Inference Engine   
- Circuit Breaker  
- API Gateway with Rate Limiting  
- LRU / LFU Cache  
- Thread Pool / Task Scheduler  
- Transaction / Credit System  
- Employee Management System 
- ... 

I used to think it's the *quickly* part that got me. Then, in a recent mock interview with a backend engineer, I came to see the real issue: my code passes test cases and looks clean, but my design choices are ridiculous for real-world backend systems --- I make choices no good backend engineer would make. For example, when designing a cart, I chose to store price, units, and other attributes directly in an `Item` data class, whereas a backend engineer may use a unique `product_id` and link it to external metadata when needed. This is one of many decisions that I don't even realize I'm getting wrong. The key point is that OOP interviews are as much about *system design* as they are about coding. Even if you finish all CodeSignal levels quickly, you can still fail due to poor design choices. Then, after you complete everything the prompt asks for, the interviewer often asks how you'd optimize and scale this system in production. That's another place where non-backend engineers may fail a seemingly smooth interview.

To join companies such as OpenAI, Anthropic, Databricks, Netflix, Roblox, Notion, and Reddit, you have to pass OOP interviews with flying colors. Some folks obsess over collecting problems other candidates have seen in interviews, but that doesn't help if you make bad design decisions when solving these problems. If you can make good decisions, you don't need to see the problem beforehand.

Without domain knowledge, we make ridiculous choices. However, we probably don't have time to work as a backend engineer before an interview. We can still fill in the gap a bit before attempting problems blindly. After that enlightening mock interview, I was pointed to an awesome [OOP crash course](https://www.coditioning.com/app/learning/courses/tech_interview_prep/4). Go through it first. Then, ask GPT to reverse-engineer CodeSignal-style prompts for each common system:

```markdown
You are an interview question designer.  

For each system listed below, generate a **CodeSignal-style 4-level coding prompt** that incrementally builds an object-oriented system.

### Systems
- Time-Based Key-Value Store  
- In-Memory Database  
- C-Like Memory Allocator  
- Type Inference Engine   
- Circuit Breaker  
- API Gateway with Rate Limiting  
- LRU / LFU Cache  
- Thread Pool / Task Scheduler  
- Transaction / Credit System  
- Employee Management System  

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

## 3. AI-Assisted Coding

Back in 2024, I interviewed with Quora. In a 90-minute coding round, you're given a realistic codebase (adapted from a real Python package repo) plus some unit tests, and you have to debug the codebase by reading error logs, identifying what's broken, and fixing them on your own, without AI. It was the hardest coding interview I've passed.

Meta's AI-assisted coding round is a tamed version of that. You get a much smaller toy codebase plus unit tests, and you're allowed to use an LLM to debug and complete the implementation. I interviewed with Meta in a hurry and only had time to look at the example question in the candidate portal. It was perhaps the easiest coding interview I've passed, as the problem was quite solvable on the spot. 

1. Read the instructions to understand the task and constraints.
2. Open and skim each file to learn what exists and how things are wired. No need to read them in detail, but note what's implemented vs. missing vs. commented to see what you must do.
3. Tell the interviewer your plan (e.g., re-read the instructions, fix unit tests, start with a naive implementation, then optimize).
4. Get unit tests passing. You can use AI to help, but you should be able to explain (a) why tests failed or were incomplete, and (b) why the changes better match the intended behavior.
5. Implement a naive solution first. The core problem you need to implement usually reduces to a LeetCode Medium problem; explain the algorithm, then use AI to write the code.
6. Measure performance. Define a performance metric and run simulations or benchmarks to evaluate the naive solution.
7. Optimize. Identify what's inefficient, propose a better algorithm, use AI to implement it, and evaluate the improved solution.

I don't think my interviewer cared how much code was written by AI vs. me. They cared more about how well I partnered with AI to achieve the end goal. I didn't write much code myself, but I designed the algorithm, broke down the task, and prompted the AI to complete each task --- similar to how I work day to day. Signals that may matter:
- Communicate clear plans to the interviewer and prompt the AI reasonably (tight prompts, explicit constraints, incremental diffs).
- Read error messages carefully and iterate accordingly.
- Explain what the AI is doing and why you accept or reject a suggested change. Don't copy & paste with no explanations.
- Stay in control: you set the plan, and the AI accelerates execution.

## 4. ML Coding

Okay, so ML coding will be right in our alley, right? That'd be the funniest thing I've heard 🤣🤣🤣. I like OOP 10x more over ML coding.

If I ask my ML engineer friends, 9 out 10 will say they love machine learning, but it will scare the hell out of all to write PyTorch code without inheriting from your team's model class or using Cursor. In ML coding interviews, however, you're expected to write PyTorch as fluently as regular Python. Even if you can Google, you have no time. 

First, be really fluent in PyTorch. Read Raschka's [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) and play with [TensorGym](https://tensorgym.com/) to learn syntax. For a more comprehensive tutorial, go over [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/). 

Next, be really familiar with Transformer architectures as well as training and inference loops. Read Sebastian Raschka's [Build a Large Language Model](https://github.com/rasbt/LLMs-from-scratch) and reproduce all code on your own. Watch Andrej Karpathy's [GPT-2 video](https://www.youtube.com/watch?v=kCc8FmEb1nY). If you have enough time on your hands, follow along with Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and read some of his repos (e.g., [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)). Also be familiar with common optimization techniques for training and inference (e.g., [Flash Attention](https://lubits.ch/flash/), [LoRA](https://lightning.ai/lightning-ai/environments/code-lora-from-scratch?section=featured), [KV cache](https://huggingface.co/blog/kv-cache)) and how to implement them from scratch. 

To test your understanding, solve LC-style ML problems on [Deep-ML](https://www.deep-ml.com/problems). You probably have no time to practice everything, but do go over common model architectures (e.g., KNN, K-Means, linear regression, logistic regression, MLP, CNN, RNN, causal self-attention, bidirectional self-attention, etc.), activation functions, optimizers (you may need to implement backpropagation and autograd from scratch instead of doing `loss.backward()`), evaluation metrics (e.g., nDCG, AUC), as well as training + inference techniques such as LoRA, beam search, KV cache, etc.. If you encounter problems outside of this list, maybe today isn't your day. You can also go over the [notebooks](https://udlbook.github.io/udlbook/) accompanying the *Understanding Deep Learning* book to cover all common ML concepts.

Last but not least, not all companies ask you to write PyTorch models. Some ask you to fit classical Scikit-learn models or use NumPy to implement from scratch. Don't be caught off the guard --- brush up with Educative's [Scikit-learn cheat sheet](https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods). To practice NumPy itself, go over the [numpy-100](https://github.com/rougier/numpy-100) repo and the [*From Python to Numpy*](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) tutorial.


## 5. ML Fundamentals

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

## 6. ML **Model** Design

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
1. **Problem framing**: understand what to build and how
   - **Clarify what system to build**: What's the UI and model entry point? How do users engage with recommendations? How do you want them to? What's the corpus size (i.e., how many items to recommend from)? What's the latency target? So on.
     - I never have a fixed list of questions. I just close my eyes for a second and let my intuitions speak about what's important for this system that could also impact the design. I sometimes play with the team or the company's product and imagine how I'd build it as a PM/EM/MLE.
   - **Understand why build it (business objectives)**: What are the desired actions you want your users to take immediately? What long term benefits do you want to bring to all parties in the system (e.g., users, the platform, and perhaps creators, advertisers, sellers, etc.)? What don't you wanna see? These tell us the {short-term vs. long-term, positive vs. negative} metrics you can use to measure the success of your system.
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
        - **Model architectures**: while Generative Recommendation is being hyped everywhere (and I {{< backlink "generative_recommendation" "wrote" >}} about it with great curiosities), most companies still use DLRM-style models with feature processing (normalize numeric features and pass them through an MLP; look up embedding features and project them to desired dimensions; pass sequence features to a Transformer module 👉 concatenate outputs), feature interaction (use MLP/CNN/RNN, attention, explicit cross networks, or throw a kitchen sink at the problem like [DHEN](https://arxiv.org/abs/2203.11014)), and feature transformation layers (e.g., expert sub-networks each process features in its own way and gating networks leverage expert outputs for each task).
        - **Objectives**: most modern L2 ranking models predict multiple `pAction` that matter to the system; for each prediction, we can use pointwise loss, pairwise loss, or listwise loss for training 👉 for ads ranking, pointwise loss is most common since it's easy to calibrate; search often uses pairwise or listwise loss to optimize for order.
4. **Deep dives**: discuss glossed-over key details, such as how to scale the system (e.g., vertical scaling, sharding, caching, replication + load balancing), how to handle positional bias, cold start, data drifts, and so on. Usually the interviewer will prompt you.

Above is just a skeleton. I find ML model design interview outcomes to be highly dependent on the interviewer. If the interviewer doesn't work on RecSys or is inexperienced, the experience can be excruciating --- the interviewer may over-index on structure and question your choices or explanations when they come from actual practice that the interviewer isn't familiar with. An experienced interviewer, by contrast, will let you skip unimportant parts and dive into the interesting components, asking how you handle tricky situations in day-to-day work or how the industry typically approaches them. Those conversations are lovely. I've only run into the former type once or twice; in those cases, I know I won't join the team.

If you're short on time, below is an essential list of papers to read:
1. **DLRM vs. Generative Recommendation paradigms**: [*Actions Speak Louder than Words*](https://arxiv.org/abs/2402.17152) 👉 like HSTU or not, this is the best paper to understand the general components in any deep learning recommendation models (DLRM) and how they differ from language models or so-called "generative recommenders" (GR). I also wrote a {{< backlink "generative_recommendation" "blogpost" >}} on this topic. Feel free to check it out.
2. **Feature extractions**: Maybe it's because of what I wrote on my resume, I find most of my interviewers were not super interested in how "normal" features are processed (e.g., normalizing numeric features and passing outputs to an MLP, looking up embeddings for categorical or ID features and projecting outputs, looking up and them pooling embeddings for an array of categorical or ID features), but instead rushed me through those features so I could elaborate on how user sequences are processed by a Transformer encoder or decoder. See my {{< backlink "seq_user_modeling" "blogpost" >}} on sequence modeling.
3. **Feature interactions**: When DLRM first got traction in RecSys, feature interactions were the major selling point --- while we may need to hand-engineer restricted-order feature crossing for "traditional models" (e.g., logistic regression), we can let deep learning models automatically learn higher-order feature interactions, either in a bit-wise (each feature dimension is an interaction unit) or a field-wise (each embedding is an interaction unit) fashion. Meta researchers found different interaction mechanisms capture different information and proposed [DHEN](https://arxiv.org/abs/2203.11014), which throws a kitchen sink at the feature interaction problem. Read this paper and cited papers to learn different interaction methods (e.g., MLP: [Deep & Wide](https://arxiv.org/abs/1606.07792); explicit cross networks: [DCN](https://arxiv.org/pdf/1708.05123), [DCNv2](https://arxiv.org/abs/2008.13535); factorization machines: [DeepFM](https://arxiv.org/abs/1703.04247), [xDeepFM](https://arxiv.org/abs/1803.05170), [Wukong](https://arxiv.org/abs/2403.02545); attention: [AutoInt](https://arxiv.org/abs/1810.11921), [MaskNet](https://arxiv.org/abs/2102.07619)). [RankMixer](https://arxiv.org/abs/2507.15551) and [OneTrans](https://arxiv.org/abs/2510.26104) are influential new papers on this topic in larger-scale RecSys.
4. **Feature transformations**: Another selling point of DLRM is its ability to leverage shared feature representations to predict different tasks, such as clicks, conversions, watch time, and other engagement outcomes one cares about. Read the seminal [MMoE](https://arxiv.org/abs/2311.09580) paper and its predecessor [PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236) on such multi-task learning.
5. **Practical concerns**: For most RecSys MLEs, unless you work on these problems, you're definitely aware of their existence but may not speak intelligently about solutions --- such as cold start (recommend new items or recommend for new users), positional bias (learn inherent item relevance, debiasing the influence from item positions), diversity (dedupe & re-rank results by topic, creator, or other criteria), or value models (if ranking by multiple objectives, find the "best" weight of each). Read one or two paper on each topic (e.g., cold start: Kuaishou [1](https://dl.acm.org/doi/10.1145/3640457.3688098) & [2](https://dl.acm.org/doi/10.1145/3701716.3715205); positional bias: [Google](https://research.google/pubs/position-bias-estimation-for-unbiased-learning-to-rank-in-personal-search/); diversity: [Kuaishou](https://arxiv.org/abs/2206.05020); value models: [Pinterest](https://arxiv.org/abs/2509.05292)), so you can say while you're not an expert, you know some solutions.

## 7. ML **Infra** Design

I have colleagues who are both strong ML engineers and ML infra engineers. They have one thing in common: 15+ years of experience, having started as backend/infra engineers before ML was cool, and then growing into ML leaders. Most of my peers and I, however, started our careers as baby ML engineers trying only to move model metrics, without deep low-level ML infra knowledge. Most of the time, you can avoid ML infra interviews --- but a handful of places (e.g., DoorDash, Reddit, Netflix, Snap) specifically ask MLE candidates to design the *infra* behind ML systems. It's funny that I've interviewed with them all and somehow never managed to escape this round.

I've summarized my ML infra interview preparation in this {{< backlink "ml_infra_interviews" "blogpost" >}}. It's pretty long, so I won't repeat its content here in the interest of space. Do check it out! Below are key lessons I've learned:

1. The best way to understand ML infra is to start from the infra teams you collaborate with and learn what they do (e.g., via design docs, [books](https://www.amazon.com/Distributed-Machine-Learning-Patterns-Yuan/dp/1617299022), and engineering blogs; see [resources](https://www.yuan-meng.com/posts/ml_infra_interviews/#references)) 
   - **Data infra**: e.g., how features are defined, computed, stored, and updated (batch vs. streaming); how engagement events are logged; how training data (features + labels) is generated via forward logging or backfill.
   - **Training infra**: e.g., how training is distributed across GPUs; checkpointing; failure recovery; continuous retraining + validation; model publishing and rollout/rollback.
   - **Serving infra**: e.g., how to fetch request (user + context) vs. document vs. cross features; when and how to batch requests; reducing end-to-end latency (caching, sharding, load balancing); real-time feature updates; pagination for large responses; engagement event logging.
2. Brush up on distributed system knowledge, but don't dwell on it. ML infra is a special case of distributed systems, but you usually don't need to go deep on things like rate limiters or post creation/updates. The focus is the infra around **ML**.
   - **NeetCode**: watch [System Design for Beginners](https://neetcode.io/courses/system-design-for-beginners/0) and checkout sections in [System Design Interview](https://neetcode.io/courses/system-design-interview/9) relevant to ML systems, such as KV stores and distributed message queues. 
   - **Hello Interview**: go through [System Design in a Hurry](https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction) as well as core concepts, patterns, key technologies, and advanced topics like time-series databases (think feature stores).
   - [**DDIA book**](https://dataintensive.net/): skim Chapters 1-11 if you have time.
3. To practice, design feature stores, training data generation, distributed training, and retrieval & ranking (organic + ads), etc.. Spend 2–3 days per design to think through all details. Below is the table of contents in a toy ads ranking system design I wrote.
    ```markdown
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

## 8. Behavior Interview
99% of the time, behavior questions come from the list below:

<details class="fold">
<summary>79 behavioral question by signal</summary>

<table>
<thead><tr><th>signal</th><th>question</th></tr></thead>
<tbody>
<tr><td>problem solving</td><td>the most challenging / successful project you've delivered</td></tr>
<tr><td></td><td>how you iterated on a model (in a single or multiple launches)</td></tr>
<tr><td></td><td>how you work with ambiguous requirements</td></tr>
<tr><td></td><td>how you iterated on a model (in a single or multiple launches)</td></tr>
<tr><td></td><td>a time when you optimized a process within/beyond your team</td></tr>
<tr><td></td><td>a time when you found an innovative solution to a problem</td></tr>
<tr><td></td><td>a time when you discovered an overlooked issue on your team</td></tr>
<tr><td></td><td>tell me about a time when you faced a challenging situation at work</td></tr>
<tr><td></td><td>a time when you understood and improved a product</td></tr>
<tr><td></td><td>a time when you had to pivot (proactively or being asked to)</td></tr>
<tr><td></td><td>a time when you deep dived into data to resolve a problem or made a decision</td></tr>

<tr><td>teamwork</td><td>how you resolve conflicts with teammates or xfn</td></tr>
<tr><td></td><td>a time when you had to work with people outside of your team</td></tr>
<tr><td></td><td>a time when you disagreed with your manager and persuaded them</td></tr>
<tr><td></td><td>a time when your team disagreed with you on your idea</td></tr>
<tr><td></td><td>a time when you worked with a difficult / non-collaborative teammate</td></tr>
<tr><td></td><td>how you get buy-in from peers/xfn/leadership</td></tr>
<tr><td></td><td>how you collaborate + earn trust from with teammates / xfn</td></tr>
<tr><td></td><td>a time when you escalated to manager/skip</td></tr>
<tr><td></td><td>a time when you helped an underperforming teammate</td></tr>
<tr><td></td><td>how you give and share credit</td></tr>
<tr><td></td><td>what to do if team is not bonding well/td></tr>
<tr><td></td><td>how do you handle disagreement with majority decision</td></tr>

<tr><td>communication</td><td>how you communicate technical ideas to non-tech audience/td></tr>
<tr><td></td><td>how you give constructive feedback</td></tr>
<tr><td></td><td>how you receive constructive feedback</td></tr>
<tr><td></td><td>how you get visibility for your work</td></tr>

<tr><td>time management</td><td>a time when you met a deadline</td></tr>
<tr><td></td><td>a time when you missed a deadline</td></tr>
<tr><td></td><td>a time when you felt overwhelmed / stressed</td></tr>
<tr><td></td><td>how you deal with conflicting priorities</td></tr>
<tr><td></td><td>how you balance responsiveness vs. focus time</td></tr>
<tr><td></td><td>a time when you must deliver an imperfect result</td></tr>

<tr><td>decision making</td><td>the hardest tech/business trade-off you had to make</td></tr>
<tr><td></td><td>a time when you made a difficult decision to give up on a project</td></tr>
<tr><td></td><td>a time when you took a (calculated) risk and succeeded</td></tr>
<tr><td></td><td>a time when you took a (calculated) risk and failed</td></tr>
<tr><td></td><td>a time when you made a wrong decision and learned from it</td></tr>
<tr><td></td><td>how you define impact and success of a project</td></tr>
<tr><td></td><td>a time when you made a decision in limited time</td></tr>
<tr><td></td><td>a time when you made a decision with incomplete information</td></tr>
<tr><td></td><td>a time when you made a decision with many sources of information</td></tr>
<tr><td></td><td>a time when you made a decision without consulting your manager</td></tr>
<tr><td></td><td>a time you refused to compromise your standards</td></tr>

<tr><td>initiative</td><td>a time when you unblocked yourself</td></tr>
<tr><td></td><td>a time when you unblocked others</td></tr>
<tr><td></td><td>a project initiated by you</td></tr>
<tr><td></td><td>a project you had to push hard for</td></tr>
<tr><td></td><td>a time when you went above and beyond your duty</td></tr>
<tr><td></td><td>a time when you worked on something before mgr approval</td></tr>

<tr><td>achievement</td><td>a time when you set a goal and achieved it</td></tr>
<tr><td></td><td>a time when you went above and beyond expectations</td></tr>
<tr><td></td><td>the biggest accomplishment in your career</td></tr>
<tr><td></td><td>your biggest strengths</td></tr>

<tr><td>growth</td><td>the biggest regret in your career</td></tr>
<tr><td></td><td>a time when you had to persevere for several months</td></tr>
<tr><td></td><td>a project you would do differently</td></tr>
<tr><td></td><td>a time when you made a big mistake (reflection → do better)</td></tr>
<tr><td></td><td>your biggest weaknesses</td></tr>
<tr><td></td><td>a time when you owned a task outside of your expertise</td></tr>
<tr><td></td><td>how you find mentors and network within/beyond your team</td></tr>
<tr><td></td><td>a time when you struggled initially and turned it around</td></tr>
<tr><td></td><td>how you find time to learn at and outside of work</td></tr>
<tr><td></td><td>career aspirations + motivation to join xyz company</td></tr>

<tr><td>leadership</td><td>how you mentor junior engineers</td></tr>
<tr><td></td><td>a time when you delegated a project</td></tr>
<tr><td></td><td>a time you advocated for someone on your team</td></tr>
<tr><td></td><td>a time when you improved company/team culture</td></tr>
<tr><td></td><td>a time when you took responsibilities for a failure</td></tr>
<tr><td></td><td>a time when you set roadmap (for a project or a team)</td></tr>
<tr><td></td><td>what you value in a leader + what kind of leader you aspire to be</td></tr>
<tr><td></td><td>a time when you drove alignment</td></tr>
<tr><td></td><td>a time you made a suggestion to improve the team's outcomes</td></tr>
<tr><td></td><td>positive leadership style you liked + influenced your work style</td></tr>

<tr><td>adaptability</td><td>a time when you incurred a tech debt to meet deadline</td></tr>
<tr><td></td><td>a time when you had to change the project scope</td></tr>
<tr><td></td><td>a time you had to make a last-minute change</td></tr>
<tr><td></td><td>a time when you delivered under time pressure / lack of resources</td></tr>
<tr><td></td><td>a time when you received an unreasonable ask or project requirements</td></tr>
<tr><td></td><td>a time when a project kept getting de-prioritized</td></tr>

</tbody>
</table>

</details>


Then it boils down to how you pick and tell your stories. While companies ask the same questions, they value different qualities in an engineer. For instance, moving fast and breaking things (e.g., incurring a tech to meet a short-term goal) may be a vice at Google, which values doing things "the right way", it's a virtue at companies like DoorDash or startups where scrappiness is the key to survival (your own and that of your company). Friends have told me that studying Amazon's leadership principles is the best way to prepare for behavior interviews, even if you don't interview with Amazon. But for me personally, I'm more at ease with the [Googlyness](https://jeffhsipe.medium.com/understanding-googelyness-4d61a70ada95) interview: instead of being grilled on reciting stories that map to rigid qualities, you're valued for being emotionally intelligent, ethical, kind, and curious.

To quickly prepare for each question, I usually start with a rough story --- just a few sentences I hastily jot down about an experience I'm proud of or learned from, throw in the company values, and prompt GPT to turn it into a SAIL (situation, action, impact, learning) story.

```markdown
## Behavioral Interview Story Expansion Prompt

### Goal
Turn a rough, real story (even a few sentences) into a clear, interview-ready
behavioral answer that is explicitly aligned with a company's values.

---

### Instructions

I'm preparing for behavioral interviews.

I will give you:
1. a **rough story** (unpolished, informal is fine)
2. a set of **company values / culture principles** (pasted text or a link)

Please expand the story into a **structured behavioral answer aligned with those values**.

Do **not** invent facts or exaggerate impact.
Your job is to organize, clarify, and align — not to embellish.

---

### Input 1 — Rough story
This story is intentionally rough and incomplete:
[PASTE YOUR STORY HERE]


### Input 2 — Company values
Here are the company's values or culture principles I want the story aligned with:
[PASTE VALUES HERE OR PROVIDE URL]

---

### Output Requirements

#### 1. Structured behavioral story
Rewrite the story using the following sections:

- **situation:** brief context; what was happening; responsibility, goal, or challenge I faced
- **action:** what I actually did  
  (focus on judgment, communication, trade-offs, and decisions)
- **impact:** concrete outcome or change  
  (metrics, behavior shift, decision made, unblocked work, etc.)
- **learning:** what I learned or how this changed my approach

**Style constraints**
- first-person
- natural, spoken tone
- concise (60–120 seconds spoken)
- honest and professional
- avoid buzzwords and filler phrases like "i learned a lot"
- do **not** add new facts — only clarify and organize what's already there

#### 2. Values alignment
Add a section titled exactly:
✅ why this is highly aligned with <company values>

In bullet points, explicitly map:
- a **specific action or decision**
- to a **specific value**

Examples of values to map against (use what's relevant):
- emotional intelligence
- collaboration
- ownership
- integrity
- comfort with ambiguity
- bias for action

Make the mapping concrete and specific (what I did → which value it shows).
---

### Reminder
The goal is **clarity and alignment**, not sounding impressive.

The final answer should still feel like something a real person would say
in an interview.
```

I created a Notion database, where each row is a question. Before interviews, I go over each question once and time myself on delivery.

## 9. Project Deep Dive

In phone screens, hiring manager chats, or design interviews, you may be asked to walk through a project you're most proud of. Some companies have a dedicated 45-minute or 1-hour round for project deep dives. If slides are allowed, 10–12 is a good amount.

There's no skill for this round. If you've led a project with great technical challenges (e.g., SOTA models, label scarcity, serving challenges) or cross-functional complexity (e.g., many stakeholders with divided opinions, hard trade-offs), this is your chance to shine. I find this round to be the deciding factor for hiring decisions. Coding interviews weed out weak candidates, but exactly no one gets hired because they did well on coding. Design interviews test your first-principles thinking, but you might never have had success in this domain yet. Project deep dives, by contrast, show exactly why you're the right person for this team --- e.g., because you bring the exact technical expertise the team needs and your leadership and decision-making style gets things done on this team or in this organization.

To prepare, I take a piece of paper and walk down the memory lane to detail everything that happened in the project. I then organize those pieces by collaboration complexity, ambiguity, and technical complexity, and tell my story in chapters: design, development, launch, learnings, and next steps. Check out my [post](https://www.yuan-meng.com/notes/project_complexity/) for more details.

# Appendices

## A: Company Selection
top pay, interesting work, modern stack, willing to offer me senior or plus title

## B: Prepare in a Hurry
xx

## C: Example System Designs
xx