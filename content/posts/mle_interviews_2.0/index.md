---
title: "MLE Interview 2.0: Research Engineering and Scary Rounds"
date: 2026-01-14
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

<!-- So for me in this moment, only a few {{< sidenote "companies" >}}In truth, Pinterest is one of the best companies in terms of ML talent density and certainly the very best in terms of immigration policies. As such, it's not really companies, but rather only a few teams, that can offer what I want. It's for those teams that I interview this time.{{< /sidenote >}} offer what I want: -->

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

<!-- Last year I said the variety of MLE interviews far exceeds that of other job families. Now I think if you only encounter these 6 rounds in a full-loop interview, count yourself lucky --- by today's standards, you've had an easy-peasy interview. Many ML and Research Engineering roles add extra rounds that most ML practitioners find genuinely hard.

I think of candidate selection as identifying the positive few from a universe of candidates and interview design as negative sampling. Most companies "only" need discriminability to find the best among hundreds, so in-batch negatives (with batch sizes of a few hundred) suffice. Frontier AI labs and top-paying companies (e.g., Netflix, Snap, Roblox, Databricks) can afford to search for the best among thousands or more --- so they have to introduce "hard negatives", designing interviews to be harder than most great ML engineers can pass. -->

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

Increasingly more companies now ask for 2–3 references from recent managers, teammates, or internal referrals. I remember this was already the case when I was interviewing for my first job in grad school --- companies like Figma and Roblox have long required reference checks. Nowadays, all frontier AI labs ask for them.

Most companies I interviewed with have a team match stage. Last year, it was pretty quick on my end: I was either matched within a day, or the hiring manager who I met during onsite extended a direct offer. This year, however, passing an onsite is still far from an offer: dozens of candidates who have cleared the interview need to talk to a handful of managers with openings. Your years of experience, project complexity, and interview performance are evaluated all over again.

# Preparing for Each Round

## 1. LC-Style Coding

### Don't Be Obsessed with Company Tags

Company tags on LC are added by candidates after seeing these problems in interviews. Some companies rarely update their question bank so their tags cover most of what you'll see in actual interviews. However, you can't plan on that, since occasional updates do happen at any company. Moreover, companies like Google purposefully avoid asking leaked problems and you can't prepare for Apple or Netflix where each team has the freedom to create their own problems. 

As hard as it may sound, you gotta let go of the idea that `solving company tags == solving coding interviews` and actually get good at problem solving and coding up the solution in your mind. 


First thing first, don't theorize problem solving until you've solved enough problems. I always recommend taking the [beginner](https://neetcode.io/courses/dsa-for-beginners/0) and [advanced](https://neetcode.io/courses/advanced-algorithms/0) NeetCode courses first and practicing [problems](https://neetcode.io/practice/practice) curated by NeetCode. I think [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) strikes a good balance between quality and quantity. There's no shortcut. If you have time, you can participate in LC [Weekly Contest](https://leetcode.com/contest/) to rehearse the pressure of solving a random Hard problem under pressure (kinda necessary for Google). 

Then, don't memorize concrete data structures you've used for a problem --- figure that out on the fly. For instance, if a problem gives you some dependencies and asks you to find a valid ordering that satisfies them, it's likely a topological sort problem. Usually, LC problems provide dependencies in the form of edge lists or adjacency lists. But what if a new problem provides dependencies as links or pointers? Can you still recognize the problem? Maybe. Can you still construct the adjacency list and the indegree map correctly? Maybe not. You might struggle between recalling old solutions and reasoning about the problem at hand, run out of time, or keep writing bugs.

The implication: when practicing LC, focus on the problem *at hand*. Given the task (e.g., finding an order under dependency constraints), what's a good/likely algorithm? Then use what you're given right here, right now to implement that algorithm, rather than trying to recall concrete bits and pieces from past solutions. If you do write a bug, refrain from asking GPT or looking at the solution right away. Because in a real interview, this is exactly when you should debug on your own --- by printing suspicious parts, or by thinking from first principles.

That said, don't practice randomly. For instance, while you don't need to (and probably never can) go over all Google tags, you can still research what types of problems are most popular at Google (e.g., [Reddit](https://www.reddit.com/r/leetcode/comments/1izv4ln/how_to_actually_prepare_for_google_via_leetcode/) says DP, graphs, and "fancy string problems") and focus on those. Track which algorithms trip you up the most and practice accordingly. For me, it's definitely advanced graphs, linked lists and trees (I hate pointers), bottom-up DP, greedy, and string evaluation (e.g., Basic Calculator I, II, III). I also accept that as an ML engineer, there are so many other rounds that I can't be perfect at coding. If I run into a red-black tree problem, I'll accept today is not my day.

{{< figure src="https://www.dropbox.com/scl/fi/d6bchbb2xewuycaynckkv/Screenshot-2026-01-11-at-10.31.08-AM.png?rlkey=a3fi4vfnhqvy55da0t2nf5612&st=hb43kyx4&raw=1" width="1800" caption="My practice plan for Google coding interviews.">}}

<!-- In my experience, if you don't have intuition 1-2 minutes after the interviewer pastes the prompt and test cases, you won't get a strong yes, because you likely won't have time to write a clean solution and handle follow-ups. In this market, you want to shoot for "strong yes" in most rounds for an offer. You can't expect to reason everything from first principles, yet you can't rely on having seen every problem before.

The more I interview, the more I see solving LC problems as cracking an oyster shell. A good solution should feel almost effortless --- one touch and a twist, and the shell opens. If you find yourself grinding so hard that you're smashing the shell, you're 100% doing it wrong.

The implication: when practicing LC, stop grinding if you find yourself writing a long-winded solution that's going nowhere. Don't build muscle memory for heavy labor --- it won't ever serve you in actual interviews. Instead, build the muscle memory for pausing to examine patterns and finding the "a-ha" moment that cracks the shell open.
 
Sometimes you know you're close but can't land on that "a-ha" moment. In such cases during practice, I tell ChatGPT about my "cloudy" intuitions, hoping it can shed light on the opening. Take [Car Fleet](https://leetcode.com/problems/car-fleet/description/) for example: I felt it could be solved with a monotonic stack ("monostack") but didn't know why or how. So I asked ChatGPT:

> I consider using a monostack but don't know why. My feeling might be stirred up by cars travelling unidirectionally, or the no passing rule.

To which it answered:
> This is a great instinct --- and you're not imagining it.
> Your brain picked up three real signals:
> 1. Unidirectional motion
> 2. No passing = irreversible merging
> 3. Local interactions resolve global structure

Using more hints that I asked for, I solved this problem and won't forget the solution. That said, if you have the right idea but are just fighting bugs, don't ask ChatGPT --- debug it yourself! Add print statements. Write small test cases. In a real interview, finding and fixing bugs is a hurdle you must overcome quickly and independently.

Of course, you can't build intuition --- however vague --- without solving enough LC problems. When friends ask me for coding prep advice, I always recommend taking the two NeetCode courses ([beginner](https://neetcode.io/courses/dsa-for-beginners/0), [advanced](https://neetcode.io/courses/advanced-algorithms/0)) first and then practicing [problems](https://neetcode.io/practice/practice) curated by NeetCode. I think [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) strikes a good balance between quality and quantity. Finish these before you are in any position at all to speak about intuitions or strategies. If you have time, you can also participate in LC [Weekly Contest](https://leetcode.com/contest/) to rehearse the pressure of solving a random Hard problem under pressure (kinda necessary for Google).  -->

### Connect LC Problems to Scalable Systems

<!-- Unless you're a competitive programmer, solving a new or a hard problem in interviews is still challenging. Time goes by quickly if you fumble around. To up a notch on general coding, I've found two kinds of preparation particularly useful for me: one tangible, one mental.

The tangible suggestion, as mentioned just now, is to finish [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) so you can get a solid grasp on common algorithms and patterns. Then, work through company-tagged problems on LC. For companies like Google, Netflix, or Apple, tags are only a rough reference --- interviewers have lots of freedom in question selection. But for many other companies, the question pool is fairly constrained, and LC tags cover a large fraction of what you'll actually see.
 -->

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
Coming soon...

## 3. ML Coding

It scares the hell out of most ML engineers to write PyTorch code without inheriting from your team's model class or using Cursor. In ML coding interviews, however, you're expected to write PyTorch as fluently as regular Python. Even if you can Google, you have no time. 

First, be really fluent in PyTorch. Read Raschka's [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) and play with [TensorGym](https://tensorgym.com/) to learn syntax. For a more comprehensive tutorial, go over [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/). 

Next, be really familiar with Transformer architectures as well as training and inference loops. Read Sebastian Raschka's [Build a Large Language Model](https://github.com/rasbt/LLMs-from-scratch) and reproduce all code on your own. Watch Andrej Karpathy's [GPT-2 video](https://www.youtube.com/watch?v=kCc8FmEb1nY). If you have enough time on your hands, follow along with Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and read some of his repos (e.g., [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)). Also be familiar with common optimization techniques for training and inference (e.g., [Flash Attention](https://lubits.ch/flash/), [LoRA](https://lightning.ai/lightning-ai/environments/code-lora-from-scratch?section=featured), [KV cache](https://huggingface.co/blog/kv-cache)) and how to implement them from scratch. 

To test your understanding, solve LC-style ML problems on [Deep-ML](https://www.deep-ml.com/problems).

Last but not least, not all companies ask you to write PyTorch models. Some ask you to fit classical Scikit-learn models. Don't be caught off the guard --- brush up with Educative's [Scikit-learn cheat sheet](https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods).


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

More recently, I like [Understanding Deep Learning](https://udlbook.github.io/udlbook/). It's less mathy and more straight to the point. At least read Chapters 1–9, 11, and 12.


## 5. ML **Model** Design
Coming soon...

(Note: depth won't be too hard --- once you've nailed a few common designs, focus on a tight & structured delivery)

## 6. ML **Infra** Design
Coming soon...


## 7. Behavior Interview
Coming soon...

## 8. Project Deep Dive
Coming soon...
