---
title: "ML Interview 2.0: Research Engineering and Scary Rounds"
date: 2025-12-23
math: true
categories: ["career", "machine learning", "interview"]
toc: true
---

# Why Do You Wanna Leave?
Okay, before we talk about interviews, ask yourself (and every recruiter and hiring manager will ask you): why do you want to leave?

Under normal circumstances, new hires don't work on high-impact projects immediately, and many don't survive long enough to ever take on one. If you're already in a rare position to lead high-impact projects, you'll likely find yourself in a worse situation at the new company.

In hindsight, I was lucky to take on a high-visibility, top-down project shortly after joining Pinterest. I won't expect such luck at a future company. Only a few companies can offer sufficient upside for leaving:

- Pay: higher level and/or 30%+ higher compensation
- Domain: the field and tech stacks are future-oriented
  - Type 1: applied AI engineering at a frontier lab
  - Type 2: foundation ranking models at one of the few companies that need and can afford to train them
- WLB: 55–65 hrs/week is OK, but 80+ hrs/week is too much
- Stability: hopefully the company will still exist in 3–4 years

Think clearly about your goals before interviewing. To prepare well to land a worthwhile offer in this market, you inevitably lose some productivity at work. Only interview when there's an undeniable upside. And once you start, commit to finishing within 2 months with full determination. Otherwise, it may be much better to stay put.

# Recap: "Standard" MLE Interviews

Now, onto interview prep. In 2025, most FAANG and similar companies still have the same rounds of MLE interviews, typically coding, ML system design, ML fundamentals, and behavior. Some companies have made slight changes. For instance, Meta added an OA round for most candidates as well as AI-assisted coding. Companies like Google, LinkedIn, and many startups (e.g., xAI, Perplexity) now conduct in-person onsite. I wrote about how to prepare for "standard" ML interviews in my 2024 {{< backlink "mle_interviews" "blogpost" >}} --- see a recap below.

1. **Coding**: Medium–to-Hard LC problems, object-oriented programming (often multi-level CodeSignal), and ML coding (e.g., debugging or building and training a PyTorch/NumPy model)
2. **ML system design**: design a {ranking, query understanding, content understanding, NLP, trust & safety} ML system, focusing on data, features, labels, and model architectures
3. **Project deep dive**: walk through a project you're most proud of
4. **ML fundamentals**: rapid-fire questions on machine learning foundations, such as common network architectures, optimization routines, loss functions, activation functions, regularization, etc.
5. **Behavior and leadership**: a time when you demonstrated problem solving, team work, conflict resolution, time management, leadership, adaptability, growth, etc. beyond your level
6. **Domain knowledge**: woven into all rounds, explaining how you shipped past projects or tackle current problems using expertise

Last year I said the variety of MLE interviews far exceeds that of other job families. Now I think if you only encounter these 6 rounds in a full-loop interview, count yourself lucky --- by today's standards, you've had an easy-peasy interview. Many ML and Research Engineering roles add extra rounds that most ML practitioners find genuinely hard.

I like to think of candidate selection as discriminating the positive few from a universe of negatives and interview design as negative sampling. Most companies "only" need discriminability to find the best among hundreds, so in-batch negatives (with batch sizes of a few hundred) suffice. Frontier AI labs and top-paying companies (e.g., Netflix, Snap, Roblox) can afford to search for the best among thousands or more --- so they have to introduce "hard negatives", designing interviews to be harder than most ML engineers can pass.

# MLE Interview 2.0: What Has Changed?

## Additional Rounds
Many companies start to include more challenging rounds:

1. **LLM ML coding**: A few years ago, you might be asked to implement a simple model from scratch (e.g., KNN, K-means, decision trees, logistic regression, linear regression, MLP) and fit it on toy data. Today, you're likely to debug or implement LLM training or inference code --- Transformer encoders/decoders, KV cache, LoRA, and more. In especially brutal interviews, you may even be asked to implement [autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). The ideal candidate is Andrej Karpathy 2.0: chatting and writing hundreds of lines of code in an hour while vividly explaining LLM fundamentals.
2. **ML infra design**: I'm **not** talking about ML system design that focuses on modeling, where you can get by with "scaling considerations" by briefly mentioning sharding, caching, and so on. I'm talking about ML **infra** design that doesn't focus on modeling at all, but instead asks in great detail about the infra around ML systems, such as feature stores, distributed training, and online serving. Such interviews are rare, but they happen to be a required round at companies like Netflix, Snap, and Reddit that offer great compensations matched only by frontier labs or Meta and Databricks. In some of these high-performing organizations, teams are lean, roles are blurred, and ML engineers are expected to unblock themselves across the entire stack. I recently wrote a {{< backlink "ml_infra_interviews" "blogpost" >}} on how to prepare for this rare round.
3. **Research presentation**: In traditional project deep dives, you verbally walk through one or two representative projects. Some Research Engineer roles require a job talk style presentation on your past work. You make ~10 slides, go over technical details, and "defend" your body of work like a PhD candidate would.

If you think these rounds are daunting, so does everyone else. Tech interviews evolve from time to time. Back in grad school, I remember watching Mayuko's [YouTube video](https://www.youtube.com/watch?v=e2Y-rhTlHHI) explaining how tech interviews were once all about domain knowledge; then Microsoft popularized domain-agnostic data structures and algorithms questions that we see today to select raw talents, regardless of their background. For a while, SWE candidates found those interviews intimidating, until LeetCode came along and gave everyone a sure fire way to prepare.

Today, new interview types (e.g., ML coding, ML infra design, research presentations) feel daunting for the same reason: no one has figured out an effective way to prepare for them. We're the pioneers now.

## Higher Bar

I used to have an accurate "feel" for when an offer was coming. Last year, if I solved all the coding problems (even with minor flaws), presented a complete system design, and had an enjoyable conversation with the hiring manager, an offer would usually come. 

That reality has changed. Our culture tries to steer clear from the idea of "perfection", seeing it as unattainable or mentally taxing. Sadly, perfection has become the new bar at companies most want to join. 

1. **Coding**: In a 45-minute LeetCode-style interview with one Hard problem, the ideal outcome is to write a bug-free solution in 15–20 minutes, come up with sharp test cases, pass them, and then discuss optimizations. If there are follow-ups, you're expected to provide extendable solutions. In a one-hour OOP-style CodeSignal interview, most people can't even type fast enough to finish all 4 levels; you should aim to finish in ~40 minutes so you have time to discuss how to scale the toy system (e.g., it's usually a database, message queue, or workflow) to production. In ML coding, even reading the full training and inference loops can overwhelm most candidates --- but you're expected to complete the implementation and improve model performance in an hour.
2. **ML system design**: By now, everyone knows how to sketch a standard ranking pipeline --- L1 (candidate generation) 👉 L1 (first-stage ranking) 👉 L2 (second-stage ranking) 👉 L3 (call it re-ranking, value models, or "special effects"). If you sound like everyone else, that won't cut it anymore. You need full control of the conversation: use the ranking funnel as scaffolding, but show insights that only a RecSys expert would have. Bring ideas and solutions from your work (e.g., you've worked on lifelong sequence models and they are actually a good choice here) and industry frontiers (e.g., this year's standouts are perhaps "One" series from Kuaishou or {{< backlink "generative_recommendation" "generative recommendation" >}} in general).
3. **Project deep dive**: I've got offers because the team needed someone to work on X, and I happened to have worked on X. In one case, my first interviewer was so interested in X that they skipped the coding question and those in the next few rounds ditched the agenda to ask more about X. If you work on models with broad industry applications and your team is ahead of the industry, you have a natural advantage over very well-prepared candidates who don't have such experience. I think this is why you need to choose your current projects or next moves carefully --- as an ML engineer, try to choose a company that has the same or a more advanced ML stack as your current one; only move to a company with a {{< sidenote "dated" >}}It doesn't matter at all, for instance, if a ranking team currently uses DCNv2, RankMixer, or some other architectures for feature interaction, because details can change in a few projects, including yours. However, if a company doesn't use DL, has no plan to support DL, and won't benefit from DL, then think twice before joining, however much you like their product. In a few years when you interview again, perhaps for a frontier AI lab or a more mature company, you can speak to scope but won't capture sufficient interests in deep dives, research presentations, or team matches. Your future career may be limited to designer-driven, pre-IPO companies.{{< /sidenote >}} ML stack if you get a once-in-a-lifetime level bump (e.g., +2) or a package that keeps you happy for five years. Otherwise, you're cashing out too early and losing your advantage in 2-3 years. That said, you can still fail this round if you talk about a complex project as if it were trivial. I wrote a [post](https://www.yuan-meng.com/notes/project_complexity/) on how to present complex projects in ways that do them justice.
4. **ML fundamentals**: Most of the time, you'll get 3–5 rapid-fire ML fundamentals questions in a phone screen, often before coding. Many candidates are confused when they fail, remembering they've solved the coding question perfectly, only to forget that they made fundamental mistakes on ML fundamentals. One or two wrong answers can be enough for rejection. And if those 5 minutes already feels exposing, know that some companies have one or two 45-minute onsite rounds that keep asking ML fundamentals to test the limit of your deep learning knowledge. 
5. **Behavior and leadership**: It's hard to reflect on your career while you're still living it. If you claim to know perfectly well how to expand scope, manage priorities, collaborate with difficult teammates, and right all the wrongs, you're essentially saying "everyone is drunk while I alone am sober" (众人皆醉我独醒). That can't be true. You need repeated, honest conversations with friends and mentors to see where you can grow. Beyond that, you need to be that mentor and friend --- support junior engineers & peers and lead teams through difficult moments --- to show leadership across the board. Just like internal promotions, getting a Senior+ offer often requires showing Staff-level qualities.

## Longer Process

Increasingly more companies now ask for 2–3 references from recent managers, teammates, or internal referrals. I remember this was already the case when I was interviewing for my first job in grad school --- companies like Figma and Roblox have long required reference checks. Nowadays, all frontier AI labs ask for them.

Most companies I interviewed with have a team match stage. Last year, it was pretty quick on my end: I was either matched within a day, or the hiring manager who I met during onsite extended a direct offer. This year, however, passing an onsite is still far from an offer: dozens of candidates who have cleared the interview need to talk to a handful of managers with openings. Your years of experience, project complexity, and interview performance are evaluated all over again.


# Up the Ante on Interview Prep

I don't intend to rewrite my {{< backlink "mle_interviews" "previous post" >}}. For standard MLE interviews, please read my old post. I'll add new insights on top.

## LC-Style Coding

### Strategy: Crack the Oyster Shell

In my experience, if you don't have intuition 1-2 minutes after the interviewer pastes the prompt and test cases, you won't get a strong yes, because you likely won't have time to write a clean solution and handle follow-ups. In this market, you want to shoot for "strong yes" in most rounds for an offer. You can't expect to reason everything from first principles, yet you can't rely on having seen every problem before.

The more I interview, the more I see solving LC problems as cracking an oyster shell. A good solution should feel almost effortless --- one touch and a twist, and the shell opens. If you find yourself grinding so hard that you're smashing the shell, you're 100% doing it wrong.

The implication: when practicing LC, stop grinding if you find yourself writing a long-winded solution going nowhere. Don't build muscle memory for labor --- it won't serve you well in interviews. Instead, practice searching for the "a-ha" moment that cracks the shell open.

Sometimes you know you're close but can't land on that "a-ha" moment. In such cases during practice, I tell ChatGPT about my "cloudy" intuitions, hoping it can shed light on the opening. Take [Car Fleet](https://leetcode.com/problems/car-fleet/description/) for example: I felt it could be solved with a monotonic stack ("monostack") but didn't know why or how. So I asked ChatGPT:

> I consider using a monostack but don't know why. My feeling might be stirred up by cars travelling unidirectionally, or the no passing rule.

To which it answered:
> This is a great instinct --- and you're not imagining it.
> Your brain picked up three real signals:
> 1. Unidirectional motion
> 2. No passing = irreversible merging
> 3. Local interactions resolve global structure

Using more hints that I asked for, I solved this problem and won't forget the solution. That said, if you have the right idea but are just fighting bugs, don't ask ChatGPT --- debug it yourself! Add print statements. Write small test cases. In a real interview, finding and fixing bugs is a hurdle you must overcome quickly and independently.

Of course, you can't build intuition --- however vague --- without solving enough LC problems. When friends ask me for coding prep advice, I always recommend NeetCode. I think [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) strikes a good balance between quality and quantity. Finish them before you and I are in any position at all to speak about intuitions or strategies.

### Mindset: Connect Problems to Scalable Systems

Unless you're a competitive programmer, solving a new or a hard problem in interviews is still challenging. Time goes by quickly as you fumble around. To up a notch on general coding, I've found two kinds of preparation particularly useful for me: one tangible, one mental.

The tangible suggestion, as mentioned just now, is to finish [NeetCode 250](https://neetcode.io/practice/practice/neetcode250) so you can get a solid grasp on common algorithms and patterns. Then, work through company-tagged problems on LC. For companies like Google, Netflix, or Apple, tags are only a rough reference --- interviewers have lots of freedom in question selection. But for many other companies, the question pool is fairly constrained, and LC tags cover a large fraction of what you'll actually see.

The mental realization mattered just as much for me. I was re-reading the classic [MapReduce](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) paper when it dawned on me that many --- or dare I say, most? --- LC problems stem from web-scale data processing. Google had massive amounts of web search event logs before it had MapReduce; engineers had to write custom scripts to answer questions like: what are the most frequent queries? how do we build inverted indices? how do we aggregate unbounded streams of data?

Many LC problems answer these exact questions and tap into your ability and intuition to process loads of data efficiently. Examples:

- Streaming data & online aggregation 👉 sliding window, two pointers, monotonic queues (e.g., [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/), [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/))
- Counting, frequency, and heavy hitters 👉 hash maps, heaps, bucket sort (e.g., [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/), [Majority Element](https://leetcode.com/problems/majority-element/description/))
- Inverted indices & lookup tables 👉 hash maps, tries (e.g., [Word Pattern](https://leetcode.com/problems/word-pattern/description/), [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/description/), [Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/description/))
- Event scanning and interval logic 👉 prefix sums, line sweep, sorting (e.g., [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/description/), [Car Pooling](https://leetcode.com/problems/car-pooling/), [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/description/))
- Scheduling and resource allocation 👉 greedy + heap (e.g., [Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/), [Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/))
- Data compaction and normalization 👉 union find, graph traversal (e.g., [Accounts Merge](https://leetcode.com/problems/accounts-merge/description/), Sentence Similarity [I](https://leetcode.com/problems/sentence-similarity/description/) & [II](https://leetcode.com/problems/sentence-similarity-ii/description/))

Many complain that LC problems are detached from real work. But once I see them as toy versions of prod data processing and systems tasks, I solve LC problems faster, with more motivation and interest. 

## Objective-Oriented Programming
TBD

## ML Coding

It scares the hell out of most ML engineers to write PyTorch code without inheriting from your team's model class or using Cursor. In ML coding interviews, however, you're expected to write PyTorch as fluently as regular Python. Even if you can Google, you have no time. 

First, be really fluent in PyTorch. Read Raschka's [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) and play with [TensorGym](https://tensorgym.com/) to learn syntax. For a more comprehensive tutorial, go over [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/). 

Next, be really familiar with Transformer architectures as well as training and inference loops. Read Sebastian Raschka's [Build a Large Language Model](https://github.com/rasbt/LLMs-from-scratch) and reproduce all code on your own. Watch Andrej Karpathy's [GPT-2 video](https://www.youtube.com/watch?v=kCc8FmEb1nY). If you have enough time on your hands, follow along with Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and read some of his repos (e.g., [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)). Also be familiar with common optimization techniques for training and inference (e.g., [Flash Attention](https://lubits.ch/flash/), [LoRA](https://lightning.ai/lightning-ai/environments/code-lora-from-scratch?section=featured), [KV cache](https://huggingface.co/blog/kv-cache)) and how to implement them from scratch. 


To test your understanding, solve LC-style ML problems on [Deep-ML](https://www.deep-ml.com/problems).

Last but not least, not all companies ask you to write PyTorch models. Some ask you to fit classical Scikit-learn models. Don't be caught off the guard --- brush up with Educative's [Scikit-learn cheat sheet](https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods).


## ML Fundamentals: Build an Impeccable Foundation

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


## Behavior Interview

## Project Deep Dive

