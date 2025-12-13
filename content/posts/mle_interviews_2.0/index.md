---
title: Tackling Challenging ML + Research Engineering Interviews
date: 2025-12-12
math: true
categories: ["career", "machine learning", "interview"]
toc: true
---

# Recap: "Standard" MLE Interviews

This is a sequel to my ML interview {{< backlink "mle_interviews" "blogpost" >}} in 2024. Back then, I argued that ML engineers are unicorns: the variety of interview rounds far exceeds that of other job families. A typical full loop had 6 rounds:

1. **Coding**: medium–hard LeetCode problems, object-oriented programming (often multi-level CodeSignal), and ML coding (e.g., debugging or building and training a simple model)
2. **ML system design**: design a {ranking, query understanding, content understanding, NLP, trust & safety} ML system, focusing on data, features, labels, and model architectures
3. **Project deep dive**: walk through a project you're most proud of
4. **ML fundamentals**: rapid-fire questions on machine learning foundations, such as common network architectures, optimization routines, loss functions, activation functions, regularization, etc.
5. **Behavior and leadership**: a time when you demonstrated problem solving, team work, conflict resolution, time management, leadership, adaptability, growth, etc. beyond your level
6. **Domain knowledge**: woven into all rounds, explaining how you shipped past projects or tackle current problems using expertise

Now I think if you only encounter these 6 rounds, count yourself lucky --- you've had an easy-peasy interview by today's standards. Many ML and Research Engineering roles add extra rounds that most ML practitioners find genuinely hard. I like to think of interview rejections as "negative sampling". When hiring amass, companies may "only" need to find the best in hundreds, so "in-batch negatives" (where the batch size is a few hundred) suffice. Frontier AI labs, however, can afford to find the best in thousands or more --- so they have to throw in hard negatives. "Hard" means rounds most of us are terrified of.

# MLE Interview 2.0: What Has Changed?

## Additional Rounds
Many companies start to include more challenging rounds:

1. **Long-winded ML coding**: A few years ago, you might be asked to implement a simple model from scratch (e.g., KNN, K-means, decision trees, logistic regression, linear regression, MLP) and fit it on toy data. Today, you're likely to debug or implement LLM training or inference code --- Transformer encoders/decoders, KV cache, LoRA, and more. In especially brutal interviews, you may even be asked to implement [autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). The ideal candidate is Andrej Karpathy 2.0: chatting and writing hundreds of lines of code in an hour while vividly explaining LLM fundamentals.
2. **ML infra design**: Some say OpenAI is a research lab that accidentally got popular after releasing a web app. Even now, it still operates like a startup. In startups and many high-performing organizations, teams are lean, roles are blurred, and ML engineers are expected to unblock themselves across the entire stack. As a result, ML infra design is a common interview round at companies where ML engineers self-serve (e.g., Netflix, Snap, Reddit). I recently wrote a {{< backlink "ml_infra_interviews" "blogpost" >}} on how to prepare for this round.
3. **Research presentation**: In traditional project deep dives, you verbally walk through one or two representative projects. Some Research Engineer roles require a job talk style presentation on your past work. You make slides, go over technical details, and "defend" your body of work like a PhD candidate would.

You may wonder how these rounds are even possible to prepare for. Most don't have to know and can land a rewarding job at Google, Meta, Pinterest, Roblox, Airbnb, or the likes. But if you're already on a core ML team at one of them, there's limited upside to jumping ship. If you crave something more challenging, rewarding, or just different, you might have no choice but to crack these additional rounds, so you can join OpenAI, Anthropic, Google DeepMind, xAI, or the likes.

## Higher Bar

I used to have a pretty accurate "feel" for when an offer was coming. Last year, if I solved all the coding problems (even with minor flaws), had a complete system design, and had an enjoyable conversation with the hiring manager, an offer would usually arrive in a few days. Among all onsites I chose to finish, I only received one rejection.


That reality has changed. Our culture tries to steer clear from the idea of "perfection", seeing it as unattainable or mentally taxing. Sadly, perfection has become the new bar at companies people want to join. 

1. **Coding**: In a 45-minute LeetCode-style interview with one Hard problem, the ideal outcome is to write a bug-free solution in 15–20 minutes, come up with sharp test cases, pass them, and then discuss optimizations. If there are follow-ups, you're expected to provide extendable solutions. In a one-hour OOP-style CodeSignal interview, most people can't even type fast enough to finish all 4 levels; you should aim to finish in ~40 minutes so you have time to discuss how to scale the toy system (e.g., it's usually a database, message queue, or workflow) to production. In ML coding, even reading the full training and inference loops can overwhelm most candidates --- but you're expected to complete the implementation and improve model performance in an hour.
2. **ML system design**: By now, everyone knows how to sketch a standard ranking pipeline --- L1 (candidate generation) 👉 L1 (first-stage ranking) 👉 L2 (second-stage ranking) 👉 L3 (call it re-ranking, value models, or "special effects"). If you sound like everyone else, that won't cut it anymore. You need full control of the conversation: use the ranking funnel as scaffolding, but show insights that only a RecSys expert would have. Bring ideas and solutions from your work (e.g., you've worked on lifelong sequence models and they are actually a good choice here) and industry frontiers (e.g., this year's standouts are perhaps "One" series from Kuaishou or {{< backlink "generative_recommendation" "generative recommendation" >}} in general).
3. **Project deep dive**: I've got offers because the team needed someone to work on X, and I happened to have worked on X. In one case, my first interviewer was so interested in X that they skipped the coding question and those in the next few rounds ditched the agenda to ask more about X. If you work on models with broad industry applications and your team is ahead of the industry, you have a natural advantage over very well-prepared candidates who don't have such experience. I think this is why you need to choose your current projects or next moves carefully --- as ML engineer, try to choose a company that has the same or a more advanced ML stack as your current one; only move to a company with a dated ML stack if you get a once-in-a-lifetime level bump (e.g., +2) or a package that keeps you happy for five years. Otherwise, you're cashing out too early and losing your advantage in 2-3 years. That said, you can still fail this round if you talk about a complex project as if it were trivial. I wrote a [post](https://www.yuan-meng.com/notes/project_complexity/) on how to present complex projects in ways that do them justice.
4. **ML fundamentals**: Most of the time, you'll get 3–5 rapid-fire ML fundamentals questions in a phone screen, often before coding. Many candidates are confused when they fail, remembering they've solved the coding question perfectly, only to forget that they made fundamental mistakes on ML fundamentals. One or two wrong answers can be enough for rejection. And if those 5 minutes already feels exposing, know that some companies have one or two 45-minute onsite rounds that keep asking ML fundamentals to test the limit of your deep learning knowledge. 
5. **Behavior and leadership**: It's hard to reflect on your career while you're still living it. If you claim to know perfectly well how to expand scope, manage priorities, collaborate with difficult teammates, and right all the wrongs, you're essentially saying "everyone is drunk while I alone am sober" (众人皆醉我独醒). That can't be true. You need repeated, honest conversations with friends and mentors to see where you can grow. Beyond that, you need to be that mentor and friend --- support junior engineers & peers and lead teams through difficult moments --- to show leadership across the board. Just like internal promotions, getting a Senior+ offer often requires showing Staff-level qualities.

## Longer Process

Increasingly more companies now ask for 2–3 references from recent managers, teammates, or internal referrals. I remember this was already the case when I was interviewing for my first job in grad school --- companies like Figma and Roblox have long required reference checks. Nowadays, all frontier AI labs ask for them.

About half of the companies have a team match stage. Last year, this process was pretty quick: for me personally, it was either done within a day, or skipped if the hiring manager I happened to meet during onsite liked me. This year, however, passing an onsite may not mean much: dozens of candidates who have clear the interview will talk to a handful of managers with openings. Your years of experience, project complexity, and interview performance are evaluated all over again.

So think carefully before jumping ship. To prepare so well to land an offer in this market, you inevitably lose some productivity at work. Only interview when there's an undeniable upside. And once you start, commit to finishing within two months with full determination. Otherwise, it may be better to stay put and wait for the storm to pass.


# Up the Ante on Interview Prep
## General Coding
### Leetcode Style 
### Objective-Oriented Programming

## ML Coding

## ML Foundations

https://udlbook.github.io/udlbook/

## Research Presentation
