---
title: "National Importance of Cognitive Science in the AGI Era"
date: 2026-02-16
math: true
categories: ["cognitive science", "agi", "niw"]
toc: true
---

# Dhanasar's Prong 1: National Importance

The life of an immigrant hangs not by a thread, but by a sequence of precisely timed visas and forms: F-1 and I-20 when we accept our PhD offers; OPT/STEM OPT and I-766 when we graduate; H-1B and I-797 when we win the "H-1B lottery"; then I-140, I-485, and eventually the green card. I've long given up trying to explain this process to American friends. Even recruiters who make a living by working with candidates of all immigration statuses are often confused by it.

As a result of tech layoffs, labor market testing needed for [PERM](https://forumtogether.org/article/explainer-perm-labor-certification-process/) certifications is getting increasingly harder to pass. For many of us, the only alternative is to show the national importance of our work and obtain a National Interest Waiver ([NIW](https://www.uscis.gov/newsroom/alerts/uscis-updates-guidance-on-eb-2-national-interest-waiver-petitions)). Dr. Dhanasar and his team fought hard in 2016 to set the current standard ([*Matter of Dhanasar*](https://www.justice.gov/eoir/page/file/920996/dl)).

For many of us, "national importance" is not an afterthought we're scrambling together for the sake of NIW, but why we decided to pursue a PhD in the first place --- we believed our work mattered to humanity, not just one nation. I did when I dreamt of becoming a cognitive scientist, believing my field held the key to what makes us human and how intelligence comes to be. I chose a tech career because it better suited my personality and goals --- it doesn't make my past research any less important. Moreover, I see the NIW petition as a fun challenge to think of future research I can realistically pursue to advance CogSci + AI --- now that I'm a bit older, hopefully wiser, and know a lot more about AI/ML than when I was a CogSci PhD student.

<!-- I can't do anything unless I believe in its meaning. For instance, I can't prepare for interviews unless I believe it makes me a better engineer. Similarly, I can't prepare for an NIW case unless I believe what I've done or am about to do has national importance. So I often go a long way in search of that meaning, including spending an insane number of hours reading and thinking about ML engineering for interviews, and in this case, cognitively inspired AI for NIW --- even though there are many shortcuts to take (e.g., memorizing answers to common interview questions, or sketching out a seemingly reasonable research agenda that has little chance of becoming real). This need to believe makes me who I am and is the motivation behind this blogpost. -->

There are many great NIW examples in the natural sciences --- check out chemist Andrey Solovyev's informative [I-140 website](https://andreychemist.github.io/)! As far as I know, this post may be the first related to CogSci NIW petitions. This is a WIP --- I'll update it as I think of more ways CogSci informs AGI.

# What CogSci Has to Say About AI

> "An important feature of a learning machine is that its teacher will often be very largely ignorant of quite what is going on inside, although he may still be able to some extent to predict his pupil's behavior." --- Alan Turing, [Computing Machinery and Intelligence](https://courses.cs.umbc.edu/471/papers/turing.pdf) (1950)

On April Fool's Day in 2019, when I was in grad school, I put together a fake newspaper article claiming "AI has achieved the intelligence of a 5-year-old" and shared it in Prof. [Alison Gopnik](https://www.alisongopnik.com/)'s [*Possible Minds*](https://www.edge.org/conversation/alison_gopnik-possible-minds-25-ways-of-looking-at-ai) seminar. A split second of shock turned into five seconds of laughter.

My practical joke landed because even the best "AGI" felt wildly unhuman-like at the time. [GPT-2](https://en.wikipedia.org/wiki/GPT-2) had just come out two months earlier, and we were already amazed how it could generate 3 pages of somewhat coherent text, even though the content was often nonsensical. It was hard to imagine how any AI could ever resemble any 5-year-old's creativity and learning abilities in any shape or form. 

Fast forward to today, the tide has completely turned. After skimming new papers by cognitive scientists and AI researchers with deep CogSci roots (e.g., [*Large Language Model Reasoning Failures*](https://www.arxiv.org/abs/2602.06176)), it feels like many researchers now afford large language models the same respect they give human children and adults. LLMs are treated as yet another (black-box) intelligence. Researchers probe their reasoning and ethical failures the way they study humans, and try to mitigate those failures by steering behavior with small amounts of data (e.g., prompting, SFT, RLHF), rather than going through pretraining/evolution again. Five-year-old intelligence was a punchline. Now it's probably a baseline.

## Contributions by Domain

A central question in CogSci is: <span style="background-color: #FFE88D">*how can humans (infants + children + adults) learn so much from so little, so quickly?*</span>

Since the early 2000s, a view popularized by Josh Tenenbaum, Tom Griffiths, and colleagues is that:
(1) humans don't consider all possible hypotheses during reasoning (e.g., we don't search the ceiling for our glasses) --- rather, we rely on rich prior knowledge to narrow the hypothesis space (e.g., desk, sofa, under the bed); and (2) new data we observe percolates via hierarchical Bayesian models to update more abstract beliefs (e.g., search first where we last used the object).

Papers expressing the above view are too many to name. For a quick overview, read [*How to Grow a Mind*](https://cocosci.princeton.edu/tom/papers/LabPublications/GrowMind.pdf). For a comprehensive treatment, read the finally published [*Bayesian Models of Cognition*](https://mitpress.mit.edu/9780262049412/bayesian-models-of-cognition/) by the OGs. Norvig and Russell's [*Artificial Intelligence: A Modern Approach*](https://aima.cs.berkeley.edu/) is not a CogSci book but it mirrors domains traditionally studied in CogSci.

My crude take is this: <span style="background-color: #D9CEFF">much of CogSci studies what looks like "post-training" and "prompt engineering" in AI</span>. The human mind is pretrained, by evolution and development, and therefore comes with strong priors. That's why we can learn quickly from so little data --- nothing magical, but because most of the knowledge is already there. What we call "learning" in CogSci experiments either exposes or updates prior knowledge rather than redoing pretraining from scratch.

If this view holds, where CogSci can contribute the most to AI is post-training and prompting. <span style="background-color: #D9CEFF">By identifying the priors humans rely on in different domains, we can design post-training or prompting strategies that compensate for gaps in current models</span> --- especially where AI still struggles but humans learn effortlessly. Below are some examples.

### AI that Reasons to Act

[Shunyu Yao](https://ysymyth.github.io/) (not to be confused with [Shunyu Yao](https://www.linkedin.com/in/shunyu-yao-204158285/)) wrote [the second half](https://ysymyth.github.io/The-Second-Half/) of AI began when "RL finally works". Karpathy went further to [say](https://www.youtube.com/watch?v=BlVnGXEzFow) that 2025 was not the year of agents, but rather we're living in the "decade of agents", where autonomous agents will be an integral part of human life, but their cognitive skills will take a decade to develop.

For language agents, Yao's [ReAct](https://arxiv.org/abs/2210.03629) remains one of the most influential papers. The core idea appeared in his earlier work, [CALM](https://ysymyth.github.io/papers/Dissertation-finalized.pdf), where language agents were trained to play [Zork](https://en.wikipedia.org/wiki/Zork)-like text games. The agent issues actions in natural language and receives text observations. 

{{< figure src="https://www.dropbox.com/scl/fi/g3a8yhv30clzencw8lz8e/Screenshot-2026-02-16-at-2.40.19-PM.png?rlkey=fsg8sk6wb6cntj9lspieoh9px&st=6lepxd0k&raw=1" caption="An example text puzzle from the game Zork." width="1800" align="center">}}

Above is an example from CALM. Unlike games with finite action spaces such as chess, Atari, or [Cluedo](https://en.wikipedia.org/wiki/Cluedo), text games have infinite action spaces. The agent can say anything, whether or not anyone has said it before. Traditional approaches restrict the action space (allowing only a small set of actions at each state) or provide ground-truth actions as supervision. Agents trained this way generalize poorly to new games. 

The key shift was to use pretrained language models with rich prior knowledge as agents. Humans don't consider all possible actions --- our prior knowledge ("inductive biases") help us focus on the plausible few. For instance, since a door is nailed shut, we should search within the room. Since the rug often hides things, let's check under it. Strong pretrained models share many such priors. Even GPT-2 outperformed traditional baselines by a large margin. If the pretrained model were much weaker, it'd be like casting pearls before swine ("对牛弹琴").

Back in grad school, when I told non-CogSci friends that people learn faster than machines because they have inductive biases that narrow the hypothesis space, they found it incredibly boring 😂. To them, the "interesting part", be it pretraining or evolution, was already done. Sometimes, I felt the way about CogSci. Sometimes, I read the same sentiment about Agentic AI ("it's nothing special but prompting a pretrained model"). But thinking again, figuring out how to elicit knowledge from a pretrained intelligence, so it can do things it was never explicitly trained to do, is fascinating, useful, and mysterious.

Later, ReAct interleaves reasoning and action in a more structured way, and by then pretrained models (e.g., GPT-3) were much stronger. Given a task with many possible outcomes (e.g., where a pepper shaker might be) --- some more likely than others --- a language agent can reason about plans (places to search), execute actions, observe results, update beliefs, and revise plans. This loop of reasoning and acting is what human adults and children implicitly do in everyday life. 

{{< figure src="https://www.dropbox.com/scl/fi/2px3uq6t19uj2tn28xhtr/Screenshot-2026-02-09-at-8.49.48-PM.png?rlkey=ahxp5hywwqrkd7visrgg2l5pu&st=hted00yq&raw=1" caption="A [ReAct](https://arxiv.org/abs/2210.03629) agent interleaves reasoning and acting to get to correct answers fact." width="1800">}}

Reading ReAct brought back to memories of grad school, especially my years spent on studying explanation (e.g., [Wilkenfeld & Lombrozo, 2015](https://cognition.princeton.edu/publications/inference-best-explanation-ibe-versus-explaining-best-inference-ebi)), information seeking (e.g., [Rothe, Lake, & Gureckis, 2018](https://www.cs.princeton.edu/~bl8144/papers/RotheEtAl2018CompBrainBehavior.pdf)), and how explanation and information seeking reinforce each other (e.g., [Lampinen et al., 2022](https://proceedings.mlr.press/v162/lampinen22a/lampinen22a.pdf) and [my dissertation](https://www.proquest.com/openview/9886c692bc27fcb566ef80fd54820735/1?pq-origsite=gscholar&cbl=18750&diss=y)). It's interesting to see many recent reasoning papers (2024–2026) only citing CogSci ideas from decades ago (e.g., working memory, System 1 vs. System 2, etc.). I feel modern CogSci has much to contribute to the "decade of agents".

Here's one idea --- <span style="background-color: #D9CEFF">leveraging the latest works on human causal reasoning to improve causal reasoning accuracy and efficiency of language agents</span>. [BIG-bench](https://github.com/google/BIG-bench) introduces tasks designed to be hard for language models. In the original [*Beyond the Imitation Game*](https://arxiv.org/abs/2206.04615) (BIG) paper, one such task was causal reasoning. In this 2023 paper, even the larger models struggled to judge *"The driver turned the wipers (A) on because it started raining (B)"* as more plausible than *"It started raining (B) because the driver turned the wipers on (A)"*, while humans instantly know A (raining) causes B (turning on wipers), but not the reverse.

Causal inference matters because it tells us which interventions change outcomes. If you want rain, turning on the wipers won't help. If you want to see clearly in the rain, turning on the wipers will. As more people turn to ChatGPT or Gemini for life decisions, giving causally grounded advice is important to help them get what they ask for.

To improve the causal reasoning of language models, we can borrow scaffolding from human learning. For example, we can prompt counterfactual thinking: without rain, would the driver turn on the wipers? Probably not. So intervening on rain changes the driver's action, making rain a plausible cause. But if the driver turns off the wipers during rain, would that stop the rain? No. So intervening on the wipers doesn't change the weather, making it an unlikely cause.

Real-world causal reasoning is often more complex than one-to-one links. Sometimes the effect is only observed after we act. For instance, in my dissertation, I recounted a night when the lights in my apartment went out. In one moment, I formed 3 hypotheses: a PG&E outage, a tripped circuit breaker, or a burned-out bulb. I ran two "experiments" in quick succession. If it were a PG&E outage, my neighbor's lights would be off --- experiment 1: I checked their lights and they were on. If the breaker had tripped, the switch would be down --- experiment 2: I opened the fuse box and found the switch was down. I flipped the switch and turned the light back on. Humans like myself perform countless experiments daily to gather data and infer causality. The process is so instinctual and efficient that we don't even realize we're doing it. If one day we lose this ability, we'll definitely notice --- I'd be sitting in the dark helplessly, or flipping the wall switch aimlessly.

Unlike adults, my former colleagues and I ([Meng, Bramley, and Xu, 2018](https://escholarship.org/uc/item/0p40378v)) found that 5-to-7-year-olds struggle when they must choose interventions to rule out wrong hypotheses in as few attempts as possible. But when we explicitly teach young children to think through how each possible intervention eliminates specific hypotheses, they explain their intervention choices better (e.g., *"I do X because different things will happen under different hypotheses."*) and also select better interventions (i.e., actions that bring larger information gain).

Similarly for agents, it may not be enough to just reason and act. The lesson from CogSci is that we can teach agents how to reason --- for example, by encouraging systematic hypothesis generation and intervention planning --- so they can act more efficiently to gather informative data, and in turn use that data to reason more accurately.

Above is one idea among many. More generally, each of us (former) CogSci researchers has studied areas where human children or even adults do seemingly irrational or ineffective things. We often design ways to improve reasoning for certain age groups or under certain circumstances. The same mitigation strategies may or may not apply to language agents. This kind of research we've done in the past gives us a good starting point to help language agents reason better today.


### AI That is Fair & Unbiased

### AI with Physical Intuitions

world models

## Contributions by Methodology

### Benchmarks for Human + Machine Learners

more realistic benchmarks based on real recsys
- recsys as environment for language agents to reason and act -- webshop benchmark
- web-scale recommender systems as the playground for agents with human priors
- frame recsys as web interaction


SOTA model paired by old cognitive tasks
cite Baddley 
collab with LMArena

### Scaffold Human + Machine Learning

can't get IRB approval. so studying LLMs may be a good way

# Carry On CogSci Research in Industry

i used to joke that every road leads to recsys
many fellow cogsci students became recsys mle
upon reflection, it's not a coincidence 
it's because recsys is important in life -- economy politics entertainment 
moreover recsys is a great way to train and deploy agents that will play a part in our lives 

## RecSys as the Playground for AI Agents
WebShop: item-first search
- RecSys is important and realistic for training, testing, and deploying AI agents

## Reach out to Institutes and Benchmark Creators

LMArena

# Resources + Inspirations

## AI + CogSci Researchers to Follow
many ai researchers in prev gen have cogsci training
nowadays rare  
- **All Things Cognitively Inspired AI**: [Geoff Hinton](https://scholar.google.com/citations?user=JicYPdAAAAAJ&hl=en), [Josh Tenenbaum](https://scholar.google.com/citations?hl=en&user=rRJ9wTJMUB8C&view_op=list_works&sortby=pubdate), [Tom Griffiths](https://cocosci.princeton.edu/publications.php), [Noah Goodman](https://cocolab.stanford.edu/publications), [Steve Piantadosi](https://colala.berkeley.edu/), [Brenden Lake](https://www.cs.princeton.edu/~bl8144/), [Sam Gershman](https://gershmanlab.com/pubs.html)
- **Agents and reasoning**: [Shunyu Yao](https://ysymyth.github.io/), [Andrew Lampien](https://scholar.google.com/citations?hl=en&user=_N44XxAAAAAJ&view_op=list_works&sortby=pubdate), [Tania Lombrozo](https://cognition.princeton.edu/publications), [Caren Walker](https://elc-lab-ucsd.com/publications), [Elizabeth Bonawitz](https://ccdlab.hsites.harvard.edu/publications), [Todd Gureckis](https://www.gureckislab.org/papers)
- **Social cognition**: [Hyowon Gweon](https://scholar.google.com/citations?user=zjl9R-oAAAAJ&hl=en), [Julian Jara-Ettinger](https://compdevlab.yale.edu/publications.html), [Bill Thompson](https://ccs-ucb.github.io/publications.html), [Xuechunzi Bai](https://baixuechunzi.github.io/uchicago/publications.html), [Alan Cowen](https://www.alancowen.com/publications), [Max Kleiman-Weiner](https://scholar.google.com/citations?user=SACXQKYAAAAJ&hl=en)
- **Intuitive physics and "World Models"**: [Fei-Fei Li](https://profiles.stanford.edu/fei-fei-li), [Yann Lecun](http://yann.lecun.com/), [Jiajun Wu](https://jiajunwu.com/), [Tomer Ullman](https://scholar.google.com/citations?user=5SF-hRsAAAAJ&hl=en), [Jessica Hamrick](https://www.jesshamrick.com/papers/), [Kevin Smith](https://www.mit.edu/~k2smith/)

## Papers, Talks, and Inspiring Words
- Opinions on CogSci + AI: Jay McClelland's [interview](https://news.stanford.edu/stories/2024/11/from-brain-to-machine-the-unexpected-journey-of-neural-networks), Reddit [post](https://www.reddit.com/r/MachineLearning/comments/10wtumf/discussion_cognitive_science_inspired_ai_research/)
- How neural nets learn to generalize: [*Explaining Grokking through Circuit Efficiency*](https://arxiv.org/abs/2309.02390)
- RecSys + reasoning: [RL-aligned ranking](https://recsysml.substack.com/p/stop-predicting-ctr-start-optimizing)

- [*The Thinking Game*](https://youtu.be/d95J8yzvjbQ?si=sPyAYlANyfm4RewK), a documentary on DeepMind CEO Demis Hassabis' life and search for AGI
- Shunyu yao's dissertation 
- https://ysymyth.github.io/The-Second-Half/
- https://karpathy.bearblog.dev/year-in-review-2025/
- https://www.youtube.com/watch?v=ff-ip0A40ks
- https://www.youtube.com/watch?v=YBVlVlBWQhs
- https://www.youtube.com/watch?v=46A-BcBbMnA
- https://runzhe-yang.science/2023-05-05-socratic/


## Writing Guide for National Importance
- Reddit posts: [*Substantial Merit vs National Importance of your EB-2 NIW Proposed Endeavor*](https://www.reddit.com/r/EB2_NIW/comments/1oygx31/substantial_merit_vs_national_importance_of_your/)
- Xiaohongshu posts on NIW writing tips: examples [1]( https://www.xiaohongshu.com/discovery/item/696f38fb000000000a02b8ea?source=webshare&xhsshare=pc_web&xsec_token=ABlkwa_GCrUBYzvCNgrH81ExLJiQLrld6J8Uu--uJQ9_M=&xsec_source=pc_share
), [2](https://www.xiaohongshu.com/discovery/item/6966946c000000000e00d8b1?source=webshare&xhsshare=pc_web&xsec_token=ABxzZX4CErOzETZ99sJ5szWXnlhqJwZUXLBMyIly__K3U=&xsec_source=pc_share), [3](https://www.xiaohongshu.com/discovery/item/6966946c000000000e00d8b1?source=webshare&xhsshare=pc_web&xsec_token=ABxzZX4CErOzETZ99sJ5szWXnlhqJwZUXLBMyIly__K3U=&xsec_source=pc_share), [4](
https://www.xiaohongshu.com/discovery/item/696e3c0d000000001a023a81?source=webshare&xhsshare=pc_web&xsec_token=ABgpq_DuFyBr3YMDGiTGmvr7QUvc5RW3xeCf_M25pGr4E=&xsec_source=pc_share)
- Grant proposals: how professors ask governments for money 
  - STEM education: [NSF #2400757](https://ecrhub.org/ecr-projects?id=2400757), [NSF #1640816](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=1640816)


- thank you emails & linkedin messages -->