---
title: "What Makes Deep Learning Ranking Beautiful"
date: "2025-02-26"
locale: "en_US"
categories: ["personal reflections"]
---

My former manager at DoorDash criticized me for including necessary details for collaboration in my design doc, saying such efforts would go unnoticed by leadership and were therefore wasted. The manager before him criticized me for having a "narrow interest" in deep learning ranking, saying one could "code up a DNN ranker in an afternoon" and that I should pursue scope and visibility instead.

When you're early in your career, it's not easy to stay cool when people much more senior than you doubt your passion, value, and future. Back then, it didn't help that I couldn't fully articulate why I was so drawn to deep learning ranking. Sure, even textbooks from years ago talk about how deep learning automates feature engineering and can approximate arbitrarily complex functional forms. But was that kind of power and expressiveness what drove me? Not quite. Still, I trusted my gut and left.

Two and a half months into my new career as a DCN guy at Pinterest, I've started to vaguely understand what makes deep learning ranking so beautiful: here, **knowledge is power**. There's so little you can cheat yet so much you can do in deep learning --- how you improve your models and how you climb the career ladder are highly aligned. You have to carefully craft the knowledge you want your model to learn into clever modules and representations.

For instance, raw IDs are great for memorization because they tell us something about the user after they engage with this *exact item*, but they suffer from sparsity, cold start, and can explode model parameters and embedding tables. On the other hand, content embeddings or semantic IDs that reconstruct them are great for generalization since they tell us what the user is like after engaging with this kind of item, but they may hurt memorization and often tank model performance on popular items. How do you balance these objectives? There's an entire body of literature dedicated to balancing memorization and generalization in ID feature learning.

This is just one example among many --- every little thing behind a deep learning ranker has a wealth of literature and domain experts behind it. You may never become an expert in everything --- rather, it takes a village to raise a deep learning ranker. And in that village, you will grow as an engineer, and maybe one day, you become an oracle that younger engineers consult. Isolated and stumbling blindly in a culture that promotes visibility and downplays craft, you will never grow into a great ML engineer.

So what makes deep learning ranking beautiful is its honesty --- flaws in the data and your knowledge are amplified, so you gotta be good. And more importantly, I love the value alignment: to have visibility and scope is to become an expert, to do theoretically sound and use-case-appropriate work, and to build alongside like-minded people.