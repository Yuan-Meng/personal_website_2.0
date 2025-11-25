---
title: Preparing for ML Infra System Design Interviews
date: 2025-11-25
math: true
categories: ["career", "machine learning", "infra", "interview"]
toc: true
---

# The Knowledge Dilemma: Those Who Build Models for Scalable RecSys Don't Work on Scalable Infra

A handful of companies (Netflix, Reddit, Snap, DoorDash, etc.) conduct an ML infra design round for MLE candidates. The focus isn't on model development --- like collecting training data and generating labels, choosing features and model architectures, or running online/offline experiments --- but on offline + online pipelines that support an ML system. You may be asked to walk through the entire online + offline pipelines or design a specific component, such as feature stores, real-time feature updates, or distributed training. The goal is to see whether you have a full picture of how different pieces of ML infra work together and know how to scale each part and make trade-offs.

Here's the funny thing: If you are an ML engineer working at a company with large-scale recommender systems like Meta, Google, or Pinterest, you may have a fairly shallow understanding of ML infra only as a user, since there are dedicated infra teams handling logging, training, and inference for you. Your job is mostly model iteration, experiments, and maybe some observability. However, if you don't need to make recommendations at scale, this knowledge is irrelevant.

# References