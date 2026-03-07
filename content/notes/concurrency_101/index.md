---
title: "Multithreading and Concurrency 101"
date: 2026-03-06
categories: ["coding", "distributed systems"]
---

# How to Prepare a Chinese New Year Feast in Two Hours

I don't like cooking much since it takes me two hours to prepare two dishes. I always wondered how my family managed to prepare a dozen dishes for the Chinese New Year feast in the same amount of time.

Turns out I'm a single-threaded CPU working on one task at a time (wash ingredients 👉 chop them 👉 boil water 👉 put in chopped ingredients), whereas my dad and aunties mastered multithreading. One would boil water, another chop ingredients, someone else would prepare the dipping sauces, while another handled stir frying.

To speed things up, modern CPUs similarly tend to work on multiple tasks ("threads") concurrently, often across multiple cores (like my dad and aunties). For example, when you load a complex web app, one thread may be busy fetching data from a database over the network while another simultaneously renders the UI on the screen.

Things can go horribly wrong in this process. What if two chefs tasted the soup around the same time and both decided to add more salt? Your soup quickly turns into sea water. What if one person grabbed the only knife and another grabbed the only cutting board? They form a deadlock. It takes skill to prevent such disasters, and such skills are what companies like OpenAI, Anthropic, xAI, and Netflix hire for --- whether you're a Software Engineer or a Research/ML Engineer.

# Key Concepts

- **Concurrency** is a program's ability to do multiple things at once.
  - More formally, concurrency is the ability of a program to make progress on multiple tasks during overlapping time periods rather than strictly sequentially.
  - The goal is to improve CPU utilization and responsiveness.
  - Different from **parallelism**, where multiple tasks execute on different CPU cores.

- Below are names of things the program works on simultaneously:
  - **Process**: An executing instance of a program with its own memory space and system resources.
  - **Thread**: The smallest unit of execution within a process; multiple threads share the same memory and resources.
    - **Shared Memory**: Threads within a process share memory, allowing efficient communication. The risk is, multiple threads can access the same data concurrently.
  - **Task**: A unit of work scheduled for execution (e.g., a function, coroutine, or job). Tasks may run on threads or processes depending on the concurrency model.

- Below are common ways to do several things simultaneously:
  - **Multithreading**: Concurrency allows a single process to run multiple threads concurrently; tasks share memory and execute in overlapping time periods.
  - **Multiprocessing**: Parallelism allows multiple processes to run concurrently on CPU cores with their own memory spaces.
  - **Asynchronous**: Tasks cooperatively yield control (e.g., `async` / `await`) so other tasks can run while waiting for I/O, typically managed by an event loop.

 - Concurrency often introduces nondeterminism:
 	- "Symptom": The same program changes results across runs.
 	- **Thread Scheduling** is one possible cause: The operating system decides which thread runs and when --- the execution order of concurrent programs is not guaranteed.
 
- **Synchronization** coordinates access to shared resources so concurrent programs behave correctly despite nondeterminism.
  - **Atomic Operations**: We should rely on operations that execute as a single, indivisible step, preventing other threads from observing intermediate states. Common tools include:
    - **Locks / Mutexes** — ensure only one thread enters a critical section at a time.
    - **Semaphores / Monitors** — coordinate access among multiple threads and manage limited resources.

- Below are common problems in multithreading systems:
  - **Race Conditions**: Multiple threads access or modify shared data without proper synchronization, producing unpredictable results depending on execution order.
    - **Critical Sections**: Code that access shared resources and must be executed by only one thread at a time.
  - **Deadlock**: Two or more threads wait indefinitely for resources held by each other.
  - **Starvation**: A thread never obtains the resources it needs because other threads continually acquire them first.

# Practice Problems

# Read More
- Educative: [Multithreading and Concurrency Fundamentals](https://www.educative.io/blog/multithreading-and-concurrency-fundamentals)
- MIT 6.031: [Reading 19: Concurrency](https://web.mit.edu/6.031/www/fa17/classes/19-concurrency/)
- Real Python: [Speed Up Your Python Program With Concurrency](https://realpython.com/python-concurrency/#exploring-concurrency-in-python)
- LeetCode: [Concurrency Collection](https://leetcode.com/problem-list/concurrency/)
