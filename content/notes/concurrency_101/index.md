---
title: "Multithreading and Concurrency 101"
date: 2026-03-06
categories: ["coding", "distributed systems"]
toc: true
math: true
---

# How to Prepare a Chinese New Year Feast in Two Hours

I don't like cooking much since it takes me two hours to prepare two dishes. I always wondered how my family managed to prepare a dozen dishes for the Chinese New Year feast in the same amount of time.

Turns out I'm a single-threaded CPU working on one task at a time (wash ingredients 👉 chop them 👉 boil water 👉 put in chopped ingredients), whereas my dad and aunties mastered multithreading. One would boil water, another chop ingredients, someone else would prepare the dipping sauces, while another handled stir frying.

To speed things up, modern CPUs similarly tend to work on multiple tasks ("threads") concurrently, often across multiple cores (like my dad and aunties). For example, when you load a complex web app, one thread may be busy fetching data from a database over the network while another simultaneously renders the UI on the screen.

Things can go horribly wrong in this process. What if two chefs tasted the soup around the same time and both decided to add more salt? Your soup quickly turns into sea water. What if one person grabbed the only knife and another grabbed the only cutting board? They form a deadlock. It takes skill to prevent such disasters, and such skills are what companies like OpenAI, Anthropic, xAI, and Netflix hire for --- whether you're a Software Engineer or a Research/ML Engineer.

# Key Concepts in Concurrency

- **Concurrency vs. Parallelism**: To [quote](https://go.dev/blog/waza-talk) Rob Pike, one of Go's creators --- "Concurrency is about *dealing* with lots of things at once. Parallelism is about *doing* lots of things at once."

  - **Concurrency**: We can cleverly structure a task into independent subtasks that can be paused, switched between, and run in *overlapping* time periods. It improves the program's CPU utilization, throughput, and responsiveness.
  - **Parallelism**: On different physical CPU cores, we can, of course, execute different tasks at *exactly the same* time.
 
  - **Our cooking analogy** 🍳
    - **Concurrency:** As the sole chef, I put water on to boil and chop meat while waiting. I only have one pair of hands (one core), but can interleave subtasks to cook faster.
    - **Parallelism:** When cooking the New Year dinner, I chop meat while my aunt stir-fry vegetables. We're separate chefs (two cores) doing tasks at the exact same time.
    - **Shared Memory:** We are both adding different ingredients to the exact same stew pot at the same time.

- Below are names of things that a program (i.e., an executable file like `train.py`) can work on simultaneously:
  - **Process**: An executing instance of a program (a training job) with its own isolated memory space and system resources.
  - **Thread**: The smallest unit of execution within a process.
    - **Shared Memory**: Multiple threads within a single process share the same memory space, allowing incredibly fast communication. The risk is that multiple threads can access and modify the same data concurrently. The result may be incorrect.
  - **Task**: A conceptual unit of work scheduled for execution (e.g., a function, coroutine, or data-loading job). Tasks may run on threads or processes depending on the architecture.
  

- Below are execution models to do things simultaneously:
  - **Multithreading**: A single process runs multiple threads concurrently. Tasks share memory and execute in overlapping time periods. Languages like Java allow tasks to run in parallel if spread across multiple cores, but languages like Python are limited by the [Global Interpreter Lock](https://en.wikipedia.org/wiki/Global_interpreter_lock) (GIL).
  - **Multiprocessing**: Spawns multiple entirely separate processes, each with its own memory space. This achieves true parallelism across CPU cores and avoids shared-memory risks, but communication between processes is slower.
  - **Asynchronous (Async/Await)**: A cooperative, usually single-threaded model where tasks explicitly yield control back to an event loop when waiting for I/O (like network requests), so other tasks can run in the meantime.

- Multithreading using shared memory introduces **nondeterminism**:
  - **Thread Scheduling**: The operating system dictates which thread runs and when. The execution order is never guaranteed, which means the exact same program might execute in a slightly different order across runs.
  
- This nondeterminism leads to common **multithreading problems**:
  - **Race Conditions**: Multiple threads access or modify shared data simultaneously without coordination. The final outcome becomes a "race" and yields unpredictable, buggy results.
    - **Critical Sections**: The specific blocks of code that access these shared resources. If multiple threads enter a critical section at once, race conditions occur.
  - **Deadlock**: Two or more threads wait indefinitely for resources held by each other (e.g., you waiting for the only knife and me waiting for the only cutting board).
  - **Starvation**: A thread never obtains the resources it needs to proceed because other threads continually acquire them first.

- **Synchronization** coordinates access to shared resources so concurrent programs behave correctly despite nondeterminism. In other words, it ensures **thread safety**, under which different threads can access shared resources without nondeterminism.
  - **Atomic Operations**: Operations that execute as a single, indivisible step from the perspective of other threads. No other thread can observe an intermediate state.
  - **Locks / Mutexes**: Enforce mutual exclusion. They ensure only *one* thread can enter a critical section at a time.
  - **Semaphores / Monitors**: A signaling mechanism to coordinate access among multiple threads and manage a limited pool of resources (allowing $N$ threads to access a resource instead of just $1$).

# Practice Concurrency Problems


## Synchronization

[1114. Print in Order](https://leetcode.com/problems/print-in-order/?envType=problem-list-v2&envId=concurrency)

[1115. Print FooBar Alternately](https://leetcode.com/problems/print-foobar-alternately/?envType=problem-list-v2&envId=concurrency)

[1188. Design Bounded Blocking Queue](https://leetcode.com/problems/design-bounded-blocking-queue/?envType=problem-list-v2&envId=concurrency)

## Thread Communication

[1117. Building H2O](https://leetcode.com/problems/building-h2o/description/?envType=problem-list-v2&envId=concurrency)




# Read More
- Educative: [Multithreading and Concurrency Fundamentals](https://www.educative.io/blog/multithreading-and-concurrency-fundamentals)
- MIT 6.031: [Reading 19: Concurrency](https://web.mit.edu/6.031/www/fa17/classes/19-concurrency/)
- Real Python: [Speed Up Your Python Program With Concurrency](https://realpython.com/python-concurrency/#exploring-concurrency-in-python)
- LeetCode: [Concurrency Collection](https://leetcode.com/problem-list/concurrency/)
- LeetCode Discussions: [Surgical Strike on Concurrency and Multithreading](https://leetcode.com/discuss/post/7605788/surgical-strike-on-concurrency-and-multi-nghz/)
