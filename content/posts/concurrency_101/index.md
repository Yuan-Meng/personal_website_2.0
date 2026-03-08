---
title: "Multithreading and Concurrency 101 for ML Engineers"
date: 2026-03-07
categories: ["coding", "distributed systems"]
toc: true
math: true
---

# How to Get a Chinese New Year Feast Ready in Two Hours

I don't like cooking much since it takes me two hours to prepare two dishes. I always wondered how my family managed to prepare a dozen dishes for the Chinese New Year feast in the same amount of time.

Turns out I'm a single-threaded CPU working on one task at a time (wash ingredients 👉 chop them 👉 boil water 👉 put in chopped ingredients), whereas my dad and aunties mastered multithreading. One would boil water, another chop ingredients, someone else would prepare the dipping sauces, while another handled stir frying.

To speed things up, modern CPUs similarly tend to work on multiple tasks ("threads") concurrently, often across multiple cores (like my dad and aunties). For example, when you load a complex web app, one thread may be busy fetching data from a database over the network while another simultaneously renders the UI on the screen.

Things can go horribly wrong in this process. What if two chefs tasted the soup around the same time and both decided to add more salt? Your soup quickly turns into sea water. What if one person grabbed the only knife and another grabbed the only cutting board? They just created a deadlock. It takes skill to prevent such disasters, and such skill is what companies like OpenAI, Anthropic, xAI, and Netflix hire for --- whether you're a Software Engineer or a Research/ML Engineer.

# Key Concepts in Concurrency

- **Concurrency vs. parallelism**: To [quote](https://go.dev/blog/waza-talk) Rob Pike, one of Go's creators --- "Concurrency is about *dealing* with lots of things at once. Parallelism is about *doing* lots of things at once."

  - **Concurrency**: We can cleverly structure a task into independent subtasks that can be paused, switched between, and run in *overlapping* time periods. It improves the program's CPU utilization, throughput, and responsiveness.
  - **Parallelism**: On different physical CPU cores, we can, of course, execute different tasks at *exactly the same* time.
 
  - **Our cooking analogy** 🍳🧨
    - **Concurrency:** As the sole chef, I put water on to boil and chop meat while waiting. I only have one set of hands (one core), but can interleave subtasks to cook faster.
    - **Parallelism:** While cooking the New Year dinner, I chop meat while my aunt stir fries vegetables. We're separate chefs (two cores) doing tasks at the exact same time.
    - **Shared memory:** My dad and aunt both add different ingredients to the exact same stew pot at the same time.

- Below are names of things that a program (i.e., an executable file like `train.py`) can work on simultaneously:
  - **Process**: An executing instance of a program (a training job) with its own isolated memory space and system resources.
  - **Thread**: The smallest unit of execution within a process.
    - **Shared memory**: Multiple threads within a single process share the same memory space, allowing incredibly fast communication. The risk is that multiple threads can access and modify the same data concurrently. The result may be incorrect.
  - **Task**: A conceptual unit of work scheduled for execution (e.g., a function, coroutine, or data-loading job). Tasks may run on threads or processes depending on the architecture.
  

- Below are execution models to do things simultaneously:
  - **Multithreading**: A single process runs multiple threads concurrently. Tasks share memory and execute in overlapping time periods. Languages like Java allow tasks to run in parallel if spread across multiple cores, but languages like Python are limited by the [Global Interpreter Lock](https://en.wikipedia.org/wiki/Global_interpreter_lock) (GIL).
  - **Multiprocessing**: Spawns multiple entirely separate processes, each with its own memory space. This achieves true parallelism across CPU cores and avoids shared-memory risks, but communication between processes is slower.
  - **Asynchronous (async/await)**: A cooperative, usually single-threaded model where tasks explicitly yield control back to an event loop when waiting for I/O (like network requests), so other tasks can run in the meantime.

- Multithreading using shared memory introduces **nondeterminism**:
  - **Thread scheduling**: The operating system dictates which thread runs and when. The execution order is never guaranteed, which means the exact same program might execute in a slightly different order across runs.
  
- This nondeterminism leads to common **multithreading problems**:
  - **Race conditions**: Multiple threads access or modify shared data simultaneously without coordination. The final outcome becomes a "race" and yields unpredictable, buggy results.
    - **Critical sections**: The specific blocks of code that access these shared resources. If multiple threads enter a critical section at once, race conditions occur.
  - **Deadlock**: Two or more threads wait indefinitely for resources held by each other (e.g., you waiting for the only knife and me waiting for the only cutting board).
  - **Starvation**: A thread never obtains the resources it needs to proceed because other threads continually acquire them first.

- **Synchronization** coordinates access to shared resources so concurrent programs behave correctly despite nondeterminism. In other words, it ensures **thread safety**, under which different threads can access shared resources without nondeterminism.
  - **Atomic operations**: Operations that execute as a single, indivisible step from the perspective of other threads. No other thread can observe an intermediate state.
  - **Locks / mutexes**: Enforce mutual exclusion. They ensure only *one* thread can enter a critical section at a time.
  - **Semaphores / monitors**: A signaling mechanism to coordinate access among multiple threads and manage a limited pool of resources (allowing $N$ threads to access a resource instead of just $1$).


# Some Practice is Better than None

Knowing the basic concepts above is far from enough. I can't speak for other ML engineers, but for me personally, concurrency problems incredibly hard because I'm familiar with neither the design patterns nor the syntax in the `threading` package. Just when I have solved one problem cleanly, thinking I've got it, the next one requires a completely, or even worse, slightly different approach. 

I heard that even backend engineers find concurrency intimidating. Still, getting some practice is way better than flying blind into OOP interviews that tap into thread-safety and concurrency. We can solve toy problems on LeetCode and a few real-world problems.

# Practice Toy Problems

## Print in Order ([LC 1114](https://leetcode.com/problems/print-in-order/))

In this problem, thread A calls `frist()` (which prints `"first"`), thread B calls `second()` (which prints `"second"`), and thread C calls `third()` (which prints `"third"`) and our goal is to make sure `frist()`  executes before `second()` and `second()` before `third()`. Without synchronization, this order is not guaranteed. 

### Lock-Based Approach
One way to ensure ordering is using locks. We can use two locks as gates to enforce ordering between the three calls:
- `__init__()`: create and immediately acquire `lock1` and `lock2` (both start in the locked state)
- `first()`: thread A executes this call without any blocks; once done, release `lock1`
- `second()`: thread B is blocked by `lock1`; when `lock1` is released, thread B executes this call; once done, release `lock2`
- `third()`: thread C is blocked by `lock2`; when `lock2` is released, thread C executes this call

```python
import threading

class Foo:
    def __init__(self):
        # create 2 locks -> initially unlocked
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()
        
        # immediately lock them
        self.lock1.acquire()
        self.lock2.acquire()

    def first(self, printFirst: 'Callable[[], None]') -> None:
        # thread A runs this. no locks block it.
        printFirst()
        
        # unlock lock1 to signal that first() is done
        self.lock1.release()

    def second(self, printSecond: 'Callable[[], None]') -> None:
        # thread B runs this. blocked until lock1 is released.
        self.lock1.acquire()
        printSecond()
        
        # unlock lock2 to signal that second() is done
        self.lock2.release()

    def third(self, printThird: 'Callable[[], None]') -> None:
        # thread C runs this. blocked until lock2 is released.
        self.lock2.acquire()
        printThird()
```

### Event-Based Approach
A more natural way to ensure ordering is using events. We can use two events as signals to coordinate between the three calls:

- `_init__()`: create `done1` and `done2` (both start in the unset state)
- `first()`: thread A executes this call without any blocks; once done, set `done1`
- `second()`: thread B waits on `done1`; when `done1` is set, thread B executes this call; once done, set `done2`
- `third()`: thread C waits on `done2`; when `done2` is set, thread C executes this call

```python3
import threading

class Foo:
    def __init__(self):
        # create 2 events -> initially unset
        self.done1 = threading.Event()
        self.done2 = threading.Event()

    def first(self, printFirst: 'Callable[[], None]') -> None:
        # thread A runs this. nothing blocks it.
        printFirst()

        # signal that first() is done
        self.done1.set()

    def second(self, printSecond: 'Callable[[], None]') -> None:
        # thread B runs this. waits until done1 is set.
        self.done1.wait()
        printSecond()

        # signal that second() is done
        self.done2.set()

    def third(self, printThird: 'Callable[[], None]') -> None:
        # thread C runs this. waits until done2 is set.
        self.done2.wait()
        printThird()
```

### Compare Two Approaches
Performance-wise, both approaches achieve the same time and space complexity: two synchronization objects, constant overhead. 

Conceptually, the event-based approach is more natural here because this is a thread communication problem --- one thread needs to tell another that something has happened. The lock-based approach is more natural for mutual exclusion problems, where multiple threads compete to access shared data and only one is allowed in at a time.

Let's take another look at the implementations above:
- The lock-based approach repurposes a [mutex](https://stackoverflow.com/questions/34524/what-is-a-mutex) primitive as a gate. Conventionally, the same thread that acquires a lock is the one that releases it — this is how we protect a shared resource from concurrent access. Acquiring in one thread and releasing in another works, but it goes against that convention.
- The event-based approach uses a signaling primitive for a signaling problem: `.set()` means "I'm done," `.wait()` means "I need you to be done." This is the problem to solve here.

## Print FooBar Alternately ([LC 1115](https://leetcode.com/problems/print-foobar-alternately/))

Now let's look at a similar but slightly more complex problem and see if we can replicate the above design patterns ("依葫芦画瓢"). Here, thread A calls `foo` while thread B calls `bar`. We execute the program `n` times and always want thread A to run before thread B. 

### Lock-Based Approach

In the lock-based approach, we create two locks for `foo()` and `bar()`, respectively, during initialization. To make sure `foo()` goes first in each round, we acquire the `bar` lock in the beginning. Before each `foo()` call, we acquire the `foo` lock; once done, we release the `bar` lock. Correspondingly, before each `bar()` call, we acquire the `bar` lock; once done, we release the `foo` lock.

```python3
import threading

class FooBar:
    def __init__(self, n):
        self.n = n

        # create 2 locks -> initially unlocked
        self.foo_lock = threading.Lock()
        self.bar_lock = threading.Lock()
        
        # lock bar_lock right away so that foo() runs first
        self.bar_lock.acquire()

    def foo(self, printFoo: 'Callable[[], None]') -> None:
        for i in range(self.n):
            # wait for permission to print foo (initially unlocked)
            self.foo_lock.acquire()
            printFoo()
            # give permission to print bar
            self.bar_lock.release()

    def bar(self, printBar: 'Callable[[], None]') -> None:
        for i in range(self.n):
            # wait for permission to print bar (initially locked, waiting on foo)
            self.bar_lock.acquire()
            printBar()
            # give permission to print foo again for the next loop iteration
            self.foo_lock.release()
```

### Event-Based Approach

In the event-based approach, we create two events for `foo()` and `bar()` during initialization. We set the `bar` event from the start to make sure `foo()` goes first. Before each `foo()` call, we wait on the `bar` event and then clear it; once done, we set the `foo` event. Correspondingly, before each `bar()` call, we wait on the `foo` event and then clear it; once done, we set the `bar` event.

```python3
import threading

class FooBar:
    def __init__(self, n):
        self.n = n

        # create 2 events -> initially unset
        self.foo_done = threading.Event()
        self.bar_done = threading.Event()
        
        # bar is "done" initially so foo can go first
        self.bar_done.set()

    def foo(self, printFoo: 'Callable[[], None]') -> None:
        for i in range(self.n):
            # wait for bar to finish
            self.bar_done.wait()   
            # reset the gate
            self.bar_done.clear() 
            # thread A executes foo
            printFoo()
            # signal foo is done
            self.foo_done.set()   

    def bar(self, printBar: 'Callable[[], None]') -> None:
        for i in range(self.n):
            # wait for foo to finish
            self.foo_done.wait()  
            # reset the gate
            self.foo_done.clear() 
            # thread B executes bar
            printBar()
            # signal bar is done
            self.bar_done.set()    
```

### Compare Two Approaches

Perhaps surprisingly, the lock-based approach is actually more natural and efficient in this case. With events, each turn requires three steps: `.wait()` → `.clear()` → `.set()`. The wait and clear are conceptually one action ("consume the signal"), but must be expressed as two separate calls. With locks, each turn requires only two steps: `.acquire()` → `.release()`, because `acquire()` both waits for the gate to open and closes it behind you in one atomic operation.

This is in contrast to LC 1114, where the signal was one-shot, which is what events are designed for. Here in LC 1115, every signal needs to be consumed and reset, which is exactly what `acquire()` does by design. The takeaway is, the "right" primitive depends on the problem pattern: one-shot signaling → events; alternating turns → locks.


## Print Zero Even Odd ([LC 1116](https://leetcode.com/problems/print-zero-even-odd/description/))

Let's solve one last printing problem. Here, threads share the instance of `ZeroEvenOdd`. Thread A calls `zero()`, thread B calls `even()`, and thread C calls `odd()`. In `n` turns, we output the series `"010203040506..."` of length `2n`. Each turn starts with `0` and alternates between odd and even: `0, 1, 0, 2, 0, 3, ...`. 

### Lock-Based Approach

Since this problem requires alternating turns, it's more natural to use locks than events to avoid repeated `.wait()` and `.clear()`. We create three locks for each thread and lock all but the `zero()` lock during initialization. Before calling a function, we first acquire the lock for that function and then release the lock on the next function.

```python
import threading

class ZeroEvenOdd:
    def __init__(self, n):
        self.n = n

        # create 3 locks -> initially unlocked
        self.zero_lock = threading.Lock()
        self.odd_lock = threading.Lock()
        self.even_lock = threading.Lock()

        # only zero should go first -> lock other 2 locks
        self.odd_lock.acquire()
        self.even_lock.acquire()

    def zero(self, printNumber):
        for i in range(1, self.n + 1):
            # wait for permission to print 0 (initially unlocked)
            self.zero_lock.acquire()
            printNumber(0)
            # decide what to print next -> release corresponding lock
            if i % 2 == 1:
                self.odd_lock.release()
            else:
                self.even_lock.release()

    def odd(self, printNumber):
        for i in range(1, self.n + 1, 2):
            # wait for permission to print odd number (initially locked)
            self.odd_lock.acquire()
            printNumber(i)
            # the next number is always 0 -> release the zero lock
            self.zero_lock.release()

    def even(self, printNumber):
        for i in range(2, self.n + 1, 2):
            # wait for permission to print even number (initially locked)
            self.even_lock.acquire()
            printNumber(i)
            # the next number is always 0 -> release the zero lock
            self.zero_lock.release()
```

### Introducing Semaphores

We can use [semaphores](https://en.wikipedia.org/wiki/Semaphore_(programming)) to simplify the initialization. A semaphore is a counter that blocks when the count is 0 and decrements when acquired. Unlike a lock, it's designed to be acquired by one thread and released by another, making it natural for cross-thread signaling.

With semaphores, we use the initial count to declare the starting state: `Semaphore(0)` starts blocked, `Semaphore(1)` starts open. The rest of the solution is nearly identical to before --- just replace `Lock()` + `.acquire()` with `Semaphore(0)`.


```python
import threading

class ZeroEvenOdd:
    def __init__(self, n):
        self.n = n
        
        # zero() can run immediately
        self.zero_sem = threading.Semaphore(1)
        # block even() and odd()
        self.odd_sem = threading.Semaphore(0)
        self.even_sem = threading.Semaphore(0)

    def zero(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1):
            # wait for a zero() permit
            self.zero_sem.acquire()
            printNumber(0)
            
            # then give permission to the next call
            if i % 2 == 1:
                self.odd_sem.release()
            else:
                self.even_sem.release()

    def even(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(2, self.n + 1, 2):
            # wait for an even() permit
            self.even_sem.acquire()
            printNumber(i)
            # the next call is always zero()
            self.zero_sem.release()

    def odd(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1, 2):
            # wait for an odd() permit
            self.odd_sem.acquire()
            printNumber(i)
            # the next call is always zero()
            self.zero_sem.release()
```

### Comparing Two Approaches

There is no performance difference between the two approaches, and most of the code is identical. The difference lies in semantic. A lock represents *ownership* --- the thread that acquires it is expected to be the one that releases it. In our implementation, we violate that contract by acquiring in one thread and releasing in another. A semaphore represents *permits* --- it tracks how many threads are allowed to proceed, with no expectation that the same thread that consumed a permit is the one that grants the next one. 

In our problem, one thread finishes its work and grants permission for another thread to go --- semaphores are the more fitting primitive. A highly similar problem to practice is [1195. Fizz Buzz Multithreaded](https://leetcode.com/problems/fizz-buzz-multithreaded/description/).

## Building H2O ([LC 1117](https://leetcode.com/problems/building-h2o/description/))

To further appreciate the power of semaphores, let's look at a chemistry-inspired problem: in order to form a water molecule, we need exactly 2 hydrogen and 1 oxygen atoms. We can use `Semaphore(2)` and `Semaphore(1)` for `hydrogen()` and `oxygen()` to control the ratio. Then, we can use `Barrier(3)` to ensure the right amount of atoms before releasing a molecule.

Inside each call, the flow is the same: acquire the corresponding semaphore to claim a spot in the current group, wait at the barrier until all 3 threads (2 `H` + 1 `O`) have arrived, release the atom, and then release the semaphore to open a permit for the next group.

```python
import threading

class H2O:
    def __init__(self):
        # semaphores control the ratio: 2 H, 1 O per group
        self.hydrogen_sem = threading.Semaphore(2)
        self.oxygen_sem = threading.Semaphore(1)
        # barrier ensures all 3 threads arrive before any proceed
        self.barrier = threading.Barrier(3)

    def hydrogen(self, releaseHydrogen: 'Callable[[], None]') -> None:
        # only 2 H threads can enter at a time
        self.hydrogen_sem.acquire()    
        # wait for the group of 3 to form
        self.barrier.wait()     
        # release H for the current group      
        releaseHydrogen()
        # open a permit for the next group
        self.hydrogen_sem.release()    

    def oxygen(self, releaseOxygen: 'Callable[[], None]') -> None:
        # only 1 O thread can enter at a time
        self.oxygen_sem.acquire()      
        # wait for the group of 3 to form
        self.barrier.wait()           
        # release O for the current group      
        releaseOxygen()
        # open a permit for the next group
        self.oxygen_sem.release()     
```

## The Dining Philosophers ([LC 1226](https://leetcode.com/problems/the-dining-philosophers/description/))

Previous problems don't require a deep understanding of concurrency. LC has this eccentric problem that sheds light on deadlock prevention. In this problem, 5 philosophers sit at a round table in clockwise order, with a fork between each pair. To eat, a philosopher must pick up **two forks** (who does that? 😂): the one on their left and the one on their right. After eating, the philosopher puts down both for others to use. 

Our task is to implement the `wantsToEat()` function. Each call to `wantsToEat` represents one philosopher's complete eating cycle: wait for permission, pick up both forks, eat, put down both forks. No philosopher should starve (i.e., everyone eventually gets both forks).

The danger is **deadlock**: imagine all 5 philosophers pick up their left fork at once. Now everyone is holding their left fork while waiting for the right fork --- but the right fork each philosopher needs is the left fork held by their neighbor. Everyone is stuck waiting and will starve.

To prevent this, we limit the number of philosophers simultaneously reaching for forks to 4. If at most 4 philosophers compete for 5 forks, at least one of them is guaranteed to get both forks. That philosopher eats, puts down their forks, and frees resources for others to eat.

```python
import threading

class DiningPhilosophers:
    def __init__(self):
        # each fork is a lock — true mutual exclusion (only one philosopher can hold it)
        self.forks = [threading.Lock() for _ in range(5)]
        # allow at most 4 philosophers to try at once — prevents deadlock
        self.limit = threading.Semaphore(4)

    def wantsToEat(self, philosopher, pickLeftFork, pickRightFork, eat, putLeftFork, putRightFork):
        # index of current philosopher
        left = philosopher
        # index of their neighbor
        right = (philosopher + 1) % 5

        # at most 4 philosopher can try to eat
        self.limit.acquire()         

        # pick up left fork and right fork
        self.forks[left].acquire()    
        self.forks[right].acquire()   

        # perform the eating action sequence
        pickLeftFork()
        pickRightFork()
        eat()
        putLeftFork()
        putRightFork()

        # pick up left fork and right fork
        self.forks[right].release()   
        self.forks[left].release()   

        self.limit.release()        
```

A similar problem is [1279. Traffic Light Controlled Intersection](https://leetcode.com/problems/traffic-light-controlled-intersection/description/?).

# Practice Real-World Problems

## Thread-Safe Time-Based KV Store

As ML engineers, we work with feature stores all the time. For a given feature, its value may change over time. It's common to use a time-based KV store to store different values at different timestamps.

### Vanilla Version ([LC 981](https://leetcode.com/problems/time-based-key-value-store/description/))

In a vanilla implementation, we can store each key's values in an append-only log, with each entry being a `(timestamp, value)` tuple. The `set()` method appends a new entry to a given key's log. When calling `get()` for a given key at a particular timestamp, we run a binary search to find this key's value at or before this timestamp.

```python
class TimeMap:
    def __init__(self):
        # key -> list of (timestamp, value)
        self.store = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.store:
            self.store[key] = []
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        # cannot get value if key doesn't exist
        if key not in self.store:
            return ""
        # get list of (timestamp, value) for given key
        arr = self.store[key]

        # find the rightmost timestamp <= given timestamp
        l, r = 0, len(arr) - 1

        while l <= r:
            mid = (l + r) // 2
            # record valid candidate -> try to find a later one
            if arr[mid][0] <= timestamp:
                l = mid + 1
            # later than given timestamp -> must search left
            else:
                r = mid - 1

        return arr[r][1] if r != -1 else ""
```

The problem is, feature jobs (streaming or batch) regularly update the feature store. If a feature job calls `set()` to write a new value for a key while an inference job calls `get()` to fetch the value of the same key, we may get wrong results. This is where thread-safety comes in.

### Per-Key Locks + Global Lock

The simplest fix is to use a global lock: lock the feature store during writes, release it afterwards. But this is silly --- if we're writing a value for, say `userId = 123`, there's no reason to block reads for any other key. We'd be serializing the entire system over one key's update.

Instead, we can create a lock per key. Each key gets its own lock, so a write to `userId = 123` only blocks reads and writes to that key. Operations on different keys proceed in parallel without contention.

Oh, that almost works, if we overlook the fact that the `locks` dictionary itself is a shared state. What if two threads call `set()` or `get()` for the same key at the same time? They would both try to simultaneously create a lock for that key, creating a race condition. To address this issue, we can use a global lock to protect per-key lock creation. When one thread is inside `_get_lock()`, the global lock is held; only after it finishes is the lock released for other threads.

```python
import threading

class ConcurrentTimeMap:
    def __init__(self):
        self.store = {}  # key -> list of (timestamp, value)
        self.locks = {}  # per-key locks: key -> threading.Lock()
        self.global_lock = threading.Lock()  # protects _get_lock()

    def _get_lock(self, key: str):
        with self.global_lock:
            # add a new per-key lock
            if key not in self.locks:
                self.locks[key] = threading.RLock()
            return self.locks[key]

    def set(self, key: str, value: str, timestamp: int) -> None:
        # get per-key lock
        lock = self._get_lock(key)
        with lock:
            # set value for given key at given timestamp
            if key not in self.store:
                self.store[key] = []
            self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        # get per-key lock
        lock = self._get_lock(key)
        with lock:
            # get value for given key <= given timestamp
            if key not in self.store:
                return ""
            arr = self.store[key]
            l, r = 0, len(arr) - 1

            while l <= r:
                mid = (l + r) // 2
                if arr[mid][0] <= timestamp:
                    l = mid + 1
                else:
                    r = mid - 1

            return arr[r][1] if r != -1 else ""
```

## Thread-Safe Message Queue ([LC 1188](https://leetcode.com/problems/design-bounded-blocking-queue/description/))

Message queues are ubiquitous in event-driven distributed systems. In a real-time feature streaming job, for instance, when a user clicks on an ad, it publishes an event to a message queue, which triggers feature updates. Web-scale events come in massive volumes. To scale up, modern systems typically use distributed message queues like Kafka.

A vanilla queue that just does `enqueue()` and `dequeue()` without any thread-safety protections creates a few problems in the concurrent setting above. First, the queue is shared state. Without a lock, concurrent reads and writes can create race conditions. Second, how do we ensure the right behavior when the queue is empty or full? If the queue is full, the producer should wait until space opens up. If empty, the consumer should wait until a new event arrives. 

To address both problems, we can use a global lock on the queue and two `Condition` variables --- `not_full` and `not_empty` --- that let threads sleep until the queue state changes. A producer waits on `not_full` to write a message, then wakes sleeping consumers via `not_empty.notify()` to consume it. A consumer does the reverse: waits on `not_empty` to read a message, then wakes sleeping producers via `not_full.notify()` to write new ones.


```python
from collections import deque
import threading


class BoundedBlockingQueue(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = deque()
        # global lock for the shared queue
        self.lock = threading.Lock()
        # condition variables for boundary behavior
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)


    def enqueue(self, element: int) -> None:
        # acquire the underlying lock via the not_full condition
        with self.not_full:
            # if queue is full, release lock and sleep until notified
            while len(self.queue) >= self.capacity:
                self.not_full.wait()
            # queue has space — enqueue element
            self.queue.appendleft(element)
            # wake up a sleeping consumer, if any
            self.not_empty.notify()


    def dequeue(self) -> int:
        # acquire the underlying lock via the not_empty condition
        with self.not_empty:
            # if queue is empty, release lock and sleep until notified
            while len(self.queue) == 0:
                self.not_empty.wait()
            # queue has elements — dequeue
            element = self.queue.pop()
            # wake up a sleeping producer, if any
            self.not_full.notify()
            return element


    def size(self) -> int:
        with self.lock:
            return len(self.queue)

```

## Multithreading Web Crawlers ([LC 1242](https://leetcode.com/problems/web-crawler-multithreaded/description/))

Web crawlers are the lifeline of search engines and RAG-based chatbots. A single-threaded crawler would take forever to index the web. We can use a [thread pool](https://en.wikipedia.org/wiki/Thread_pool) to crawl webpages in parallel. In this example problem, the core algorithm is BFS. We start with a seed URL, submit it to a thread pool, and collect newly discovered URLs as each page finishes loading. Each new URL is submitted back to the pool, and the process repeats until no more URLs are discovered. 

We may run into concurrency problems with the `visited` set. Two threads may both check the same URL, see it as unvisited, and crawl it, creating a race condition. To prevent this, we can use a global lock to protect `visited`. The actual HTTP call (`htmlParser.getUrls()`) happens outside the lock so threads can fetch pages in parallel.

```python
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        hostname = urlparse(startUrl).hostname

        # track visited URLs to avoid duplicate crawls
        visited = set()
        visited.add(startUrl)

        # lock protects the visited set — nothing else
        lock = threading.Lock()

        result = [startUrl]

        def crawl_url(url):
            """Fetch a URL and return new same-hostname URLs we haven't seen."""
            new_urls = []
            # this HTTP call is outside the lock — threads fetch in parallel
            for link in htmlParser.getUrls(url):
                # skip URLs on different hostnames
                if urlparse(link).hostname != hostname:
                    continue
                # atomic check-and-add: only one thread can register a URL
                with lock:
                    if link not in visited:
                        visited.add(link)
                        new_urls.append(link)
            return new_urls

        # thread pool with 8 workers to crawl pages in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            # seed the pool with the starting URL
            futures = {executor.submit(crawl_url, startUrl)}

            while futures:
                # block until any one task finishes
                done = next(as_completed(futures))
                futures.discard(done)

                # each finished task returns new URLs — submit them to the pool
                for url in done.result():
                    result.append(url)
                    futures.add(executor.submit(crawl_url, url))

        return result
```

# Concurrency 101 Resources

## Tutorials
- Educative: [Multithreading and Concurrency Fundamentals](https://www.educative.io/blog/multithreading-and-concurrency-fundamentals)
- MIT 6.031: [Reading 19: Concurrency](https://web.mit.edu/6.031/www/fa17/classes/19-concurrency/)
- Real Python: [Speed Up Your Python Program With Concurrency](https://realpython.com/python-concurrency/#exploring-concurrency-in-python)

## Collections
- LeetCode: [Concurrency Collection](https://leetcode.com/problem-list/concurrency/)
- LeetCode Discussions: [Surgical Strike on Concurrency and Multithreading](https://leetcode.com/discuss/post/7605788/surgical-strike-on-concurrency-and-multi-nghz/)
