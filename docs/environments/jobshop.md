# JobShop Environment

<p align="center">
        <img src="../img/jobshop.gif" height="300"/>
</p>

We provide here a JAX jittable implementation of the [job shop scheduling problem](https://developers.google.com/optimization/scheduling/job_shop).
It is NP-hard and one of the most well-known combinatorial optimisation problems. The problem formulation is:
- $N$ **jobs**, each consisting of a sequence of **operations**, need to be scheduled on $M$ machines.
- For each job, its operations must be processed **in order**. This is called the **precedence constraints**.
- Only one operation in a job can be processed at any given time.
- A machine can only work on one operation at a time.
- Once started, an operation must run to completion.

The goal of the agent is to determine the schedule that minimises the time needed to process all the jobs.
The length of the schedule is also known as its _makespan_.

## Observation
The **observation** seen by the agent is a `NamedTuple` containing the following:
- `operations_machine_ids`: for each job, it specifies the machine each op must be processed on.
    Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    Has shape `(num_jobs, max_num_ops)`
- `operations_durations`: for each job, it specifies the processing time of each operation.
    Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    Has shape `(num_jobs, max_num_ops)`
- `operations_mask`: for each job, indicates which operations remain to be scheduled. False if the
    op has been scheduled or if the op was added for padding, True otherwise. The first True in
    each row (i.e. each job) identifies the next operation for that job.
    Has shape `(num_jobs, max_num_ops)`
- `machines_job_ids`: for each machine, it specifies the job currently being processed.
    Note that -1 means no-op in which case the remaining time until available is always 0.
    Has shape `(num_machines,)`
- `machines_remaining_times`: for each machine, it specifies the number of time steps until
    available.
    Has shape `(num_machines,)`
- `action_mask`: for each machine, it indicates which jobs (or no-op) can legally be scheduled.
    The last column corresponds to no-op.
    Has shape `(num_machines, num_jobs + 1)`

## Action
An action consists of the following: for each machine, decide which job (or no-op) to schedule at the current time step.
The action is represented as a 1-dimensional array of length `num_machines`.
For example, suppose we have $M=5$ machines and there are $N=10$ jobs. A legal action might be
```python
action = [4, 7, 0, 10, 10]
```
This action represents scheduling
- Job 4 on Machine 0,
- Job 7 on Machine 1,
- Job 0 on Machine 2,
- No-op on Machine 3,
- No-op on Machine 4.

As such, the action is multidimensional and can be thought of as each machine (each agent) "deciding" which job (or no-op) to schedule.
Importantly, the action space is a product of the marginal action space of each agent (machine).

The rationale for having a no-op is the following:
- A machine might be busy processing an operation, in which case a no-op is the only allowed action for that machine.
- There might not be any jobs that can be scheduled on a machine.
- There may be scenarios where waiting to schedule a job via one or more no-op(s) ultimately minimises the makespan.

## Reward
The reward function is configurable, but defaults to a dense reward where -1 is given for each time step. Another natural choice
would be a sparse reward function with a reward of minus the episode length.

An episode will terminate in any of the following three scenarios:
- The schedule has finished: all operations for every job have been processed.
- The agent ignores the action mask and does an illegal action.
- All machines are idle and do a no-op at the same time.

## Registered Versions 📖
- `JobShop-v0`
