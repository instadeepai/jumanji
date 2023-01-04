# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple, Optional

import chex
import jax.random


class Observation(NamedTuple):
    """The observation that the agent sees.

    operations_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    operations_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    operations_mask: for each job, indicates which operations remain to be scheduled. False if the
        op has been scheduled or if the op was added for padding, True otherwise. The first True in
        each row (i.e. each job) identifies the next operation for that job.
    machines_job_ids: for each machine, it specifies the job currently being processed.
        Note that -1 means no-op in which case the remaining time until available is always 0.
    machines_remaining_times: for each machine, it specifies the number of time steps until
        available.
    action_mask: for each machine, it indicates which jobs (or no-op) can legally be scheduled.
        The last column corresponds to no-op.
    """

    operations_machine_ids: chex.Array  # (num_jobs, max_num_ops)
    operations_durations: chex.Array  # (num_jobs, max_num_ops)
    operations_mask: chex.Array  # (num_jobs, max_num_ops)
    machines_job_ids: chex.Array  # (num_machines,)
    machines_remaining_times: chex.Array  # (num_machines,)
    action_mask: chex.Array  # (num_machines, num_jobs + 1)


class State(NamedTuple):
    """The environment state containing a complete description of the job shop scheduling problem.

    operations_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    operations_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    operations_mask: for each job, indicates which operations remain to be scheduled. False if the
        op has been scheduled or if the op was added for padding, True otherwise. The first True in
        each row (i.e. each job) identifies the next operation for that job.
    machines_job_ids: for each machine, it specifies the job currently being processed.
        Note that -1 means no-op in which case the remaining time until available is always 0.
    machines_remaining_times: for each machine, it specifies the number of time steps until
        available.
    action_mask: for each machine, it indicates which jobs (or no-op) can legally be scheduled.
        The last column corresponds to no-op.
    current_timestep: used to track time, which is necessary for updating scheduled_times.
    scheduled_times: for each job, it specifies the time at which each operation was scheduled.
        Note that -1 means the operation has not been scheduled yet.
    key: not used inside the environment, but may be used e.g. in wrappers.
    """

    operations_machine_ids: chex.Array  # (num_jobs, max_num_ops)
    operations_durations: chex.Array  # (num_jobs, max_num_ops)
    operations_mask: chex.Array  # (num_jobs, max_num_ops)
    machines_job_ids: chex.Array  # (num_machines,)
    machines_remaining_times: chex.Array  # (num_machines,)
    action_mask: Optional[chex.Array]  # (num_machines, num_jobs + 1)
    current_timestep: int
    scheduled_times: chex.Array  # (num_jobs, max_num_ops)
    key: chex.PRNGKey = jax.random.PRNGKey(0)
