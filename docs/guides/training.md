# Training

Jumanji provides two example agents to get you started with training on Jumanji environments:
- Random agent
- A2C agent

## Configuration

In each environment-specific config YAML file, you will see a "training" section like below:

```yaml
training:
    num_epochs: 1000
    num_learner_steps_per_epoch: 50
    n_steps: 20
    total_batch_size: 64
```
Here,
- `num_epochs` corresponds to the number of data points in your plots. An epoch can be thought
as an iteration.
- `num_learner_steps_per_epoch` is the number of learner steps that happen in each epoch.
After every learner step, the A2C agent's policy is updated.
- `n_steps` is the sequence length (consecutive environment steps in a batch).
- `total_batch_size` is the number of environments that are run in parallel.

So in the above example,
- 64 environments are running in parallel.
- Each of these 64 environments run 20 environment steps. After this, the agent's policy is
updated via SGD. This constitutes a single learner step.
- 50 such learner steps are done for the epoch in question. After this, evaluation is done
using the updated policy.
- The above procedure is done for 1000 epochs.

## Evaluation
Two types of evaluation are recorded:
- Stochastic evaluation (same policy used during training)
- Greedy evaluation (argmax over the action logits)
