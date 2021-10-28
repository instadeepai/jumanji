from gym.envs.registration import register

# register gym environments
entry_pt = (
    "jumanji.environments.fetch_environments.base_fetch_env:BaseMultiTaskFetchEnv"
)
for num_blocks in range(1, 10):
    entry_point = entry_pt
    register(
        id=f"BaseMultiTaskFetchArm-{num_blocks}blocks-v1",
        entry_point=entry_point,
        kwargs=dict(num_blocks=num_blocks),
        max_episode_steps=50,
    )

entry_pt = "jumanji.environments.exploration_environments.point_maze:PointMaze"
register(id="PointMaze-v1", entry_point=entry_pt)

entry_pt = "jumanji.environments.exploration_environments.ant_trap:AntTrap"
register(id="AntTrap-v1", entry_point=entry_pt, max_episode_steps=1000)

entry_pt = "jumanji.environments.exploration_environments.humanoid_trap:HumanoidTrap"
register(id="HumanoidTrap-v1", entry_point=entry_pt, max_episode_steps=1000)

entry_pt = "jumanji.environments.exploration_environments.ant_maze:AntMaze"
register(id="AntMaze-v1", entry_point=entry_pt, max_episode_steps=3000)
