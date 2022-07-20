# Contributing

When contributing to Jumanji, please first discuss the change you wish to make via an issue and then
create a Merge Request (MR) to start implementing your changes (MR in draft mode). When the code is
ready for review, please mark the MR as ready and ping a reviewer for review.

Please note we have a code of conduct and ask that you follow it in all your interactions with the project.

## Merge Request Process

Before marking an MR as ready for review, please make sure the following steps are implemented.

1. Changes are documented in the MR description and in the README if applicable. Make sure to close
related issues in the MR description by using `Closes #[ISSUE_ID]` or similar, see [GitLab's
managing issue documentation](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically) for more details.
2. MR name and squash commit message are modified according to
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
3. If implementing a new Jax environment, please make sure to do the following:
   1. Implement the JaxEnv API: step, reset, specs, etc...
   2. Implement unit tests for every function used by the environment, including a
   `test_[your_env]__does_not_smoke` test that runs a `JaxEnvironmentLoop` to test compilation.
   3. Add an environment README in the `jumanji/your_env` folder that describes the environment
   4. Add an image (or gif) in the readme above. Images are located in `docs/img`.
   5. Update the documentation api in `docs/api` to add your environment to the doc.
   6. Optional, add a `run_random_agent_[your_env].py` script in `examples/` to benchmark your
   environment.

## Code of Conduct

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
