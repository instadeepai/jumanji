# Contribution Guidelines

Thank you for considering contributing to Jumanji! We love welcoming contributions from fellow programmers. There are several ways to contribute to the library, e.g. raising issues, writing documentation and code. This guide will help you in that process.

## Pull Request Checklist

Before sending your pull requests, make sure you do the following:
- Read the [contributing guidelines](#Contributing-Code).
- Read the [Code of Conduct](#Code-of-Conduct).
- Ensure you have signed the [Contributor License Agreement](#Contributor-License-Agreement) (CLA).
- Check if your changes are consistent with the [guidelines](#General-guidelines-and-philosophy-for-contribution) and the [Coding Style](#Coding-Style).
- Run the [unit tests](#Testing).
- Run the [pre-commits](#Pre-Commit).

## Contributing Code

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose.
Consult [GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
for more information on using pull requests.

Before sending your pull request for review, make sure your changes are consistent with the guidelines and follow the coding style and code of conduct.

### General guidelines and philosophy for contribution
- Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
- When you contribute a new feature to Jumanji, the maintenance burden is (by default) transferred to the Jumanji team. This means that the benefit of the contribution must be compared against the cost of maintaining the feature.
- Keep API compatibility in mind when you change code. Non-backward-compatible API changes will not be made if they don't greatly improve the library.
- As every PR requires CI testing, we discourage submitting PRs to fix one typo, one warning, etc. We recommend fixing the same issue at the file level at least (e.g.: fix all typos in a file, fix all compiler warnings in a file, etc.)

#### When contributing a new environment

When contributing a new environment, make sure to do the following:
1. Implement all the functions of the Environment abstraction: step, reset, specs, etc.
2. Implement unit tests for every function used by the environment, including a `test_[your_env]__does_not_smoke` using the testing utils.
3. Add an environment README in the `docs/environments/` folder describing the environment you implemented.
4. Add an image (or gif) in the readme above. Images are located in `docs/img/`.
5. Update the documentation api in `docs/api/environments/` to add your environment to the doc.
6. Update the `mkdocs.yml` file to include the newly added markdown files.


### Coding Style
In general, we follow the [Google Style Guide](https://google.github.io/styleguide/pyguide.html).
In addition, to guarantee the quality and uniformity of the code, we use various linters:

- [Black](https://black.readthedocs.io/en/stable/#) is a deterministic code formatter that is compliant with PEP8 standards.
- [Isort](https://pycqa.github.io/isort/) sorts imports alphabetically and separates them into sections.
- [Flake8](https://flake8.pycqa.org/en/latest/) is a library that wraps PyFlakes and PyCodeStyle. It is a great toolkit for checking your codebase against coding style (PEP8), programming, and syntax errors. Flake8 also benefits from an ecosystem of plugins developed by the community that extend its capabilities. You can read more about Flake8 plugins on the documentation and find a curated list of plugins here.
- [MyPy](https://mypy.readthedocs.io/en/stable/#) is a static type checker that can help you detect inconsistent typing of variables.


#### Pre-Commit
To help in automating the quality of the code, we use [pre-commit](https://pre-commit.com/), a framework that manages the installation and execution of git hooks that will be run before every commit. These hooks help to automatically point out issues in code such as formatting mistakes, unused variables, trailing whitespace, debug statements, etc. By pointing these issues out before code review, it allows a code reviewer to focus on the architecture of a change while not wasting time with trivial style nitpicks. Each commit should be preceded by a call to pre-commit to ensure code quality and formatting. The configuration is in .pre-commit-config.yaml and includes Black, Flake8, MyPy and checks for the yaml formatting, trimming trailing whitespace, etc.
Try running: `pre-commit run --all-files`. All linters must pass before committing your change.

### Code of Conduct
We ask you to help us develop a positive working environment. Behaviours that contribute to it include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

### Testing
Please make sure that your PR passes all tests by running [pytest](https://docs.pytest.org/en/latest/) on your local machine.
Also, you can run only tests that are affected by your code changes, but you will need to select them manually.

### Contributor License Agreement
Contributions to this project must be accompanied by a Contributor License Agreement.
You (or your employer) retain the copyright to your contribution, this simply gives us permission to use and
redistribute your contributions as part of the project. Head over to https://cla.developers.google.com/ to
see your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it again.
