# Contributing

Thanks for taking the time to contribute to this project. 🙏🏼

This document covers how to get started working on this project, and proposing changes to make it better.

## Start with an issue

Rather than jump straight to an implementation in a pull request, please [check out the open issues first](https://github.com/replicate/flux-fine-tuner/issues) to see if there's already a discussion around the feature you want to add or the bug you want to fix. If no issue exists, please [open a new issue](https://github.com/replicate/flux-fine-tuner/issues/new) to discuss the change you want to make.

This process helps avoid wasting time working on something that may not fit the needs of the project.

## Development

Once you've found an issue to work on, it's time to jump into making changes.

Before submitting a PR, format the code and run the linter locally.

```shell
pip install -r requirements-test.txt  # only need to do this once
ruff format
ruff check
```

## Open a pull request

We use the [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow) model for managing changes.

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Add or update tests, if applicable
5. Add or update documentation, if applicable
6. Open a pull request, and include a reference to the original issue