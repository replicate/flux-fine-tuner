# Contributing

Thanks for taking the time to contribute to this project. 🙏🏼

This document covers how to get started working on this project, and proposing changes to make it better.

## Start with an issue

Rather than jump straight to an implementation in a pull request, please start with an issue.

This gives everyone a chance to weigh in on the idea, and for you to get some early feedback. It also helps avoid wasting time on something that the maintainers don't think is a good fit for this project.

If you've found a bug, please include steps to reproduce it.

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