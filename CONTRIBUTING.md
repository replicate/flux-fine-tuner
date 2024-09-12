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

---

## Deploy to production

> [!NOTE]
> 👇 The rest of this document is for internal Replicate use. It contains instructions for deploying the model to production and updating hotswap bases. If you're an external contributor, you can safely ignore it.

### Push model

Manually trigger the [Github Push action](https://github.com/replicate/flux-fine-tuner/actions/workflows/push.yaml). Uncheck the "Test only, without pushing to prod" button. You might also have to uncheck the "Compare outputs..." button.

The push action takes half an hour since it tests both training and inference. But it's worth doing to be safe.

Once you've deployed to production, make a test fine-tune and run predictions on the trained model, just to be super sure it works.

### Update hotswap bases

When you've tested the pushed model, you need to update all existing fine-tuned versions to use the predictor you just pushed.

First, only update a single model or a subset of models to test that this step works.

In your local checkout of [web](https://github.com/replicate/web), run

```shell
script/manage-prod update_hotswap_base_version \
    --from-model ostris/flux-dev-lora-trainer \
    --to-latest \
    --trained-version-filter="<your-username>/<your-model-name>"
```

This is a dry run to list the versions that will be updated. The `--trained-version-filter` can be just "<your-username>" if you want to test all your models.

When you're happy to with the list of models that will be updated, run

```shell
script/manage-prod update_hotswap_base_version \
    --from-model ostris/flux-dev-lora-trainer \
    --to-latest \
    --trained-version-filter="<your-username>/<your-model-name>" \
    --force
```

Now you can test that the predictor works for the updated model(s).

When you're happy with that, do the same thing but for all models. First, a dry run:

```shell
script/manage-prod update_hotswap_base_version \
    --from-model ostris/flux-dev-lora-trainer \
    --to-latest \
```

And then actually update all the trained Flux versions:

```shell
script/manage-prod update_hotswap_base_version \
    --from-model ostris/flux-dev-lora-trainer \
    --to-latest \
    --force
```

This process is being improved -- soon it will be possible to configure base versions in Django admin.
