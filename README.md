# flux-fine-tuner

## Development

Before submitting a PR, format the code and run the linter locally.

```shell
pip install -r requirements-test.txt  # only need to do this once
ruff format
ruff check
```

## Deploy to production

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
