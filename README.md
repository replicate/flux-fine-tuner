# flux-fine-tuner

This is a [Cog](https://cog.run) training model that creates LoRA-based fine-tunes for the [FLUX.1](https://replicate.com/blog/flux-state-of-the-art-image-generation) family of image generation models.

It's live at [replicate.com/ostris/flux-dev-lora-trainer](https://replicate.com/ostris/flux-dev-lora-trainer).

It also includes code for running inference with a fine-tuned model.

## Features

- Automatic image captioning during training
- Image generation using the LoRA (inference)
- Optionally uploads fine-tuned weights to Hugging Face after training
- Automated test suite with [cog-safe-push](https://github.com/replicate/cog-safe-push) for continuous deployment
- Weights and biases integration

## Getting Started

If you're looking to create your own fine-tuned model on Replicate, you don't need to do anything with this codebase.

Check out these guides to get started:

👉 [Fine-tune Flux to create images of yourself](https://replicate.com/blog/fine-tune-flux-with-faces)

👉 [Fine-tune Flux with an API](https://replicate.com/blog/fine-tune-flux-with-an-api)

## Contributing

If you're here to help improve [the trainer that Replicate uses to fine-tune Flux models](https://replicate.com/ostris/flux-dev-lora-trainer), you've come to the right place.

Check out the [contributing guide](CONTRIBUTING.md) to get started.

## Credits

This project is based on the [ai-toolkit](https://github.com/ostris/ai-toolkit) project, which was created by [@ostris](https://github.com/ostris). ❤️

## License

The code in this repository is licensed under the [Apache-2.0 License](LICENSE).

The [ai-toolkit](https://github.com/ostris/ai-toolkit) project is licensed under the [MIT License](https://github.com/ostris/ai-toolkit/blob/main/LICENSE).

Flux Dev falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

`FLUX.1 [dev]` fine-tuned weights and their outputs are non-commercial by default, but can be used commercially when running on Replicate.

Flux Schnell falls under the [Apache-2.0 License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).
