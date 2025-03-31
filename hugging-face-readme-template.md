---
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
language:
- en
tags:
- flux
- diffusers
- lora
- replicate
base_model: "black-forest-labs/FLUX.1-dev"
pipeline_tag: text-to-image
# widget:
#   - text: >-
#       prompt
#     output:
#       url: https://...
$instance_prompt
---

# $title

<Gallery />

## About this LoRA

This is a [LoRA](https://replicate.com/docs/guides/working-with-loras) for the FLUX.1-dev text-to-image model. It can be used with diffusers or ComfyUI.

It was trained on [Replicate](https://replicate.com/) using AI toolkit: https://replicate.com/ostris/flux-dev-lora-trainer/train

$trigger_section

## Run this LoRA with an API using Replicate

```py
import replicate

input = {
    "prompt": "$trigger_word",
    "lora_weights": "https://huggingface.co/$repo_id/resolve/main/lora.safetensors"
}

output = replicate.run(
    "black-forest-labs/flux-dev-lora",
    input=input
)
for index, item in enumerate(output):
    with open(f"output_{index}.webp", "wb") as file:
        file.write(item.read())
```

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('$repo_id', weight_name='lora.safetensors')
image = pipeline('$trigger_word').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

$training_details

## Contribute your own examples

You can use the [community tab](https://huggingface.co/$repo_id/discussions) to add images that show off what you’ve made with this LoRA.
