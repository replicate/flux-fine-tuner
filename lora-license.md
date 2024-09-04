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
[instance_prompt]
---

# [title]

<!-- <Gallery /> -->

Trained on Replicate using:

https://replicate.com/ostris/flux-dev-lora-trainer/train

[trigger_section]

## Use this lora with Replicate

[Grab your replicate token](https://replicate.com/account)

```bash
pip install replicate
export REPLICATE_API_TOKEN=r8_*************************************
```

```py
import replicate

output = replicate.run(
    "sdxl-black-light@sha256:0b682d5744e86e988216141edd6a99be821941fd1a49a64786ad47fa48c33a95",
    input={"prompt": "your prompt"}
)
print(output)
```

You may also do inference via the API with Node.js or curl, and locally with Cog and Docker, [check out the Replicate API page for this model](https://replicate.com/fofr/sdxl-black-light/api)


## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('[hf_repo_id]', weight_name='lora.safetensors')
image = pipeline('your prompt').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)
