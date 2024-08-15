from cog import BasePredictor, Input, Path
import os
import re
import time
import torch
import subprocess
from typing import List
import base64
import tempfile
import tarfile
from io import BytesIO
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
from safetensors.torch import load_file
from weights import WeightsDownloadCache
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

# MODEL_URL_DEV = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/model-cache.tar"
MODEL_URL_DEV = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
# MODEL_URL_SCHNELL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/model.tar"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights: Path | str, pipe: FluxPipeline, lora_scale: float):
        if isinstance(weights, str) and weights.startswith("data:"):
            # Handle data URL
            print("Loading LoRA weights from data URL")
            _, encoded = weights.split(",", 1)
            data = base64.b64decode(encoded)
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tar:
                    tar.extractall(path=temp_dir)
                lora_path = os.path.join(
                    temp_dir, "output/flux_train_replicate/lora.safetensors"
                )
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=lora_scale)
        else:
            # Handle local path
            print(f"Loading LoRA weights from {weights}")
            local_weights_cache = self.weights_cache.ensure(str(weights))
            lora_path = os.path.join(
                local_weights_cache, "output/flux_train_replicate/lora.safetensors"
            )
            pipe.load_lora_weights(lora_path)

        print("LoRA weights loaded successfully")

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        # Dont pull weights
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.weights_cache = WeightsDownloadCache()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Flux dev pipeline")
        if not os.path.exists("FLUX.1-dev"):
            download_weights(MODEL_URL_DEV, ".")
        self.dev_pipe = FluxPipeline.from_pretrained(
            "FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("setup took: ", time.time() - start)

    @torch.amp.autocast("cuda")
    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        lora_scale: float = Input(description="Determines how strongly the LoRA should be applied. Sane results between 0 and 1.", default=1.0, le=2.0, ge=-1.0),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=50,
            default=28,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0,
            le=10,
            default=3.5,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation", default=None
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        replicate_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if replicate_weights:
            self.load_trained_weights(replicate_weights, self.dev_pipe, lora_scale)

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        max_sequence_length = 512

        flux_kwargs = {}
        print(f"Prompt: {prompt}")
        print("txt2img mode")
        flux_kwargs["width"] = width
        flux_kwargs["height"] = height
        if replicate_weights:
            flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
        pipe = self.dev_pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil",
        }

        output = pipe(**common_args, **flux_kwargs)

        if replicate_weights:
            self.dev_pipe.unload_lora_weights()

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker and has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
