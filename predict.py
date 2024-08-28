import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, cast

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

from weights import WeightsDownloadCache

MODEL_URL_DEV = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
MODEL_URL_SCHNELL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/slim.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
SAFETY_CACHE_PATH = Path("safety-cache")
FLUX_DEV_PATH = Path("FLUX.1-dev")
FLUX_SCHNELL_PATH = Path("FLUX.1-schnell")
FEATURE_EXTRACTOR = Path("/src/feature-extractor")

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


@dataclass
class LoadedLoRAs:
    main: str | None
    extra: str | None


class Predictor(BasePredictor):
    def setup(self) -> None:  # pyright: ignore
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        # Dont pull weights
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.weights_cache = WeightsDownloadCache()

        print("Loading safety checker...")
        if not SAFETY_CACHE_PATH.exists():
            download_base_weights(SAFETY_URL, SAFETY_CACHE_PATH)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE_PATH, torch_dtype=torch.float16
        ).to("cuda")  # pyright: ignore
        self.feature_extractor = cast(
            CLIPImageProcessor, CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        )

        print("Loading Flux dev pipeline")
        if not FLUX_DEV_PATH.exists():
            download_base_weights(MODEL_URL_DEV, Path("."))
        dev_pipe = FluxPipeline.from_pretrained(
            "FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("Loading Flux schnell pipeline")
        if not FLUX_SCHNELL_PATH.exists():
            download_base_weights(MODEL_URL_SCHNELL, FLUX_SCHNELL_PATH)
        schnell_pipe = FluxPipeline.from_pretrained(
            "FLUX.1-schnell",
            text_encoder=dev_pipe.text_encoder,
            text_encoder_2=dev_pipe.text_encoder_2,
            tokenizer=dev_pipe.tokenizer,
            tokenizer_2=dev_pipe.tokenizer_2,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        self.pipes = {
            "dev": dev_pipe,
            "schnell": schnell_pipe,
        }
        self.loaded_lora_urls = {
            "dev": LoadedLoRAs(main=None, extra=None),
            "schnell": LoadedLoRAs(main=None, extra=None),
        }

        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(  # pyright: ignore
        self,
        prompt: str = Input(
            description="Prompt for generated image. Include the `trigger_word` used in the training process to activate the trained object, style, or concept in the resulting image."
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image. The size will always be 1 megapixel, i.e. 1024x1024 if aspect ratio is 1:1. To use arbitrary width and height, set aspect ratio to 'custom'.",
            choices=list(ASPECT_RATIOS.keys()) + ["custom"],  # pyright: ignore
            default="1:1",
        ),
        width: int = Input(
            description="Width of the generated image. Optional, only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16)",
            ge=256,
            le=1440,
            default=None,
        ),
        height: int = Input(
            description="Height of the generated image. Optional, only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16)",
            ge=256,
            le=1440,
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        lora_scale: float = Input(
            description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1.",
            default=1.0,
            le=2.0,
            ge=-1.0,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps. More steps generate more detailed images.",
            ge=1,
            le=50,
            default=28,
        ),
        model: str = Input(
            description="Which model to run inferences with. The dev model needs around 28 steps but the schnell model only needs around 4 steps.",
            choices=["dev", "schnell"],
            default="dev",
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process. Higher values generate images that are more strictly adherent to the text prompt. Lower values allow for more creative and varied results.",
            ge=0,
            le=10,
            default=3.5,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation.", default=None
        ),
        extra_lora: str = Input(
            description="Combine this fine-tune with another LoRA. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
            default=None,
        ),
        extra_lora_scale: float = Input(
            description="Determines how strongly the extra LoRA should be applied.",
            ge=0,
            le=1,
            default=0.8,
        ),
        output_format: str = Input(
            description="Format of the output images.",
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
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if aspect_ratio == "custom":
            if width is None or height is None:
                raise ValueError(
                    "width and height must be defined if aspect ratio is 'custom'"
                )
            width = make_multiple_of_16(width)
            height = make_multiple_of_16(height)
        else:
            width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        max_sequence_length = 512

        flux_kwargs = {}
        print(f"Prompt: {prompt}")
        print("txt2img mode")
        flux_kwargs["width"] = width
        flux_kwargs["height"] = height
        if replicate_weights:
            flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
        if model == "dev":
            print("Using dev model")
            max_sequence_length = 512
        else:
            print("Using schnell model")
            max_sequence_length = 256
            guidance_scale = 0

        pipe = self.pipes[model]

        if replicate_weights:
            start_time = time.time()
            if extra_lora:
                flux_kwargs["joint_attention_kwargs"] = {"scale": 1.0}
                print(f"Loading extra LoRA weights from: {extra_lora}")
                self.load_multiple_loras(replicate_weights, extra_lora, model)
                pipe.set_adapters(
                    ["main", "extra"], adapter_weights=[lora_scale, extra_lora_scale]
                )
            else:
                flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
                self.load_single_lora(replicate_weights, model)
                pipe.set_adapters(["main"], adapter_weights=[lora_scale])
            print(f"Loaded LoRAs in {time.time() - start_time:.2f}s")
        else:
            pipe.unload_lora_weights()
            self.loaded_lora_urls[model] = LoadedLoRAs(main=None, extra=None)

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

        has_nsfw_content = None
        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if has_nsfw_content is not None and has_nsfw_content[i]:
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

    def load_single_lora(self, lora_url: str, model: str):
        # If no change, skip
        if lora_url == self.loaded_lora_urls[model].main:
            print("Weights already loaded")
            return

        pipe = self.pipes[model]
        pipe.unload_lora_weights()
        lora_path = self.weights_cache.ensure(lora_url)
        pipe.load_lora_weights(lora_path, adapter_name="main")
        self.loaded_lora_urls[model] = LoadedLoRAs(main=lora_url, extra=None)

    def load_multiple_loras(self, main_lora_url: str, extra_lora_url: str, model: str):
        pipe = self.pipes[model]
        loaded_lora_urls = self.loaded_lora_urls[model]

        # If no change, skip
        if (
            main_lora_url == loaded_lora_urls.main
            and extra_lora_url == self.loaded_lora_urls[model].extra
        ):
            print("Weights already loaded")
            return

        # We always need to load both?
        pipe.unload_lora_weights()

        main_lora_path = self.weights_cache.ensure(main_lora_url)
        pipe.load_lora_weights(main_lora_path, adapter_name="main")

        extra_lora_path = self.weights_cache.ensure(extra_lora_url)
        pipe.load_lora_weights(extra_lora_path, adapter_name="extra")

        self.loaded_lora_urls[model] = LoadedLoRAs(
            main=main_lora_url, extra=extra_lora_url
        )

    @torch.amp.autocast("cuda")  # pyright: ignore
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

    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]


def download_base_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def make_multiple_of_16(n):
    return ((n + 15) // 16) * 16
