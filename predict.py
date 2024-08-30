import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, cast, Tuple

import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
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
        # Don't pull weights
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
        self.inpaint_pipes = {
            "dev": None,
            "schnell": None,
        }
        self.current_model = "dev"
        self.current_inpaint = None

        self.loaded_models = ["safety_checker", "dev", "schnell"]
        print(f"[!] Loaded models: {self.loaded_models}")

        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(  # pyright: ignore
        self,
        prompt: str = Input(description="Prompt for generated image"),
        image: Path = Input(
            description="Input image for img2img or inpaint mode", default=None
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
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
            description="Number of inference steps",
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
            description="Guidance scale for the diffusion process",
            ge=0,
            le=10,
            default=3.5,
        ),
        prompt_strength: float = Input(
            description="Strength for img2img or inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation", default=None
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

        inpaint_mode = image is not None and mask is not None
        self.configure_active_model(model, inpaint_mode)

        if inpaint_mode:
            print("inpaint mode")
            input_image = Image.open(image).convert("RGB")
            mask_image = Image.open(mask).convert("RGB")
            width, height = self.resize_image_dimensions(input_image.size)
            flux_kwargs["image"] = input_image.resize((width, height), Image.LANCZOS)
            flux_kwargs["mask_image"] = mask_image.resize(
                (width, height), Image.LANCZOS
            )
            flux_kwargs["strength"] = prompt_strength
            print(f"Using {model} model for inpainting")
            pipe = self.inpaint_pipes[model]
        else:
            # TODO add img2img mode (when we have just image and not mask)
            print("txt2img mode")
            pipe = self.pipes[model]

        flux_kwargs["width"] = width
        flux_kwargs["height"] = height
        if replicate_weights:
            flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}

        # Avoid a footgun in case we update the model input but forget to
        # update clauses in this if statement
        assert model in ["dev", "schnell"]
        if model == "dev":
            print("Using dev model")
            max_sequence_length = 512
        else:  # model == "schnell":
            print("Using schnell model")
            max_sequence_length = 256
            guidance_scale = 0

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

        # Ensure all model components are on the correct device
        device = pipe.device
        for component_name in ["unet", "text_encoder", "text_encoder_2", "vae"]:
            if hasattr(pipe, component_name):
                component = getattr(pipe, component_name)
                if isinstance(component, torch.nn.Module):
                    component.to(device)

        generator = torch.Generator(device=device).manual_seed(seed)

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

    def configure_active_model(self, model: str, inpaint: bool = False):
        start_time = time.time()
        initial_models = set(self.loaded_models)

        print(f"[~] Configuring active model: {model}, inpaint: {inpaint}")

        # Ensure core models are always loaded
        assert "dev" in self.loaded_models, "dev must always be loaded"
        assert (
            "safety_checker" in self.loaded_models
        ), "safety_checker must always be loaded"

        if inpaint:
            inpaint_model = f"{model}_inpaint"
            if inpaint_model not in self.loaded_models:
                # Unload any model in the "swappable seat" (schnell or inpainting models)
                for model_to_unload in ["schnell", "dev_inpaint", "schnell_inpaint"]:
                    if (
                        model_to_unload in self.loaded_models
                        and model_to_unload != "dev"
                    ):
                        print(f"[~] Moving {model_to_unload} to CPU...")
                        cpu_start = time.time()
                        # Unload either schnell or an inpainting model
                        if model_to_unload == "schnell":
                            self.pipes["schnell"].to("cpu")
                        else:
                            self.inpaint_pipes[model_to_unload.split("_")[0]].to("cpu")
                        print(
                            f"[!] Moved {model_to_unload} to CPU in {time.time() - cpu_start:.2f}s"
                        )
                        self.loaded_models.remove(model_to_unload)

                # Load the required inpainting pipeline
                print(f"[~] Loading {inpaint_model} pipeline...")
                if self.inpaint_pipes[model] is None:
                    # Create a new inpainting pipeline if it doesn't exist
                    print(f"[~] Creating new {model} inpaint pipeline...")
                    create_start = time.time()
                    base_pipe = self.pipes[model]
                    self.inpaint_pipes[model] = FluxInpaintPipeline.from_pretrained(
                        f"FLUX.1-{model}",
                        text_encoder=base_pipe.text_encoder,
                        text_encoder_2=base_pipe.text_encoder_2,
                        tokenizer=base_pipe.tokenizer,
                        tokenizer_2=base_pipe.tokenizer_2,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                    print(
                        f"[~] Created {model} inpaint pipeline in {time.time() - create_start:.2f}s"
                    )
                else:
                    # Move existing inpainting pipeline to CUDA
                    cuda_start = time.time()
                    self.inpaint_pipes[model].to("cuda")
                    print(
                        f"[!] Moved {model} inpaint to CUDA in {time.time() - cuda_start:.2f}s"
                    )
                self.loaded_models.append(inpaint_model)
            self.current_inpaint = model
        else:
            # Non-inpainting mode
            if "schnell" in self.loaded_models:
                # If schnell is already loaded, no action needed
                print("Schnell model already loaded")
                return

            # Unload any inpainting models
            for inpaint_model in ["dev_inpaint", "schnell_inpaint"]:
                if inpaint_model in self.loaded_models:
                    print(f"[~] Moving {inpaint_model} to CPU...")
                    cpu_start = time.time()
                    self.inpaint_pipes[inpaint_model.split("_")[0]].to("cpu")
                    print(
                        f"[!] Moved {inpaint_model} to CPU in {time.time() - cpu_start:.2f}s"
                    )
                    self.loaded_models.remove(inpaint_model)

            # Load schnell model
            print("[~] Moving schnell model to CUDA...")
            cuda_start = time.time()
            self.pipes["schnell"].to("cuda")
            print(f"[!] Moved schnell to CUDA in {time.time() - cuda_start:.2f}s")
            self.loaded_models.append("schnell")
            self.current_inpaint = None

        self.current_model = model

        # Ensure all model components are on the correct device
        pipe = self.pipes[model] if not inpaint else self.inpaint_pipes[model]
        component_start = time.time()
        for component_name in ["unet", "text_encoder", "text_encoder_2", "vae"]:
            if hasattr(pipe, component_name):
                component = getattr(pipe, component_name)
                if isinstance(component, torch.nn.Module):
                    component.to("cuda")
        print(
            f"[!] Moved model components to CUDA in {time.time() - component_start:.2f}s"
        )

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

        # Log changes in loaded models
        if set(self.loaded_models) != initial_models:
            print(f"[!] Loaded models: {self.loaded_models}")
        print(
            f"[!] Total time for configure_active_model: {time.time() - start_time:.2f}s"
        )

    def resize_image_dimensions(
        self,
        original_resolution_wh: Tuple[int, int],
        maximum_dimension: int = 1024,
    ) -> Tuple[int, int]:
        width, height = original_resolution_wh

        if width > height:
            scaling_factor = maximum_dimension / width
        else:
            scaling_factor = maximum_dimension / height

        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        new_width = new_width - (new_width % 32)
        new_height = new_height - (new_height % 32)

        return new_width, new_height


def download_base_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def make_multiple_of_16(n):
    return ((n + 15) // 16) * 16
