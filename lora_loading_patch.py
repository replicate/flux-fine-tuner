# ruff: noqa
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
    get_peft_kwargs,
    is_peft_version,
    get_adapter_name,
    logging,
)

logger = logging.get_logger(__name__)


# patching inject_adapter_in_model and load_peft_state_dict with low_cpu_mem_usage=True until it's merged into diffusers
def load_lora_into_transformer(
    cls, state_dict, network_alphas, transformer, adapter_name=None, _pipeline=None
):
    """
    This will load the LoRA layers specified in `state_dict` into `transformer`.

    Parameters:
        state_dict (`dict`):
            A standard state dict containing the lora layer parameters. The keys can either be indexed directly
            into the unet or prefixed with an additional `unet` which can be used to distinguish between text
            encoder lora layers.
        network_alphas (`Dict[str, float]`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the
            same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
            link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
        transformer (`SD3Transformer2DModel`):
            The Transformer model to load the LoRA layers into.
        adapter_name (`str`, *optional*):
            Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
            `default_{i}` where i is the total number of adapters being loaded.
    """
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    keys = list(state_dict.keys())

    transformer_keys = [k for k in keys if k.startswith(cls.transformer_name)]
    state_dict = {
        k.replace(f"{cls.transformer_name}.", ""): v
        for k, v in state_dict.items()
        if k in transformer_keys
    }

    if len(state_dict.keys()) > 0:
        # check with first key if is not in peft format
        first_key = next(iter(state_dict.keys()))
        if "lora_A" not in first_key:
            state_dict = convert_unet_state_dict_to_peft(state_dict)

        if adapter_name in getattr(transformer, "peft_config", {}):
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
            )

        rank = {}
        for key, val in state_dict.items():
            if "lora_B" in key:
                rank[key] = val.shape[1]

        if network_alphas is not None and len(network_alphas) >= 1:
            prefix = cls.transformer_name
            alpha_keys = [
                k
                for k in network_alphas.keys()
                if k.startswith(prefix) and k.split(".")[0] == prefix
            ]
            network_alphas = {
                k.replace(f"{prefix}.", ""): v
                for k, v in network_alphas.items()
                if k in alpha_keys
            }

        lora_config_kwargs = get_peft_kwargs(
            rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict
        )
        if "use_dora" in lora_config_kwargs:
            if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                )
            else:
                lora_config_kwargs.pop("use_dora")
        lora_config = LoraConfig(**lora_config_kwargs)

        # adapter_name
        if adapter_name is None:
            adapter_name = get_adapter_name(transformer)

        # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
        # otherwise loading LoRA weights will lead to an error
        is_model_cpu_offload, is_sequential_cpu_offload = (
            cls._optionally_disable_offloading(_pipeline)
        )

        inject_adapter_in_model(
            lora_config, transformer, adapter_name=adapter_name, low_cpu_mem_usage=True
        )
        incompatible_keys = set_peft_model_state_dict(
            transformer, state_dict, adapter_name, low_cpu_mem_usage=True
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />
