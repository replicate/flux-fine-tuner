import netrc
from pathlib import Path
from typing import Any, Sequence
from contextlib import suppress
import wandb
from wandb.sdk.wandb_settings import Settings


def logout_wandb():
    netrc_path = Path("/root/.netrc")
    if not netrc_path.exists():
        return

    n = netrc.netrc(netrc_path)

    if "api.wandb.ai" in n.hosts:
        del n.hosts["api.wandb.ai"]

        netrc_path.write_text(repr(n))


class WeightsAndBiasesClient:
    def __init__(
        self,
        api_key: str,
        project: str,
        config: dict,
        sample_prompts: list[str],
        entity: str | None,
        name: str | None,
    ):
        self.api_key = api_key
        self.sample_prompts = sample_prompts
        wandb.login(key=self.api_key, verify=True)
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                save_code=False,
                settings=Settings(_disable_machine_info=True),
            )
        except Exception as e:
            raise ValueError(f"Failed to log in to Weights & Biases: {e}")

    def log_loss(self, loss_dict: dict[str, Any], step: int | None):
        try:
            wandb.log(data=loss_dict, step=step)
        except Exception as e:
            print(f"Failed to log to Weights & Biases: {e}")

    def log_samples(self, image_paths: Sequence[Path], step: int | None):
        data = {
            f"samples/{truncate(prompt)}": wandb.Image(str(path))
            for prompt, path in zip(self.sample_prompts, image_paths)
        }
        try:
            wandb.log(data=data, step=step)
        except Exception as e:
            print(f"Failed to log to Weights & Biases: {e}")

    def save_weights(self, lora_path: Path):
        try:
            wandb.save(lora_path)
        except Exception as e:
            print(f"Failed to save to Weights & Biases: {e}")

    def finish(self):
        with suppress(Exception):
            wandb.finish()


def truncate(text, max_chars=50):
    if len(text) <= max_chars:
        return text
    half = (max_chars - 3) // 2
    return f"{text[:half]}...{text[-half:]}"
