from pathlib import Path
from typing import Any, Sequence
import wandb
from wandb.sdk.wandb_settings import Settings


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
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            save_code=False,
            settings=Settings(_disable_machine_info=True),
        )

    def log_loss(self, loss_dict: dict[str, Any], step: int | None):
        wandb.log(data=loss_dict, step=step)

    def log_samples(self, image_paths: Sequence[Path], step: int | None):
        data = {
            f"samples/{prompt}": wandb.Image(str(path))
            for prompt, path in zip(self.sample_prompts, image_paths)
        }
        wandb.log(data=data, step=step)

    def save_weights(self, lora_path: Path):
        wandb.save(lora_path)

    def finish(self):
        wandb.finish()
