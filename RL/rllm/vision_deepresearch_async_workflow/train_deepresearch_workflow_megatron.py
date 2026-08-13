import hydra

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    get_all_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import deepresearch_reward_fn_async
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    dataset_name = config.data.dataset_name
    train_dataset = DatasetRegistry.load_dataset(
        dataset_name, config.data.train_split
    )
    val_dataset = DatasetRegistry.load_dataset(
        dataset_name, config.data.val_split
    )

    if train_dataset is None:
        raise FileNotFoundError(
            f"Training split '{config.data.train_split}' for dataset "
            f"'{dataset_name}' was not found in DatasetRegistry."
        )
    if val_dataset is None:
        raise FileNotFoundError(
            f"Validation split '{config.data.val_split}' for dataset "
            f"'{dataset_name}' was not found in DatasetRegistry."
        )

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "reward_function": deepresearch_reward_fn_async,
            "tools": get_all_tools(),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
