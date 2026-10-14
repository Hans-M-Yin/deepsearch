"""Episode JSON Logger for saving detailed episode information."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from rllm.agents.agent import Episode


class EpisodeLogger:
    """Logger to save episodes to individual JSON files with step and data hash."""

    def __init__(
        self,
        base_dir: str,
        subdirectory: str = "episodes",
        max_episodes_per_batch: int | None = 50,
    ):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Subdirectory within base_dir for episodes (default: "episodes")
                         Final path will be: {base_dir}/{subdirectory}/
        """
        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # A rollout batch can contain thousands of episodes.  Limit only the
        # persisted episode files; all episodes still participate in training.
        self.max_episodes_per_batch = max_episodes_per_batch

    def select_episodes_for_logging(
        self, episodes: list[Episode]
    ) -> list[Episode]:
        """Randomly select the episodes to persist for one rollout batch."""
        limit = self.max_episodes_per_batch
        if limit is None or limit <= 0 or len(episodes) <= limit:
            return list(episodes)
        return random.sample(episodes, k=limit)

    @staticmethod
    def compute_task_hash(task: Any, length: int = 8) -> str:
        """Compute a hash from the task data.

        Args:
            task: The task dictionary or data
            length: Length of the hash to use (default 8 chars)

        Returns:
            Hash string
        """
        # Convert task to a stable string representation
        task_str = json.dumps(task, sort_keys=True, default=str)
        # Compute SHA256 hash
        hash_obj = hashlib.sha256(task_str.encode("utf-8"))
        # Return first `length` characters of hex digest
        return hash_obj.hexdigest()[:length]

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the directory path for a specific training or validation step.

        Args:
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0

        Returns:
            Path object for the step directory
        """
        step_dir = self.log_dir / f"{mode}_step_{step}_epoch_{epoch}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate filename for an episode.

        Format: episode_hash{task_hash}_id{episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        task_hash = self.compute_task_hash(episode.task)
        # Clean episode_id to make it filesystem-safe
        episode_id_safe = str(episode.id).replace(":", "_").replace("/", "_")

        filename = f"episode_hash{task_hash}_id{episode_id_safe}.json"
        return filename

    @staticmethod
    def _image_reference(image: Any) -> Any:
        """Return a JSON-safe image reference without serializing image bytes.

        Rollout messages may carry URLs, local paths, PIL images, or dictionaries
        containing bytes. The saved trajectory describes the supplied image slot
        without copying the image payload into every episode file.
        """
        if isinstance(image, str):
            if image.startswith("data:image/"):
                return {"type": "data_image", "value": "<inline_image_omitted>"}
            return image

        if isinstance(image, dict):
            reference = {}
            for key in ("url", "path", "name", "id", "mime_type", "media_type"):
                value = image.get(key)
                if isinstance(value, (str, int, float, bool)):
                    reference[key] = value
            if image.get("bytes") is not None:
                try:
                    reference["byte_size"] = len(image["bytes"])
                except TypeError:
                    reference["byte_size"] = "<unknown>"
            if not reference:
                reference = {"type": type(image).__name__}
            return reference

        # Avoid importing PIL just for logging. These attributes are available
        # on PIL.Image.Image and keep the descriptor small and deterministic.
        if hasattr(image, "size") and hasattr(image, "mode"):
            return {
                "type": type(image).__name__,
                "size": list(image.size),
                "mode": str(image.mode),
            }

        return {"type": type(image).__name__, "repr": str(image)[:200]}

    @classmethod
    def build_model_visible_messages(cls, episode: Episode) -> list[dict[str, Any]]:
        """Reconstruct the final model-visible conversation for an episode.

        Each DeepResearch ``Step.chat_completions`` is a cumulative prefix ending
        at one assistant generation. The last step therefore contains the full
        conversation sent to the last LLM request, including the system prompt,
        all prior user observations, and all prior assistant tool calls. Using
        that prefix avoids writing the same long context once per step.

        The observation produced after the final assistant generation is not
        appended: it was not presented to the model and is not model-visible.
        Image payloads are represented only by safe references; binary data and
        ModelOutput tensors are omitted.
        """
        last_step = None
        for trajectory in episode.trajectories or []:
            if trajectory.steps:
                last_step = trajectory.steps[-1]

        if last_step is not None and last_step.chat_completions:
            messages = last_step.chat_completions
        else:
            # A rollout can fail before the first assistant generation. Keep a
            # useful user prompt fallback for those episodes; no system prompt is
            # available in the Episode object in that exceptional path.
            task = episode.task if isinstance(episode.task, dict) else {}
            question = task.get("question")
            return (
                [{"role": "user", "content": str(question)}]
                if question is not None
                else []
            )

        serialized_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                serialized_messages.append(
                    {"role": "unknown", "content": str(message)}
                )
                continue

            serialized: dict[str, Any] = {
                "role": message.get("role"),
                "content": message.get("content", ""),
            }
            if "images" in message:
                serialized["images"] = [
                    cls._image_reference(image)
                    for image in (message.get("images") or [])
                ]
            serialized_messages.append(serialized)
        return serialized_messages

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to its own JSON file in a step-specific directory.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        # Keep rollout logs compact and safe: retain the model-visible text
        # conversation, but never serialize image bytes or ModelOutput tensors.
        info = episode.info or {}
        # Explicit allowlist: do not persist the entire task/info (images,
        # token tensors, private search traces, or service configuration).
        reward_metadata = info.get("reward_metadata") or {}
        reward_metadata = reward_metadata if isinstance(reward_metadata, dict) else {}
        metrics = getattr(episode, "metrics", None) or {}
        task = episode.task if isinstance(episode.task, dict) else {}
        extra = task.get("extra_info") or {}
        extra = extra if isinstance(extra, dict) else {}
        # Dataset.to_verl_dataset wraps the original entry in extra_info;
        # the workflow receives that entry as its task (id is not episode.id).
        data_id = task.get("id")
        if data_id is None:
            data_id = extra.get("id")
        reward_components = {
            name: metrics.get(key)
            for name, key in {
                "total": "reward/total",
                "accuracy": "reward/accuracy",
                "format": "reward/format",
                "redundancy_penalty": "reward/search_redundancy_penalty",
                "redundancy_raw_penalty": "reward/search_redundancy_raw_penalty",
                "partial_reward": "reward/statement_partial_credit_applied_bonus",
                "partial_raw_bonus": "reward/statement_partial_credit_raw_bonus",
                "masked": "masked",
                "termination_penalty": "reward/termination_penalty",
            }.items()
        }
        # Fatal first-step rollouts have their entire reward hard-masked,
        # even though their diagnostic penalty metric may still be nonzero.
        if metrics.get("masked") == 1:
            reward_components["redundancy_penalty"] = 0.0
        for name, key in (("partial_mode", "statement_partial_credit"),
                          ("redundancy_mode", "search_redundancy")):
            record = info.get(key)
            reward_components[name] = record.get("mode") if isinstance(record, dict) else None
        episode_data = {
            "training_step": step,
            "epoch": epoch,
            "mode": mode,
            "episode_id": episode.id,
            "data_id": data_id,
            "data_provenance": {
                key: extra.get(key, task.get(key))
                for key in ("source_question_id", "source_sample_id", "source_path_id",
                            "source_run_dir", "source_index")
                if key in extra or key in task
            },
            # accuracy/format are unweighted scores; penalties are positive
            # deductions. Missing metrics are null, never invented zeroes.
            "reward_components": reward_components,
            "task_hash": self.compute_task_hash(episode.task),
            "is_correct": bool(episode.is_correct),
            "prediction": info.get("prediction"),
            "answer": info.get("answer"),
            "reward_metadata": {
                key: reward_metadata[key]
                for key in (
                    "extracted_answer", "normalized_prediction", "normalized_refs",
                    "judge_used", "judge_decided", "judge_attempts", "judgment",
                    "exact_match", "fallback_reason", "judge_execution_mode",
                )
                if key in reward_metadata
            },
            "termination_reason": (
                episode.termination_reason.value
                if episode.termination_reason
                else None
            ),
            "raw_termination": (episode.info or {}).get("raw_termination"),
            "is_fatal": bool(info.get("is_fatal", False)),
            "is_model_failure": bool(info.get("is_model_failure", False)),
            "fatal_step_index": info.get("fatal_step_index"),
            "fatal_reason": info.get("fatal_reason"),
            "format_error_rounds": info.get("format_error_rounds", []),
            "rollout_markers": (episode.info or {}).get(
                "rollout_markers", []
            ),
            # Exact cumulative message prefix used by the last model request:
            # system prompt, user prompts/tool observations, and assistant
            # generations including <tool_call> blocks.
            "messages": self.build_model_visible_messages(episode),
            "trajectories": [],
        }

        for traj in episode.trajectories:
            traj_data = {
                "name": traj.name,
                "uid": traj.uid,
                "reward": float(traj.reward),
                "model_responses": [
                    step.model_response
                    if isinstance(step.model_response, str)
                    else str(step.model_response)
                    for step in traj.steps
                ],
                "step_errors": [
                    {
                        "round": index + 1,
                        "step_error": bool(step.info.get("step_error", False)),
                        "error_type": step.info.get("error_type", ""),
                    }
                    for index, step in enumerate(traj.steps)
                ],
            }
            episode_data["trajectories"].append(traj_data)

        # Write to individual file in step-specific directory
        step_dir = self.get_step_dir(step, mode, epoch)
        filename = self.get_episode_filename(episode, step)
        filepath = step_dir / filename

        try:
            with open(filepath, "w") as f:
                json_str = json.dumps(episode_data, indent=2, default=str)
                f.write(json_str + "\n")
                f.flush()  # Ensure data is written to disk
        except Exception as e:
            print(f"Error writing episode to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes, each to its own file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        print(
            f"[EpisodeLogger] Logging {len(episodes)} episodes for "
            f"step={step}, mode={mode}, epoch={epoch}"
        )
        saved = 0
        for i, episode in enumerate(episodes):
            try:
                self.log_episode(episode, step, mode, epoch)
                saved += 1
            except Exception as e:
                print(
                    f"[EpisodeLogger] Failed to log episode "
                    f"{i + 1}/{len(episodes)} ({episode.id}): {e}"
                )
                raise
        print(f"[EpisodeLogger] Logged {saved}/{len(episodes)} episodes")

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes and optionally create a batch summary in step-specific directory.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Whether to create a summary file for the batch
        """
        # Persist only a random subset. The complete ``episodes`` list is still
        # retained by the caller and used for training/statistics.
        episodes_to_save = self.select_episodes_for_logging(episodes)
        self.log_episodes(episodes_to_save, step, mode, epoch)

        # Optionally create batch summary in step-specific directory
        if batch_summary and episodes:
            summary_data = {
                "training_step": step,
                "epoch": epoch,
                "mode": mode,
                "num_episodes": len(episodes),
                "num_episodes_saved": len(episodes_to_save),
                "sampling_limit": self.max_episodes_per_batch,
                "episode_files": [
                    self.get_episode_filename(ep, step) for ep in episodes_to_save
                ],
                "summary_stats": {
                    "total_correct": sum(1 for ep in episodes if ep.is_correct),
                    "total_incorrect": sum(1 for ep in episodes if not ep.is_correct),
                    "accuracy": sum(1 for ep in episodes if ep.is_correct) / len(episodes) if episodes else 0,
                    "avg_trajectories_per_episode": sum(len(ep.trajectories) for ep in episodes) / len(episodes) if episodes else 0,
                },
            }

            step_dir = self.get_step_dir(step, mode, epoch)
            summary_file = step_dir / "batch_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)
