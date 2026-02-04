import os
import sys
import time

import gym
import torch

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError as exc:
    missing = str(exc)
    raise ModuleNotFoundError(
        "Missing dependency. Install with:\n"
        "  pip install stable-baselines3\n"
        "For Colab, also pin versions:\n"
        "  pip install numpy==1.26.4 gym==0.19.0 matplotlib==3.7.1\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  pip install stable-baselines3"
    ) from exc


def _setup_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    td3_dir = os.path.join(repo_root, "TD3-SingleUT-Time")
    sys.path.insert(0, td3_dir)
    os.chdir(td3_dir)
    return repo_root


class ProgressPrinter(BaseCallback):
    def __init__(self, check_freq=10000):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            print(f"[SAC] Steps: {self.num_timesteps} | Elapsed: {elapsed/60:.1f} min")
        return True


def main():
    repo_root = _setup_paths()
    import gym_foo  # noqa: F401

    log_dir = os.path.join(repo_root, "sac_logs")
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make("foo-v0", Train=True)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device=device,
    )

    print("Starting SAC training...")
    callback = ProgressPrinter(check_freq=10000)
    model.learn(total_timesteps=1_620_000, log_interval=10, callback=callback)

    save_path = os.path.join(log_dir, "sac_final")
    model.save(save_path)
    print(f"Saved SAC model to {save_path}.zip")


if __name__ == "__main__":
    main()
