import argparse
import os
import sys
import time

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for gym-only installs
    import gym

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


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
            print(f"[PPO] Steps: {self.num_timesteps} | Elapsed: {elapsed/60:.1f} min")
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on UAV-RIS environment.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="ppo_logs")
    parser.add_argument("--model-name", type=str, default="ppo_final")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Force training device (auto uses CUDA if available).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Limit Torch CPU threads to reduce CPU contention.",
    )
    parser.add_argument(
        "--torch-inter-op-threads",
        type=int,
        default=None,
        help="Limit Torch inter-op CPU threads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = _setup_paths()
    import gym_foo  # noqa: F401

    log_dir = os.path.join(repo_root, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_inter_op_threads is not None:
        torch.set_num_interop_threads(args.torch_inter_op_threads)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    env = gym.make("foo-v0", Train=True)
    env.reset(seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device=device,
        seed=args.seed,
    )

    print("Starting PPO training...")
    callback = ProgressPrinter(check_freq=10000)
    model.learn(total_timesteps=args.timesteps, log_interval=10, callback=callback)

    save_path = os.path.join(log_dir, args.model_name)
    model.save(save_path)
    print(f"Saved PPO model to {save_path}.zip")


if __name__ == "__main__":
    main()
