import argparse
import os
import sys

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for gym-only installs
    import gym

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DDPG, SAC, PPO, TD3


def _setup_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    td3_dir = os.path.join(repo_root, "TD3-SingleUT-Time")
    sys.path.insert(0, td3_dir)
    os.chdir(td3_dir)
    return repo_root


def _quantize_theta(action_slice, levels=8):
    phase_levels = np.linspace(0, 2 * np.pi, levels, endpoint=False)
    phase_indices = np.minimum((action_slice * levels).astype(int), levels - 1)
    return phase_levels[phase_indices]


def compute_sinr_sumrate(env, tau, power_1, theta_r, l_u, l_ap, ut_0):
    awgn = globe.get_value("AWGN")
    bw = globe.get_value("BW")
    bs_z = globe.get_value("BS_Z")
    ris_l = globe.get_value("RIS_L")

    g_br = env.pl_BR(l_u, l_ap)
    small_fading_g = env.SmallFading_G(bs_z, ris_l)
    g = np.ones((bs_z, ris_l)) * g_br * small_fading_g
    coefficients = np.diag(np.exp(1j * theta_r))

    h_ru = env.Channel_RU(l_u, ut_0, bs_z, ris_l)
    ut_link = np.linalg.multi_dot([g, coefficients, h_ru])
    signal_ut = np.sum(np.abs(ut_link * np.conjugate(ut_link))) * power_1

    sinr_db = 10 * np.log10(signal_ut / awgn)
    if sinr_db > 0:
        sum_rate = bw * np.log2(1 + sinr_db) * (1 - tau)
    else:
        sum_rate = 0.0

    return sinr_db, sum_rate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple trained policies and plot SINR/Sum-Rate."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form algo=path (e.g., sac=sac_logs/sac_colab.zip).",
    )
    parser.add_argument("--episodes-per-pt", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="eval_plots")
    return parser.parse_args()


def load_model(algo, model_path, device):
    if algo == "sac":
        return SAC.load(model_path, device=device)
    if algo == "ppo":
        return PPO.load(model_path, device=device)
    if algo == "td3":
        return TD3.load(model_path, device=device)
    if algo == "ddpg":
        return DDPG.load(model_path, device=device)
    raise ValueError(f"Unsupported algo: {algo}")


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid model spec '{spec}'. Use algo=path.")
    algo, path = spec.split("=", 1)
    algo = algo.strip().lower()
    path = path.strip()
    if not algo or not path:
        raise ValueError(f"Invalid model spec '{spec}'. Use algo=path.")
    return algo, path


def main():
    args = parse_args()
    repo_root = _setup_paths()
    import gym_foo  # noqa: F401
    global globe
    import globe

    model_specs = [parse_model_spec(spec) for spec in args.model]

    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make("foo-v0", Train=False)
    env.reset(seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pt_values = [10, 15, 20, 25, 30]
    sinr_series = {}
    sumrate_series = {}

    for algo, model_path in model_specs:
        resolved_path = model_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(repo_root, resolved_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Model not found at {resolved_path}")

        model = load_model(algo, resolved_path, device=device)

        avg_sinr = []
        avg_sumrate = []

        for pt_dbm in pt_values:
            sinr_runs = []
            sumrate_runs = []
            for _ in range(args.episodes_per_pt):
                obs, _ = env.reset()
                terminated = False
                truncated = False
                episode_sinr = []
                episode_sumrate = []
                while not (terminated or truncated):
                    action, _states = model.predict(obs, deterministic=True)
                    action = np.array(action, dtype=np.float32)
                    action[1] = pt_dbm / 30.0

                    step = globe.get_value("step")
                    t = globe.get_value("t")
                    if step < t - 1:
                        l_u = globe.get_value("UAV_Trajectory")[step + 1]
                        ut_0 = globe.get_value("UT_0")[step + 1]
                    else:
                        l_u = globe.get_value("UAV_Trajectory")[step]
                        ut_0 = globe.get_value("UT_0")[step]
                    l_ap = globe.get_value("L_AP")

                    tau = action[0]
                    power_1 = 10 ** (((action[1] - 1) * 30 / 10) + 3)
                    theta_r = _quantize_theta(action[2:])

                    sinr_db, sum_rate = compute_sinr_sumrate(
                        env, tau, power_1, theta_r, l_u, l_ap, ut_0
                    )
                    episode_sinr.append(sinr_db)
                    episode_sumrate.append(sum_rate)

                    obs, reward, terminated, truncated, info = env.step(action)

                sinr_runs.append(float(np.mean(episode_sinr)))
                sumrate_runs.append(float(np.mean(episode_sumrate)))

            avg_sinr.append(float(np.mean(sinr_runs)))
            avg_sumrate.append(float(np.mean(sumrate_runs)))

        sinr_series[algo] = avg_sinr
        sumrate_series[algo] = avg_sumrate

    plt.figure()
    for algo, values in sinr_series.items():
        plt.plot(pt_values, values, marker="o", label=algo.upper())
    plt.xlabel("Transmit Power Pt (dBm)")
    plt.ylabel("Average SINR (dB)")
    plt.title("SINR vs Transmit Power")
    plt.grid(True)
    plt.legend()
    sinr_path = os.path.join(output_dir, "SINR_vs_Pt_compare.png")
    plt.savefig(sinr_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    for algo, values in sumrate_series.items():
        plt.plot(pt_values, values, marker="o", label=algo.upper())
    plt.xlabel("Transmit Power Pt (dBm)")
    plt.ylabel("Average Sum-Rate")
    plt.title("Sum-Rate vs Transmit Power")
    plt.grid(True)
    plt.legend()
    sumrate_path = os.path.join(output_dir, "SumRate_vs_Pt_compare.png")
    plt.savefig(sumrate_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plots to {sinr_path} and {sumrate_path}")


if __name__ == "__main__":
    main()
