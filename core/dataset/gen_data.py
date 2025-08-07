import os
import numpy as np
import pickle
from tqdm import tqdm
from core.env.leader import PlanarMultiAgentEnvWrapper
import yaml
import argparse

def generate_trajectory(env, obs_horizon=2, pred_horizon=96):
    obs = env.reset()
    traj = {
        "obs": [],
        "action": [],
        "next_obs": [],
    }

    for t in range(pred_horizon):
        action = env.sample_random_action()
        next_obs, _, done, _ = env.step(action)

        traj["obs"].append(obs)
        traj["action"].append(action)
        traj["next_obs"].append(next_obs)

        obs = next_obs
        if done:
            break

    # Convert list to np.array
    for k in traj:
        traj[k] = np.array(traj[k])
    return traj

def main(config_path, save_path, num_episodes):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    obs_horizon = config["obs_horizon"]
    pred_horizon = config["pred_horizon"]

    # Init environment
    env = PlanarMultiAgentEnvWrapper(config)

    dataset = {
        "obs": [],
        "action": [],
        "next_obs": [],
    }

    for i in tqdm(range(num_episodes), desc="Generating data"):
        traj = generate_trajectory(env, obs_horizon, pred_horizon)

        for k in dataset:
            dataset[k].append(traj[k])

    # Stack all
    for k in dataset:
        dataset[k] = np.stack(dataset[k], axis=0)

    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"[INFO] Saved dataset to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="data/multi_agent_dataset.pkl", help="Save path")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")

    args = parser.parse_args()
    main(args.config, args.output, args.num_episodes)

from gen_data import generate_dataset

generate_dataset(
    config_path="config.yaml",
    output_path="trn_data.joblib"
)
