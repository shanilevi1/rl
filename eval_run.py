"""
Run 100-episode final evaluation on a saved checkpoint and write final_eval.json.

Usage:
    python eval_run.py --run_id exp_38_s0 --config configs/exp_38_lowlr.yaml
    python eval_run.py --run_id exp_38_s0 --config configs/exp_38_lowlr.yaml --episodes 100
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import yaml

from dqn.env import make_env, preprocess, FrameStack
from dqn.network import DQNNetwork
from dqn.agent import DQNAgent, AgentConfig
from dqn.replay_buffer import ReplayBuffer
from dqn.evaluate import evaluate


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        if v == "true":  v = True
        if v == "false": v = False
        cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",   required=True)
    parser.add_argument("--config",   required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=None,
                        help="eval env seed (defaults to run_id suffix 0/1/2)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # infer seed from run_id suffix (_s0 → 0, _s1 → 1, _s2 → 2) if not given
    if args.seed is None:
        suffix = args.run_id.split("_")[-1]
        args.seed = int(suffix[1]) if suffix.startswith("s") and suffix[1:].isdigit() else 0
    cfg["seed"] = args.seed

    checkpoint_dir = os.path.join("runs", args.run_id)
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_ckpt = ckpts[-1]
    ckpt_step = int(os.path.basename(latest_ckpt).replace("ckpt_", "").replace(".pt", ""))
    print(f"Run    : {args.run_id}  seed={args.seed}")
    print(f"Config : {args.config}")
    print(f"Ckpt   : {latest_ckpt}  (step {ckpt_step:,})")
    print(f"Eval   : {args.episodes} episodes")

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.get("use_amp"):
        cfg["use_amp"] = False

    num_frames       = cfg.get("num_frames", 4)
    frame_stack_mode = cfg.get("frame_stack_mode", "stack")
    frame_skip       = cfg["frame_skip"]
    noop_max         = cfg.get("noop_max", 0)

    eval_env = make_env(cfg["ale_game_id"], seed=cfg["seed"] + 1000)
    num_actions = eval_env.action_space.n

    network   = DQNNetwork(num_actions, in_channels=num_frames).to(device)
    buffer    = ReplayBuffer(1000, frame_history=num_frames, frame_stack_mode=frame_stack_mode)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=cfg["lr"],
                                    eps=cfg["rmsprop_eps"], alpha=cfg["rmsprop_alpha"])
    agent = DQNAgent(
        network, buffer, optimizer,
        AgentConfig(
            eps_start=cfg["eps_start"], eps_end=cfg["eps_end"],
            eps_anneal_steps=cfg["eps_anneal_steps"], eps_eval=cfg["eps_eval"],
            gamma=cfg["gamma"], batch_size=cfg["batch_size"],
            min_replay_size=cfg.get("min_replay_size", 50000),
            use_amp=cfg.get("use_amp", False),
            grad_clip=cfg.get("grad_clip", 0.0),
            loss_fn=cfg.get("loss_fn", "mse"),
        ),
        device,
    )

    ckpt_data = torch.load(latest_ckpt, map_location=device)
    network.load_state_dict(ckpt_data["network_state"])
    print("Checkpoint loaded. Running evaluation...")

    reward = evaluate(agent, eval_env, args.episodes, device, frame_skip,
                      num_frames=num_frames, frame_stack_mode=frame_stack_mode,
                      noop_max=noop_max)
    eval_env.close()

    out = {
        "final_eval_reward": reward,
        "episodes": args.episodes,
        "ckpt_step": ckpt_step,
        "total_steps": cfg.get("total_steps", 5_000_000),
        "ckpt_is_final": ckpt_step >= cfg.get("total_steps", 5_000_000),
        "run_id": args.run_id,
        "seed": args.seed,
    }
    out_path = os.path.join(checkpoint_dir, "final_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFINAL EVAL ({args.episodes} eps, ckpt@{ckpt_step:,}): {reward:.2f}")
    print(f"Ckpt is final step: {out['ckpt_is_final']}")
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
