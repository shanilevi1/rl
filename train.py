"""
Train a DQN agent on a single Atari game.

Usage:
    python train.py --run_id run_0 --seed 0
    python train.py --run_id run_1 --seed 1
    python train.py --run_id run_2 --seed 2

Key=value overrides:
    python train.py --run_id smoke --seed 0 total_steps=20000
"""
import argparse
import glob
import json
import os
import yaml
import numpy as np
import torch
import wandb
from tqdm import tqdm

from dqn.env import make_env, preprocess, FrameStack
from dqn.network import DQNNetwork
from dqn.replay_buffer import ReplayBuffer
from dqn.agent import DQNAgent, AgentConfig
from dqn.evaluate import evaluate


def load_config(config_path: str, overrides: list) -> dict:
    with open(config_path) as f:
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


def save_checkpoint(network, optimizer, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{step:010d}.pt")
    torch.save({
        "network_state":   network.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
    }, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    cfg["run_id"] = args.run_id
    cfg["seed"]   = args.seed
    checkpoint_dir = os.path.join("runs", args.run_id)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["run_id"],
        config=cfg,
        mode="offline" if args.wandb_offline else "online",
    )

    # Device: fall back to CPU if CUDA unavailable (e.g., login node)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.get("use_amp"):
        cfg["use_amp"] = False
        print("AMP disabled: CUDA not available")

    # Environments
    train_env = make_env(cfg["ale_game_id"], seed=cfg["seed"])
    eval_env  = make_env(cfg["ale_game_id"], seed=cfg["seed"] + 1000)

    obs, reset_info = train_env.reset(seed=cfg["seed"])
    num_actions = train_env.action_space.n

    # Build components
    num_frames = cfg.get("num_frames", 4)
    frame_stack_mode = cfg.get("frame_stack_mode", "stack")
    network = DQNNetwork(num_actions, in_channels=num_frames).to(device)
    buffer  = ReplayBuffer(cfg["replay_buffer_capacity"],
                           frame_history=num_frames,
                           frame_stack_mode=frame_stack_mode)
    optimizer = torch.optim.RMSprop(
        network.parameters(),
        lr=cfg["lr"],
        eps=cfg["rmsprop_eps"],
        alpha=cfg["rmsprop_alpha"],
    )
    agent = DQNAgent(
        network, buffer, optimizer,
        AgentConfig(
            eps_start=cfg["eps_start"],
            eps_end=cfg["eps_end"],
            eps_anneal_steps=cfg["eps_anneal_steps"],
            eps_eval=cfg["eps_eval"],
            gamma=cfg["gamma"],
            batch_size=cfg["batch_size"],
            min_replay_size=cfg["min_replay_size"],
            use_amp=cfg["use_amp"],
            grad_clip=cfg["grad_clip"],
            loss_fn=cfg.get("loss_fn", "mse"),
        ),
        device,
    )

    # Resume from latest checkpoint if one exists
    step_count = 0
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_*.pt")))
    if ckpts:
        ckpt_data = torch.load(ckpts[-1], map_location=device)
        network.load_state_dict(ckpt_data["network_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        step_count = ckpt_data["step"]
        print(f"Resumed from {ckpts[-1]} at step {step_count:,}")

    frame_skip          = cfg["frame_skip"]
    total_steps         = cfg["total_steps"]
    update_freq         = cfg["update_freq"]
    eval_freq           = cfg["eval_freq_steps"]
    ckpt_freq           = cfg["checkpoint_freq_steps"]
    eval_eps            = cfg["eval_episodes"]
    life_loss_terminal  = cfg.get("life_loss_terminal", False)
    maxpool_all_frames  = cfg.get("maxpool_all_frames", False)
    noop_max            = cfg.get("noop_max", 0)
    lr_decay_start      = cfg.get("lr_decay_start_step", 0)
    lr_min              = cfg.get("lr_min", 0.0)
    use_best_ckpt_eval  = cfg.get("use_best_checkpoint_eval", False)
    best_eval_reward    = float("-inf")
    best_eval_step      = step_count
    best_ckpt_path      = os.path.join(checkpoint_dir, "ckpt_best.pt")

    # Apply no-op start to initial state (randomizes starting position per Mnih 2013)
    if noop_max > 0:
        n_noops = np.random.randint(1, noop_max + 1)
        for _ in range(n_noops):
            obs, _, _nt, _ntr, reset_info = train_env.step(0)
            if _nt or _ntr:
                obs, reset_info = train_env.reset()

    print(f"Game : {cfg['ale_game_id']}")
    print(f"Run  : {args.run_id}  seed={args.seed}  device={device}")
    print(f"Actions={num_actions}  total_steps={total_steps:,}")

    # Training state
    frame_stack    = FrameStack(num_frames=num_frames, mode=frame_stack_mode)
    preprocessed   = preprocess(obs)
    phi            = frame_stack.reset(preprocessed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    stats_path = os.path.join(checkpoint_dir, "training_stats.jsonl")
    stats_file = open(stats_path, "a")

    episode_reward = 0.0
    last_eval_r    = float("nan")
    last_loss      = float("nan")
    prev_lives     = reset_info.get("lives", float("inf"))

    pbar = tqdm(
        total=total_steps,
        initial=step_count,
        desc=args.run_id,
        unit="step",
        dynamic_ncols=True,
        smoothing=0.01,
    )

    while step_count < total_steps:
        action = agent.select_action(phi, step_count)

        # Frame skip: repeat action, max-pool frames to handle flickering
        total_reward = 0.0
        terminated = truncated = life_lost = False
        skip_frames = [preprocessed] if maxpool_all_frames else [preprocessed, preprocessed]

        for k in range(frame_skip):
            raw_obs, reward, terminated, truncated, info = train_env.step(action)
            step_count += 1
            total_reward += reward
            new_frame = preprocess(raw_obs)
            if maxpool_all_frames:
                skip_frames.append(new_frame)
            else:
                skip_frames[k % 2] = new_frame
            pbar.update(1)

            if life_loss_terminal:
                curr_lives = info.get("lives", prev_lives)
                if curr_lives < prev_lives:
                    life_lost = True
                prev_lives = curr_lives

            # Evaluate at exact eval_freq-step intervals (inside frame-skip loop for precision)
            if step_count % eval_freq == 0:
                last_eval_r = evaluate(agent, eval_env, eval_eps, device, frame_skip,
                                       num_frames=num_frames, frame_stack_mode=frame_stack_mode,
                                       noop_max=noop_max)
                wandb.log({"eval/mean_reward": last_eval_r}, step=step_count)
                if use_best_ckpt_eval and last_eval_r > best_eval_reward:
                    best_eval_reward = last_eval_r
                    best_eval_step   = step_count
                    torch.save({"network_state":   network.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "step":            step_count}, best_ckpt_path)
                pbar.write(
                    f"[step {step_count:,}] eval_reward={last_eval_r:.1f}"
                    f"  eps={agent.get_epsilon(step_count):.3f}"
                    f"  buf={len(buffer):,}"
                    + (f"  best={best_eval_reward:.1f}@{best_eval_step//1000}k"
                       if use_best_ckpt_eval else "")
                )

            if terminated or truncated or life_lost:
                break

        done = terminated or truncated or life_lost
        # Max-pool collected frames to reduce sprite flickering
        next_frame = skip_frames[-1].copy()
        for f in skip_frames[:-1]:
            np.maximum(next_frame, f, out=next_frame)
        phi_next   = frame_stack.step(next_frame)

        # Store the frame that was current BEFORE the action
        agent.store_transition(preprocessed, action, total_reward, done)
        episode_reward += total_reward

        # Train every update_freq raw steps once buffer is warm
        if step_count >= cfg["min_replay_size"] and step_count % update_freq == 0:
            metrics = agent.train_step()
            if metrics is not None:
                last_loss = metrics["loss"]
                if step_count % 1000 == 0:
                    wandb.log(
                        {
                            "train/loss":        metrics["loss"],
                            "train/grad_norm":   metrics["grad_norm"],
                            "train/q_mean":      metrics["q_mean"],
                            "train/q_max":       metrics["q_max"],
                            "train/q_std":       metrics["q_std"],
                            "train/target_mean": metrics["target_mean"],
                            "train/target_std":  metrics["target_std"],
                            "train/td_error":    metrics["td_error"],
                            "train/epsilon":     agent.get_epsilon(step_count),
                            "train/buffer_size": len(buffer),
                            "train/lr":          optimizer.param_groups[0]["lr"],
                        },
                        step=step_count,
                    )
                    record = {"step": step_count, **metrics,
                              "epsilon": agent.get_epsilon(step_count),
                              "eval_reward": last_eval_r}
                    stats_file.write(json.dumps(record) + "\n")
                    stats_file.flush()

        # Update progress bar postfix every action
        pbar.set_postfix(
            eps=f"{agent.get_epsilon(step_count):.3f}",
            buf=f"{len(buffer):,}",
            loss=f"{last_loss:.4f}",
            eval=f"{last_eval_r:.1f}",
            refresh=False,
        )

        if done:
            wandb.log({"train/episode_reward": episode_reward}, step=step_count)
            if terminated or truncated:
                # True game over: reset environment
                obs, reset_info = train_env.reset()
                if noop_max > 0:
                    n_noops = np.random.randint(1, noop_max + 1)
                    for _ in range(n_noops):
                        obs, _, _nt, _ntr, reset_info = train_env.step(0)
                        if _nt or _ntr:
                            obs, reset_info = train_env.reset()
                preprocessed = preprocess(obs)
                prev_lives   = reset_info.get("lives", float("inf"))
            else:
                # Life lost but game continues: env keeps running, reset frame stack only
                preprocessed = next_frame
            phi            = frame_stack.reset(preprocessed)
            episode_reward = 0.0
        else:
            preprocessed = next_frame
            phi          = phi_next

        # LR warmdown: linear decay from cfg["lr"] → lr_min starting at lr_decay_start_step
        if lr_decay_start > 0 and step_count >= lr_decay_start:
            frac = min((step_count - lr_decay_start) / max(total_steps - lr_decay_start, 1), 1.0)
            new_lr = cfg["lr"] * (1.0 - frac) + lr_min * frac
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        if step_count % ckpt_freq == 0:
            path = save_checkpoint(network, optimizer, step_count, checkpoint_dir)
            pbar.write(f"  checkpoint → {path}")

    pbar.close()
    stats_file.close()

    # ── Final evaluation (primary agent metric) ────────────────────────────────
    final_eval_n    = cfg.get("final_eval_episodes", 100)
    eval_ckpt_step  = total_steps

    if use_best_ckpt_eval and os.path.exists(best_ckpt_path):
        print(f"\nLoading peak checkpoint (step {best_eval_step:,}, "
              f"1-ep eval={best_eval_reward:.1f})...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        network.load_state_dict(ckpt["network_state"])
        eval_ckpt_step = best_eval_step

    print(f"\nRunning final evaluation ({final_eval_n} episodes, "
          f"checkpoint step={eval_ckpt_step:,})...")
    final_reward = evaluate(agent, eval_env, final_eval_n, device, frame_skip,
                            num_frames=num_frames, frame_stack_mode=frame_stack_mode,
                            noop_max=noop_max)
    print(f"FINAL EVAL ({final_eval_n} eps): {final_reward:.2f}")
    wandb.log({"eval/final_reward": final_reward, "eval/final_ckpt_step": eval_ckpt_step})
    with open(os.path.join(checkpoint_dir, "final_eval.json"), "w") as f:
        json.dump({"final_eval_reward": final_reward, "episodes": final_eval_n,
                   "step": total_steps, "eval_checkpoint_step": eval_ckpt_step,
                   "run_id": args.run_id, "seed": args.seed}, f, indent=2)

    train_env.close()
    eval_env.close()
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
