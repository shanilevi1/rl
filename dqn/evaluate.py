import numpy as np
import torch

from .env import preprocess, FrameStack


def evaluate(agent, env, num_episodes: int, device, frame_skip: int = 4,
             num_frames: int = 4, frame_stack_mode: str = "stack",
             noop_max: int = 0) -> float:
    """Run num_episodes full episodes in eval mode (eps=eps_eval), return mean reward.
    Uses frame skip for consistency with training. No replay buffer writes."""
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        if noop_max > 0:
            n_noops = np.random.randint(1, noop_max + 1)
            for _ in range(n_noops):
                obs, _, term, trunc, _ = env.step(0)
                if term or trunc:
                    obs, _ = env.reset()
        fs = FrameStack(num_frames=num_frames, mode=frame_stack_mode)
        phi = fs.reset(preprocess(obs))
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(phi, step=0, eval_mode=True)

            total_r = 0.0
            for _ in range(frame_skip):
                obs, r, terminated, truncated, _ = env.step(action)
                total_r += r
                done = terminated or truncated
                if done:
                    break

            phi = fs.step(preprocess(obs))
            episode_reward += total_r

        rewards.append(episode_reward)

    return float(np.mean(rewards))
