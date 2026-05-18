import numpy as np
import cv2
from collections import deque
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB (210,160,3) uint8 → grayscale (84,84) uint8. Paper preprocessing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # resize: (width=84, height=110) in cv2 convention → numpy shape (110, 84)
    resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
    return resized[18:102, :]  # crop to (84, 84), uint8


class FrameStack:
    """Rolling window of the last N preprocessed frames.

    mode='stack': returns N raw frames stacked → (N, 84, 84) uint8
    mode='diff':  returns N consecutive frame differences, centered at 128
                  → (N, 84, 84) uint8.  128 = no motion, <128 = darker, >128 = brighter.
    """

    def __init__(self, num_frames: int = 4, mode: str = "stack"):
        self._n = num_frames
        self._mode = mode
        # diff mode needs one extra frame to compute the first difference
        self._frames = deque(maxlen=num_frames + (1 if mode == "diff" else 0))

    def reset(self, frame: np.ndarray) -> np.ndarray:
        n_fill = self._n + (1 if self._mode == "diff" else 0)
        for _ in range(n_fill):
            self._frames.append(frame)
        return self._get()

    def step(self, frame: np.ndarray) -> np.ndarray:
        self._frames.append(frame)
        return self._get()

    def _get(self) -> np.ndarray:
        if self._mode == "stack":
            return np.array(self._frames, dtype=np.uint8)
        # diff: f[i+1] - f[i] shifted to uint8 centre 128
        frames = list(self._frames)
        diffs = []
        for i in range(self._n):
            d = frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)
            diffs.append(np.clip(d // 2 + 128, 0, 255).astype(np.uint8))
        return np.array(diffs, dtype=np.uint8)


def make_env(ale_game_id: str, seed: int = None) -> gym.Env:
    # frameskip=1 → NoFrameskip; we implement our own skip in the training loop
    # repeat_action_probability=0.0 → deterministic actions (matches v4 behaviour)
    env = gym.make(ale_game_id, frameskip=1, repeat_action_probability=0.0, render_mode=None)
    if seed is not None:
        env.reset(seed=seed)
    return env
