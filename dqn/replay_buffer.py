import numpy as np
import torch


class ReplayBuffer:
    """
    Ring buffer storing one preprocessed (84,84) uint8 frame per step.
    States (4-frame stacks) are reconstructed at sample time, which uses
    ~7 GB for 1M capacity vs ~28 GB for pre-stacked storage.
    Episode boundaries are handled by zeroing frames across terminal transitions.
    """

    def __init__(self, capacity: int, frame_history: int = 4, frame_stack_mode: str = "stack"):
        self._cap = capacity
        self._H = frame_history
        self._mode = frame_stack_mode
        self._frames   = np.zeros((capacity, 84, 84), dtype=np.uint8)
        self._actions  = np.zeros(capacity, dtype=np.int32)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=bool)
        self._ptr  = 0
        self._size = 0

    def store(self, frame: np.ndarray, action: int, reward: float, done: bool):
        """Store a single (84,84) uint8 frame and transition metadata."""
        self._frames[self._ptr]  = frame
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr]   = done
        self._ptr  = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def _stack(self, idx: int) -> np.ndarray:
        """Reconstruct an H-frame state ending at buffer index idx.
        Slots that cross an episode boundary are zero-padded.
        In diff mode, returns H consecutive centered frame differences."""
        # need H frames for stack, H+1 for diff
        n_needed = self._H + (1 if self._mode == "diff" else 0)
        raw = np.zeros((n_needed, 84, 84), dtype=np.uint8)
        raw[n_needed - 1] = self._frames[idx]
        for k in range(1, n_needed):
            prev = (idx - k) % self._cap
            if self._dones[prev]:
                break
            raw[n_needed - 1 - k] = self._frames[prev]
        if self._mode == "stack":
            return raw
        # diff: (f[i+1] - f[i]) centered at 128 as uint8
        diffs = []
        for i in range(self._H):
            d = raw[i + 1].astype(np.int16) - raw[i].astype(np.int16)
            diffs.append(np.clip(d // 2 + 128, 0, 255).astype(np.uint8))
        return np.array(diffs, dtype=np.uint8)

    def sample(self, batch_size: int, device: torch.device):
        """Sample a random batch. Returns 5 tensors (float32 on device for frames)."""
        assert self._size > batch_size, "Not enough transitions in buffer"

        idxs = []
        while len(idxs) < batch_size:
            i = np.random.randint(0, self._size - 1)  # -1: phi_next needs i+1
            # When buffer is full, avoid indices whose lookback overlaps the write pointer
            if self._size == self._cap:
                danger = any(
                    (i + k) % self._cap == self._ptr
                    for k in range(-self._H + 1, 2)
                )
                if danger:
                    continue
            idxs.append(i)

        idxs = np.array(idxs)
        phi_b      = np.stack([self._stack(i) for i in idxs])
        phi_next_b = np.stack([self._stack((i + 1) % self._cap) for i in idxs])

        # Single GPU transfer; /255 on device
        phi_t      = torch.from_numpy(phi_b).to(device, non_blocking=True).float().div_(255.0)
        phi_next_t = torch.from_numpy(phi_next_b).to(device, non_blocking=True).float().div_(255.0)
        actions_t  = torch.from_numpy(self._actions[idxs]).long().to(device, non_blocking=True)
        rewards_t  = torch.from_numpy(self._rewards[idxs]).to(device, non_blocking=True)
        dones_t    = torch.from_numpy(self._dones[idxs]).to(device, non_blocking=True)

        return phi_t, actions_t, rewards_t, phi_next_t, dones_t

    def __len__(self) -> int:
        return self._size
