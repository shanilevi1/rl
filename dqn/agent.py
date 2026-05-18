from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .network import DQNNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_anneal_steps: int = 1_000_000
    eps_eval: float = 0.05
    gamma: float = 0.99
    batch_size: int = 32
    min_replay_size: int = 50_000
    use_amp: bool = True
    grad_clip: float = 0.0   # 0 = disabled
    loss_fn: str = "mse"     # "mse" or "huber"


class DQNAgent:

    def __init__(
        self,
        network: DQNNetwork,
        replay_buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        config: AgentConfig,
        device: torch.device,
    ):
        self.network = network
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.cfg = config
        self.device = device
        self._use_amp = config.use_amp and device.type == "cuda"

    def get_epsilon(self, step: int) -> float:
        progress = min(1.0, step / max(1, self.cfg.eps_anneal_steps))
        return self.cfg.eps_start + progress * (self.cfg.eps_end - self.cfg.eps_start)

    def select_action(self, phi: np.ndarray, step: int, eval_mode: bool = False) -> int:
        eps = self.cfg.eps_eval if eval_mode else self.get_epsilon(step)
        if np.random.random() < eps:
            return int(np.random.randint(self.network.num_actions))
        phi_t = torch.from_numpy(phi).unsqueeze(0).float().div_(255.0).to(self.device)
        with torch.no_grad():
            return int(self.network(phi_t).argmax(dim=1).item())

    def store_transition(self, frame: np.ndarray, action: int, reward: float, done: bool):
        self.replay_buffer.store(frame, action, float(np.sign(reward)), done)

    def train_step(self):
        if len(self.replay_buffer) < self.cfg.min_replay_size:
            return None

        phi, actions, rewards, phi_next, dones = self.replay_buffer.sample(
            self.cfg.batch_size, self.device
        )

        self.optimizer.zero_grad()

        if self._use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, q_pred, targets = self._bellman_loss(phi, actions, rewards, phi_next, dones)
        else:
            loss, q_pred, targets = self._bellman_loss(phi, actions, rewards, phi_next, dones)

        loss.backward()

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.network.parameters()
            if p.grad is not None
        ) ** 0.5

        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_clip)

        self.optimizer.step()

        with torch.no_grad():
            td_error = (q_pred.detach() - targets).abs().mean()

        return {
            "loss":        loss.item(),
            "grad_norm":   grad_norm,
            "q_mean":      q_pred.detach().mean().item(),
            "q_max":       q_pred.detach().max().item(),
            "q_std":       q_pred.detach().std().item(),
            "target_mean": targets.mean().item(),
            "target_std":  targets.std().item(),
            "td_error":    td_error.item(),
        }

    def _bellman_loss(self, phi, actions, rewards, phi_next, dones):
        q_next = self.network(phi_next).detach().max(dim=1).values
        targets = (rewards + self.cfg.gamma * q_next * (1.0 - dones.float())).detach()
        q_pred = self.network(phi).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, targets) if self.cfg.loss_fn == "huber" else F.mse_loss(q_pred, targets)
        return loss, q_pred, targets
