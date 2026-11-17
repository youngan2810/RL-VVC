"""Isolated, replay-based Dueling Double DQN for five discrete ROI actions.

No claim of superior coding efficiency is implied by this implementation.
"""
from __future__ import annotations
from collections import deque
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QpDuelingNet(nn.Module):
    def __init__(self, n_actions=5):
        super().__init__()
        if n_actions < 2:
            raise ValueError('Need >=2 actions')
        self.encoder = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((5, 7)), nn.Flatten())
        self.fuse = nn.Sequential(nn.Linear(32 * 35 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU())
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, n_actions)
        for layer in (self.value, self.advantage):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, observation):
        local, glob = observation['mb'], observation['global']
        if local.ndim != 4 or tuple(local.shape[1:]) != (7, 20, 30):
            raise ValueError('Expected [B,7,20,30] local map')
        if glob.ndim != 2 or tuple(glob.shape) != (local.shape[0], 32):
            raise ValueError('Expected [B,32] global state')
        hidden = self.fuse(torch.cat([self.encoder(local), glob], dim=1))
        value, advantage = self.value(hidden), self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class Replay:
    def __init__(self, capacity=10000, seed=20261111):
        if capacity < 1:
            raise ValueError('Invalid replay capacity')
        self.rows = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def pack(obs):
        if set(obs) != {'global', 'mb'}:
            raise ValueError('Unexpected state keys')
        g = np.asarray(obs['global'], dtype=np.float32)
        m = np.asarray(obs['mb'], dtype=np.float32)
        if g.shape != (32,) or m.shape != (7, 20, 30) or not (np.isfinite(g).all() and np.isfinite(m).all()):
            raise ValueError('Invalid D3QN state')
        return (g.copy(), m.astype(np.float16))

    def add(self, obs, action, reward, nxt, done):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 5 or not math.isfinite(reward):
            raise ValueError('Invalid D3QN transition')
        self.rows.append((self.pack(obs), int(action), float(reward), self.pack(nxt), bool(done)))

    def sample(self, count, device):
        indices = self.rng.choice(len(self.rows), size=count, replace=False)
        batch = [self.rows[i] for i in indices]
        def observations(which):
            return {'global': torch.as_tensor(np.stack([r[which][0] for r in batch]), device=device),
                    'mb': torch.as_tensor(np.stack([r[which][1] for r in batch]).astype(np.float32), device=device)}
        return (observations(0),
                torch.as_tensor([r[1] for r in batch], dtype=torch.int64, device=device),
                torch.as_tensor([r[2] for r in batch], dtype=torch.float32, device=device),
                observations(3),
                torch.as_tensor([r[4] for r in batch], dtype=torch.bool, device=device))

    def __len__(self):
        return len(self.rows)


class D3QN:
    def __init__(self, *, seed=20261111, device='cpu', lr=1e-4, gamma=1.0,
                 batch_size=64, warmup=512, train_every=4, sync_every=256,
                 replay_capacity=10000, reward_scale=10.0):
        if not (0 <= gamma <= 1 and batch_size >= 2 and warmup >= batch_size and
                train_every >= 1 and sync_every >= 1 and reward_scale > 0):
            raise ValueError('Invalid D3QN hyperparameters')
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.online = QpDuelingNet().to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = Replay(replay_capacity, seed)
        self.rng = np.random.default_rng(seed)
        self.gamma, self.batch_size = gamma, batch_size
        self.warmup, self.train_every, self.sync_every = warmup, train_every, sync_every
        self.reward_scale = reward_scale
        self.steps = self.updates = 0

    def predict(self, obs, epsilon=0.0):
        if not 0 <= epsilon <= 1:
            raise ValueError('Epsilon must be within [0,1]')
        if self.rng.random() < epsilon:
            return int(self.rng.integers(5))
        with torch.no_grad():
            glob, local = Replay.pack(obs)
            vals = self.online({'global': torch.as_tensor(glob[None], device=self.device),
                                'mb': torch.as_tensor(local[None].astype(np.float32), device=self.device)})
        return int(vals.argmax(dim=1).item())

    def observe(self, obs, action, reward, nxt, done):
        self.replay.add(obs, action, reward * self.reward_scale, nxt, done)
        self.steps += 1
        if self.steps % self.train_every or len(self.replay) < self.warmup:
            return None
        obs_b, act, rewards, nxt_b, dones = self.replay.sample(self.batch_size, self.device)
        with torch.no_grad():
            # Double DQN: online SELECTS, independent target EVALUATES.
            next_action = self.online(nxt_b).argmax(dim=1, keepdim=True)
            next_q = self.target(nxt_b).gather(1, next_action).squeeze(1)
            td_target = rewards + self.gamma * next_q * (~dones).float()
        prediction = self.online(obs_b).gather(1, act[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(prediction, td_target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite D3QN TD loss')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 5., error_if_nonfinite=True)
        self.opt.step()
        self.updates += 1
        if self.updates % self.sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {'loss': float(loss.detach()), 'mean_abs_td_error': float((prediction-td_target).abs().mean().detach()),
                'gradient_norm': float(grad_norm), 'q_gap': float((self.online(obs_b).max(1).values-
                                                                  self.online(obs_b).min(1).values).mean().detach())}

    def save(self, path):
        torch.save({'algorithm': 'Dueling Double DQN', 'network': self.online.state_dict(),
                    'target': self.target.state_dict(), 'optimizer': self.opt.state_dict(),
                    'steps': self.steps, 'updates': self.updates,
                    'spec': {'gamma': self.gamma, 'reward_scale': self.reward_scale,
                             'warmup': self.warmup, 'train_every': self.train_every,
                             'sync_every': self.sync_every}}, path)
