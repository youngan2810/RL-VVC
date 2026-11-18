"""V6 action-conditioned Double DQN with coverage-aware data collection.

The native 25-action QP interface, objective and V3 25-D causal observations
are intentionally unchanged. The Q-function and replay/behavior learner are
new. No future reconstructed pixels or target metrics enter observations.
"""
from __future__ import annotations

from collections import deque
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GLOBAL_DIM = 25
MAP_SHAPE = (5, 10, 15)
N_ACTIONS = 25
QP_VALUES = (-2, -1, 0, 1, 2)
NEUTRAL_ACTION = 12
CHECKPOINT_VERSION = 'd3qn-native-checkpoint/v6-action-conditioned'


def action_coordinates():
    """Fixed physically meaningful action coordinates, not arbitrary IDs."""
    return torch.tensor([[r / 2., b / 2., (r - b) / 4., (r + b) / 4., r * b / 4.]
                         for r in QP_VALUES for b in QP_VALUES], dtype=torch.float32)


def pack(obs):
    if not isinstance(obs, dict) or set(obs) != {'global', 'maps', 'legal'}:
        raise ValueError('V6 observation must contain global/maps/legal only')
    g = np.asarray(obs['global'], dtype=np.float32)
    m = np.asarray(obs['maps'], dtype=np.float32)
    l = np.asarray(obs['legal'])
    if (g.shape != (GLOBAL_DIM,) or m.shape != MAP_SHAPE or
            l.shape != (N_ACTIONS,) or l.dtype != np.bool_ or not l.any() or
            not np.isfinite(g).all() or not np.isfinite(m).all()):
        raise ValueError('Invalid or noncausal V6 observation contract')
    return g.copy(), m.astype(np.float16), l.copy()


def batch_obs(items, device):
    return {'global': torch.as_tensor(np.stack([v[0] for v in items]), dtype=torch.float32, device=device),
            'maps': torch.as_tensor(np.stack([v[1] for v in items]).astype(np.float32), device=device),
            'legal': torch.as_tensor(np.stack([v[2] for v in items]), dtype=torch.bool, device=device)}


class ActionConditionedQNetwork(nn.Module):
    """Physical ROI/BG embeddings interact with the current video state.

    A shared context produces V(s), and a bilinear action-conditioned term
    produces centered A(s,a). Legal centering does not change argmax.
    """
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(),
            nn.Linear(640, 96), nn.SiLU())
        self.codec = nn.Sequential(nn.LayerNorm(GLOBAL_DIM), nn.Linear(GLOBAL_DIM, 96), nn.SiLU(),
                                   nn.Linear(96, 64), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(160, 128), nn.SiLU(),
                                    nn.Linear(128, 96), nn.SiLU())
        self.value = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
        self.state_action = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 32))
        self.action_embedding = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 32))
        self.action_bias = nn.Linear(5, 1, bias=False)
        self.register_buffer('action_coords', action_coordinates())
        # Start with a low-magnitude Q ranking instead of an arbitrary high-confidence action.
        nn.init.orthogonal_(self.state_action[-1].weight, gain=.02)
        nn.init.zeros_(self.state_action[-1].bias)
        nn.init.zeros_(self.action_bias.weight)
        nn.init.orthogonal_(self.value[-1].weight, gain=.02)
        nn.init.zeros_(self.value[-1].bias)

    def components(self, obs):
        g, m, legal = obs['global'], obs['maps'], obs['legal']
        if (g.ndim != 2 or g.shape[1] != GLOBAL_DIM or m.ndim != 4 or
                tuple(m.shape[1:]) != MAP_SHAPE or m.shape[0] != g.shape[0] or
                tuple(legal.shape) != (g.shape[0], N_ACTIONS) or
                legal.dtype != torch.bool or not bool(legal.any(dim=1).all().item())):
            raise ValueError('Invalid batched V6 observation')
        context = self.fusion(torch.cat((self.codec(g), self.spatial(m)), dim=1))
        value = self.value(context)
        action = self.action_embedding(self.action_coords)
        relative = (self.state_action(context) @ action.T) / math.sqrt(32.)
        advantage = relative + self.action_bias(self.action_coords).T
        mean_legal = (advantage * legal).sum(dim=1, keepdim=True) / legal.sum(dim=1, keepdim=True)
        q = value + advantage - mean_legal
        return q, value, advantage

    def forward(self, obs):
        return self.components(obs)[0]


class ActionBalancedReplay:
    """Mixture sampling by action and coding boundary with exact IS weights.

    Sampling is with replacement. A rare action receives more drawing
    probability; per-draw importance weights correct the mixture's induced
    sampling distribution (up to minibatch max-normalization).
    """
    def __init__(self, capacity=20000, seed=20261111, action_mix=.45, boundary_mix=.10):
        if capacity < 1 or action_mix < 0 or boundary_mix < 0 or action_mix + boundary_mix >= 1:
            raise ValueError('Invalid replay configuration')
        self.rows = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
        self.action_mix = float(action_mix)
        self.boundary_mix = float(boundary_mix)

    def __len__(self):
        return len(self.rows)

    def append(self, obs, action, reward, next_obs, done, horizon, coding_step):
        a = int(action)
        pre, post = pack(obs), pack(next_obs)
        if (not 0 <= a < N_ACTIONS or not pre[2][a] or not math.isfinite(reward)
                or not isinstance(horizon, int) or not 1 <= horizon <= 256
                or not isinstance(coding_step, int) or coding_step < 0):
            raise ValueError('Invalid replay transition')
        self.rows.append((pre, a, float(reward), post, bool(done), int(horizon), int(coding_step)))

    def probabilities(self):
        n = len(self.rows)
        if not n:
            raise ValueError('Empty replay')
        acts = np.fromiter((r[1] for r in self.rows), dtype=np.int64, count=n)
        boundaries = np.fromiter((int(r[6] % 32 == 0) for r in self.rows), dtype=np.int64, count=n)
        counts = np.bincount(acts, minlength=N_ACTIONS)
        n_acts = np.count_nonzero(counts)
        b_counts = np.bincount(boundaries, minlength=2)
        n_bounds = np.count_nonzero(b_counts)
        p = np.full(n, (1. - self.action_mix - self.boundary_mix) / n)
        p += self.action_mix / (n_acts * counts[acts])
        p += self.boundary_mix / (n_bounds * b_counts[boundaries])
        if not np.isclose(p.sum(), 1., atol=1e-12) or np.any(p <= 0):
            raise AssertionError('Replay probabilities are not normalized')
        return p

    def sample(self, batch_size, device):
        n = len(self.rows)
        if n < batch_size:
            raise ValueError('Replay warmup incomplete')
        p = self.probabilities()
        indices = self.rng.choice(n, size=batch_size, replace=True, p=p)
        sampled = [self.rows[int(i)] for i in indices]
        # Self-normalized full-importance correction; no 1/p training explosion.
        weights = (1. / (n * p[indices]))
        weights /= weights.max()
        obs = batch_obs([r[0] for r in sampled], device)
        nxt = batch_obs([r[3] for r in sampled], device)
        return (obs,
                torch.tensor([r[1] for r in sampled], dtype=torch.long, device=device),
                torch.tensor([r[2] for r in sampled], dtype=torch.float32, device=device),
                nxt,
                torch.tensor([r[4] for r in sampled], dtype=torch.bool, device=device),
                torch.tensor([r[5] for r in sampled], dtype=torch.float32, device=device),
                torch.tensor(weights, dtype=torch.float32, device=device),
                {'sampled_terminal': int(sum(r[4] for r in sampled)),
                 'sampled_boundary': int(sum(r[6] % 32 == 0 for r in sampled)),
                 'unique_sampled_actions': len({r[1] for r in sampled}),
                 'importance_weight_mean': float(weights.mean())})

    def state_dict(self):
        rows = list(self.rows)
        def os_stack(slot, part):
            return torch.as_tensor(np.stack([r[slot][part] for r in rows]).copy())
        if not rows:
            raise ValueError('Empty replay cannot be serialized')
        return {'capacity': self.rows.maxlen,
                'action_mix': self.action_mix, 'boundary_mix': self.boundary_mix,
                'obs_global': os_stack(0, 0), 'obs_maps': os_stack(0, 1), 'obs_legal': os_stack(0, 2),
                'actions': torch.tensor([r[1] for r in rows]),
                'rewards': torch.tensor([r[2] for r in rows], dtype=torch.float32),
                'next_global': os_stack(3, 0), 'next_maps': os_stack(3, 1),
                'next_legal': os_stack(3, 2), 'dones': torch.tensor([r[4] for r in rows]),
                'horizons': torch.tensor([r[5] for r in rows]),
                'coding_steps': torch.tensor([r[6] for r in rows]),
                'rng_state': self.rng.bit_generator.state}

    def load_state_dict(self, payload):
        if (payload['capacity'] != self.rows.maxlen or
                payload['action_mix'] != self.action_mix or
                payload['boundary_mix'] != self.boundary_mix):
            raise ValueError('Replay identity mismatch')
        n = len(payload['actions'])
        if not (1 <= n <= self.rows.maxlen):
            raise ValueError('Invalid replay size')
        for key in ('obs_global', 'obs_maps', 'obs_legal', 'rewards', 'next_global',
                    'next_maps', 'next_legal', 'dones', 'horizons', 'coding_steps'):
            if len(payload[key]) != n:
                raise ValueError('Inconsistent replay tensor lengths')
        self.rows.clear()
        for i in range(n):
            obs = {'global': payload['obs_global'][i].numpy(),
                   'maps': payload['obs_maps'][i].numpy(),
                   'legal': payload['obs_legal'][i].numpy()}
            nxt = {'global': payload['next_global'][i].numpy(),
                   'maps': payload['next_maps'][i].numpy(),
                   'legal': payload['next_legal'][i].numpy()}
            self.append(obs, int(payload['actions'][i]), float(payload['rewards'][i]), nxt,
                        bool(payload['dones'][i]), int(payload['horizons'][i]),
                        int(payload['coding_steps'][i]))
        self.rng.bit_generator.state = payload['rng_state']


def exploration_epsilon(steps, *, start=1.0, floor=.20, decay_steps=10240):
    if steps < 0 or not (0 <= floor <= start <= 1) or decay_steps < 1:
        raise ValueError('Invalid exploration schedule')
    return max(floor, start - (start - floor) * min(steps / decay_steps, 1.))


class StableActionConditionedDQN:
    def __init__(self, *, seed=20261111, device='cpu', lr=5e-5, gamma=1.,
                 n_step=8, target_tau=.005, batch_size=64,
                 replay_capacity=20000, warmup=512, train_every=4,
                 reward_scale=25., action_mix=.45, boundary_mix=.10):
        if (gamma != 1. or not 1 <= n_step <= 32 or lr <= 0 or
                not 0 < target_tau <= 1 or not 2 <= batch_size <= warmup or
                train_every < 1 or reward_scale <= 0):
            raise ValueError('Invalid V6 learner settings')
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.online = ActionConditionedQNetwork().to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ActionBalancedReplay(replay_capacity, seed, action_mix, boundary_mix)
        self.rng = np.random.default_rng(seed)
        self.pending = deque()
        self.coverage_counts = np.zeros((5, N_ACTIONS), dtype=np.int64)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.target_tau = float(target_tau)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.train_every = int(train_every)
        self.reward_scale = float(reward_scale)
        self.steps = 0
        self.updates = 0

    @staticmethod
    def _rate_index(obs):
        # global[2] = log(target_bps / 120000), shared native V3 contract.
        rate = 120 * math.exp(float(obs['global'][2]))
        return int(np.argmin(np.abs(np.asarray((50., 80., 120., 160., 200.)) - rate)))

    def predict(self, obs, epsilon=0.):
        packed = pack(obs)
        if not 0 <= epsilon <= 1:
            raise ValueError('Invalid epsilon')
        legal = np.flatnonzero(packed[2])
        if self.rng.random() < epsilon:
            counts = self.coverage_counts[self._rate_index(obs), legal]
            least_visited = legal[counts == counts.min()]
            return int(self.rng.choice(least_visited))
        self.online.eval()
        with torch.no_grad():
            q = self.online(batch_obs([packed], self.device))[0]
            q = q.masked_fill(~torch.as_tensor(packed[2], device=self.device), float('-inf'))
        return int(q.argmax().item())

    def _emit_nstep(self):
        horizon = min(self.n_step, len(self.pending))
        data = list(self.pending)[:horizon]
        actual = 0
        reward_sum = 0.
        for item in data:
            reward_sum += item[2]
            actual += 1
            if item[4]:
                break
        head = data[0]
        tail = data[actual - 1]
        self.replay.append(head[0], head[1], reward_sum, tail[3], tail[4], actual, head[5])
        self.pending.popleft()

    def observe(self, obs, action, reward, next_obs, done):
        pack(obs)
        pack(next_obs)
        if not np.isfinite(reward) or not obs['legal'][int(action)]:
            raise ValueError('Invalid action/reward observed')
        coding_step = self.steps % 256
        rate = self._rate_index(obs)
        self.coverage_counts[rate, int(action)] += 1
        self.pending.append((obs, int(action), float(reward) * self.reward_scale,
                             next_obs, bool(done), int(coding_step)))
        self.steps += 1
        if done:
            while self.pending:
                self._emit_nstep()
        elif len(self.pending) >= self.n_step:
            self._emit_nstep()
        if len(self.replay) < self.warmup or self.steps % self.train_every:
            return None
        return self._update()

    def _update(self):
        obs, actions, rewards, nxt, dones, horizons, weights, extra = self.replay.sample(
            self.batch_size, self.device)
        self.online.train()
        with torch.no_grad():
            greedy_nxt = self.online(nxt).masked_fill(~nxt['legal'], float('-inf')).argmax(dim=1)
            bootstrap = self.target(nxt).gather(1, greedy_nxt[:, None]).squeeze(1)
            target = rewards + (self.gamma ** horizons) * bootstrap * (~dones).float()
        q = self.online(obs)
        pred = q.gather(1, actions[:, None]).squeeze(1)
        td = target - pred
        loss = (weights * F.smooth_l1_loss(pred, target, reduction='none')).sum() / weights.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite V6 loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.online.parameters(), 5., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        with torch.no_grad():
            for target_p, online_p in zip(self.target.parameters(), self.online.parameters()):
                target_p.lerp_(online_p, self.target_tau)
            q_eval = self.online(obs).masked_fill(~obs['legal'], float('-inf'))
            min_legal = self.online(obs).masked_fill(~obs['legal'], float('inf')).min(dim=1).values
            best = q_eval.max(dim=1).values
        return {'loss': float(loss.detach()), 'mean_abs_td': float(td.abs().mean().detach()),
                'grad_norm': float(grad), 'updates': self.updates,
                'q_gap_valid': float((best - min_legal).mean()),
                'q_max_valid': float(best.mean()),
                'unique_greedy_actions_in_batch': int(torch.unique(q_eval.argmax(dim=1)).numel()),
                'sampled_terminal': extra['sampled_terminal'],
                'sampled_boundary': extra['sampled_boundary'],
                'unique_sampled_actions': extra['unique_sampled_actions'],
                'importance_weight_mean': extra['importance_weight_mean'],
                'least_exposed_action_rate': int(self.coverage_counts.min())}

    def save(self, path, *, identity, next_episode_seed=None):
        if self.steps < 256 or self.steps % 256 or self.pending:
            raise ValueError('V6 checkpoint only at completed episode boundary')
        if next_episode_seed is None:
            raise ValueError('Next episode seed is required')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': CHECKPOINT_VERSION, 'identity': identity,
                   'next_episode_seed': int(next_episode_seed),
                   'online': self.online.state_dict(), 'target': self.target.state_dict(),
                   'optimizer': self.optimizer.state_dict(),
                   'steps': self.steps, 'updates': self.updates,
                   'config': {'lr': self.optimizer.param_groups[0]['lr'], 'gamma': self.gamma,
                              'n_step': self.n_step, 'target_tau': self.target_tau,
                              'batch_size': self.batch_size, 'warmup': self.warmup,
                              'train_every': self.train_every, 'reward_scale': self.reward_scale,
                              'replay_capacity': self.replay.rows.maxlen,
                              'action_mix': self.replay.action_mix,
                              'boundary_mix': self.replay.boundary_mix},
                   'replay': self.replay.state_dict(),
                   'coverage_counts': torch.as_tensor(self.coverage_counts.copy()),
                   'agent_rng_state': self.rng.bit_generator.state,
                   'torch_cpu_rng_state': torch.get_rng_state(),
                   'torch_cuda_rng_states': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []}
        temp = path.with_suffix('.tmp')
        torch.save(payload, temp)
        temp.replace(path)

    @classmethod
    def load(cls, path, *, identity, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('version') != CHECKPOINT_VERSION or payload.get('identity') != identity:
            raise ValueError('V6 checkpoint version or experiment identity mismatch')
        cfg = payload['config']
        agent = cls(seed=identity['seed'], device=device, **cfg)
        agent.online.load_state_dict(payload['online'])
        agent.target.load_state_dict(payload['target'])
        agent.optimizer.load_state_dict(payload['optimizer'])
        for state in agent.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(agent.device)
        agent.steps, agent.updates = int(payload['steps']), int(payload['updates'])
        agent.replay.load_state_dict(payload['replay'])
        agent.coverage_counts = payload['coverage_counts'].numpy().astype(np.int64).copy()
        agent.rng.bit_generator.state = payload['agent_rng_state']
        torch.set_rng_state(payload['torch_cpu_rng_state'])
        if agent.device.type == 'cuda' and payload['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(payload['torch_cuda_rng_states'])
        return agent, int(payload['next_episode_seed'])
