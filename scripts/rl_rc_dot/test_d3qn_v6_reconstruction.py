"""Local V6 learner checks. Synthetic rewards are NOT video coding evidence."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .d3qn_v6_agent import (
    ActionConditionedQNetwork, ActionBalancedReplay, StableActionConditionedDQN,
    action_coordinates, batch_obs, exploration_epsilon, pack,
)


def observation(step=0, *, x=0., rate=120, only=None):
    g = np.zeros(25, dtype=np.float32)
    g[0] = x
    g[1] = (step % 256) / 256.
    g[2] = np.log(rate / 120.)
    maps = np.zeros((5, 10, 15), dtype=np.float32)
    maps[4] = .5
    legal = np.ones(25, dtype=np.bool_)
    if only is not None:
        legal[:] = False
        legal[only] = True
    return {'global': g, 'maps': maps, 'legal': legal}


class V6ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_physical_action_coordinates(self):
        coords = action_coordinates()
        self.assertEqual(tuple(coords.shape), (25, 5))
        self.assertTrue(torch.allclose(coords[2, :2], torch.tensor([-1., 0.])))
        self.assertTrue(torch.allclose(coords[9, :2], torch.tensor([-.5, 1.])))

    def test_strict_input_schema_and_no_future_fields(self):
        self.assertEqual(pack(observation())[0].shape, (25,))
        bad = observation()
        bad['global'] = np.zeros(28, dtype=np.float32)
        with self.assertRaises(ValueError):
            pack(bad)
        bad = observation()
        bad['future_f1'] = .9
        with self.assertRaises(ValueError):
            pack(bad)

    def test_q_gap_dueling_identity_and_gradient(self):
        torch.manual_seed(123)
        net = ActionConditionedQNetwork()
        obs = batch_obs([pack(observation(x=x)) for x in (-.1, .5)], 'cpu')
        q, value, advantage = net.components(obs)
        self.assertEqual(tuple(q.shape), (2, 25))
        self.assertTrue(torch.allclose(q[:, 2]-q[:, 9], advantage[:, 2]-advantage[:, 9], atol=1e-6))
        net.zero_grad(set_to_none=True)
        (q[:, 2] - q[:, 9]).sum().backward()
        self.assertTrue(net.value[-1].weight.grad is None or
                        torch.all(net.value[-1].weight.grad == 0))
        self.assertGreater(float(net.action_embedding[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(net.state_action[-1].weight.grad.abs().sum()), 0.)

    def test_legal_mask_respects_mean_and_action_argmax(self):
        net = ActionConditionedQNetwork()
        q = net(batch_obs([pack(observation(only=[2, 9]))], 'cpu'))[0]
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(len(torch.topk(q, 2).indices), 2)

    def test_replay_mixture_is_normalized_and_action_balanced(self):
        rep = ActionBalancedReplay(capacity=1000, seed=1)
        for i in range(300):
            o = observation(i)
            rep.append(o, 2 if i < 290 else 9, .1, o, False, 1, i)
        p = rep.probabilities()
        self.assertAlmostEqual(float(p.sum()), 1., places=10)
        self.assertGreater(float(p[295]), float(p[1]))
        batch = rep.sample(64, 'cpu')
        self.assertEqual(tuple(batch[0]['global'].shape), (64, 25))
        self.assertLessEqual(float(batch[6].max()), 1.00001)
        self.assertTrue(torch.all(batch[6] > 0))

    def test_horizon_and_terminal_flush_telescoping(self):
        agent = StableActionConditionedDQN(n_step=8, warmup=256, batch_size=32)
        obs = observation()
        for i in range(11):
            agent.observe(obs, 2, i / 100, observation(i+1), i == 10)
        rows = list(agent.replay.rows)
        self.assertEqual(len(rows), 11)
        self.assertEqual([r[5] for r in rows[:4]], [8, 8, 8, 8])
        self.assertEqual([r[5] for r in rows[-8:]], [8, 7, 6, 5, 4, 3, 2, 1])
        self.assertTrue(rows[-1][4])
        self.assertAlmostEqual(rows[0][2], 25 * sum(i/100 for i in range(8)), places=5)
        self.assertEqual(len(agent.pending), 0)

    def test_exploration_balances_underrepresented_action(self):
        agent = StableActionConditionedDQN(seed=10)
        a = observation(0, only=[2, 9])
        agent.coverage_counts[2, 2] = 90
        agent.coverage_counts[2, 9] = 0
        for _ in range(12):
            self.assertEqual(agent.predict(a, epsilon=1.), 9)
        self.assertAlmostEqual(exploration_epsilon(5120), .6)

    def test_synthetic_state_specific_relative_action_ranking_can_be_learned(self):
        """Controlled supervised representational sanity, not a VVC result."""
        torch.manual_seed(22)
        net = ActionConditionedQNetwork()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        examples = [pack(observation(x=sign * .75, step=j))
                    for j in range(32) for sign in (-1., 1.)]
        obs = batch_obs(examples, 'cpu')
        labels = torch.tensor([2 if i % 2 == 0 else 9 for i in range(len(examples))])
        for _ in range(250):
            q = net(obs)
            loss = F.cross_entropy(q, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            greedy = net(obs).argmax(dim=1)
        self.assertGreater(float((greedy == labels).float().mean()), .97)

    def test_true_terminal_dqn_learns_state_specific_two_action_choices(self):
        """Actual DDQN optimizer on a synthetic known-optimal terminal MDP."""
        agent = StableActionConditionedDQN(seed=20261111, lr=2e-3, n_step=1,
            target_tau=.02, batch_size=32, warmup=64, train_every=1, reward_scale=1.)
        for i in range(300):
            sign = -1. if i % 2 == 0 else 1.
            obs = observation(i, x=sign, only=[2, 9])
            action = agent.predict(obs, epsilon=.45)
            reward = 1. if action == (2 if sign < 0 else 9) else -1.
            agent.observe(obs, action, reward, observation(i+1, x=sign, only=[2, 9]), True)
        self.assertEqual(agent.predict(observation(x=-1, only=[2, 9]), epsilon=0), 2)
        self.assertEqual(agent.predict(observation(x=1, only=[2, 9]), epsilon=0), 9)

    def test_checkpoint_full_resume_equivalence_at_256_boundary(self):
        kwargs = {'seed': 20261111, 'n_step': 8, 'batch_size': 32, 'warmup': 64,
                  'train_every': 4, 'lr': 2e-4}
        data = []
        for i in range(512):
            data.append((observation(i, x=(-1 if i % 2 else 1) * .5),
                         2 if i % 2 else 9, float(i % 7) / 300,
                         observation(i + 1), (i + 1) % 256 == 0))
        identity = {'seed': 20261111, 'protocol': 'v6-test', 'learning': {'n_step': 8}}
        def train(agent, start, end):
            for i in range(start, end):
                agent.observe(*data[i])
        full = StableActionConditionedDQN(**kwargs)
        train(full, 0, 512)
        first = StableActionConditionedDQN(**kwargs)
        train(first, 0, 256)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'v6.pt'
            first.save(path, identity=identity, next_episode_seed=999)
            self.assertTrue(path.exists())
            resumed, seed = StableActionConditionedDQN.load(path, identity=identity)
            self.assertEqual(seed, 999)
            with self.assertRaises(ValueError):
                StableActionConditionedDQN.load(path, identity={'seed': 0})
            train(resumed, 256, 512)
        self.assertEqual(resumed.steps, full.steps)
        self.assertEqual(resumed.updates, full.updates)
        self.assertEqual(len(resumed.replay), len(full.replay))
        for key in full.online.state_dict():
            self.assertTrue(torch.equal(full.online.state_dict()[key], resumed.online.state_dict()[key]), key)
        np.testing.assert_array_equal(full.coverage_counts, resumed.coverage_counts)


if __name__ == '__main__':
    unittest.main()
