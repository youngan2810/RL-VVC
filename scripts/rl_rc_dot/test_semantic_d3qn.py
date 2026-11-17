"""Unit tests for causal ROI controls and the isolated D3QN update.

Run: python -m unittest scripts.rl_rc_dot.test_semantic_d3qn -v
Native VVenC/VVdeC verification still requires the Kaggle runtime.
"""
import unittest
import numpy as np
import torch

from scripts.rl_rc_dot.semantic_qp import (
    source_ctu_roi, coarse_roi, expanded_roi, template_to_qpbins,
    augment_observation, TEMPLATES)
from scripts.rl_rc_dot.qp_development import map_qp_bins
from scripts.rl_rc_dot.d3qn_qp import D3QN, Replay


class SemanticQP(unittest.TestCase):
    def test_absent_source_detections_do_not_change_qpa(self):
        coarse = coarse_roi(source_ctu_roi(np.empty((0, 5))))
        for action in range(5):
            np.testing.assert_array_equal(map_qp_bins(template_to_qpbins(action, coarse)),
                                          np.zeros((20, 30)))

    def test_integer_qp_survives_qg_map_and_spatial_centers(self):
        boxes = np.array([[64, 64, 95, 95, .9], [350, 100, 390, 130, .75]])
        mask = coarse_roi(source_ctu_roi(boxes))
        effective = expanded_roi(mask)
        self.assertGreater(float(effective.sum()), 0)
        for action, (roi, background) in enumerate(TEMPLATES):
            categories = template_to_qpbins(action, mask)
            actual = map_qp_bins(categories)
            np.testing.assert_array_equal(actual, np.where(effective > 0, roi, background))
            for y in range(0, 20, 2):
                for x in range(0, 30, 2):
                    self.assertEqual(np.unique(actual[y:y+2, x:x+2]).size, 1)
        obs = augment_observation({'global': np.zeros(32, np.float32),
                                   'mb': np.zeros((6, 20, 30), np.float32)}, mask)
        self.assertEqual(obs['mb'].shape, (7, 20, 30))
        np.testing.assert_array_equal(obs['mb'][6], effective)

    def test_invalid_source_boxes_and_actions(self):
        for box in (np.array([[1, 2, 0, 3, .5]]),
                    np.array([[1, 2, 5, 7, np.nan]])):
            with self.assertRaises(ValueError):
                source_ctu_roi(box)
        m = np.zeros((5, 7), bool)
        for action in (-1, 5, 1.5, True):
            with self.assertRaises(ValueError):
                template_to_qpbins(action, m)


class DuelingDoubleDQN(unittest.TestCase):
    def test_target_update_and_finite_td_training(self):
        torch.set_num_threads(1)
        agent = D3QN(seed=7, batch_size=4, warmup=4, train_every=1, sync_every=2)

        def state(seed):
            g = np.zeros(32, np.float32); g[0] = seed
            mb = np.zeros((7, 20, 30), np.float32); mb[6, 0, 0] = seed
            return {'global': g, 'mb': mb}

        action = agent.predict(state(0), epsilon=0.0)
        self.assertTrue(0 <= action < 5)
        losses = []
        for i in range(7):
            outcome = agent.observe(state(i), i % 5, 0.2, state(i+1), i == 6)
            if outcome is not None:
                losses.append(outcome['loss'])
        self.assertEqual(agent.updates, 4)
        self.assertEqual(len(losses), 4)
        self.assertTrue(np.isfinite(losses).all())
        with torch.no_grad():
            result = agent.online({'global': torch.zeros(3, 32),
                                   'mb': torch.zeros(3, 7, 20, 30)})
        self.assertEqual(tuple(result.shape), (3, 5))

    def test_bad_transition(self):
        replay = Replay(capacity=2)
        obs = {'global': np.zeros(32, np.float32),
               'mb': np.zeros((7, 20, 30), np.float32)}
        with self.assertRaises(ValueError):
            replay.add(obs, 9, .5, obs, False)
        with self.assertRaises(ValueError):
            replay.add(obs, 0, float('nan'), obs, False)


if __name__ == '__main__':
    unittest.main()
