"""Non-invasive Discrete(5) wrapper around audited qpdev1-qbins native VVenC.

The source-frame detector is called before encoding (explicit encoder-side
semantic prior), and its result is cached for the existing reward path.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .semantic_qp import (source_ctu_roi, coarse_roi, expanded_roi,
                          template_to_qpbins, augment_observation)


class CachedSourceDetector:
    def __init__(self, detector):
        self.detector = detector
        self.pending = None

    def prime(self, source):
        boxes = np.asarray(self.detector(source)).copy()
        self.pending = (source, boxes)
        return boxes

    def __call__(self, pixels):
        if self.pending is not None:
            source, boxes = self.pending
            if (pixels.shape == source.shape and pixels.dtype == source.dtype and
                    np.shares_memory(pixels, source)):
                self.pending = None
                return boxes.copy()
        return self.detector(pixels)

    def clear(self):
        self.pending = None

    def close(self):
        return self.detector.close()


class SemanticQPWrapper(gym.Wrapper):
    def __init__(self, env):
        if env.agent_profile != 'qpdev1-qbins':
            raise ValueError('D3QN wrapper requires native audited qpdev1-qbins profile')
        super().__init__(env)
        self.env.detector = CachedSourceDetector(env.detector)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'global': spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32),
            'mb': spaces.Box(-np.inf, np.inf, (7, 20, 30), dtype=np.float32)})
        self.coarse = np.zeros((5, 7), bool)

    def _augment(self, observation):
        if self.env.native is None or self.env.native.current is None:
            self.coarse = np.zeros((5, 7), bool)
            return augment_observation(observation, self.coarse)
        poc = self.env.native.current['poc']
        if not isinstance(poc, int) or not 0 <= poc < self.env.frames:
            raise ValueError('Bad causal source POC')
        source = self.env.source[poc]
        raw = self.env.detector.prime(source)
        self.coarse = coarse_roi(source_ctu_roi(raw))
        return augment_observation(observation, self.coarse)

    def reset(self, *, seed=None, options=None):
        self.env.detector.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        categories = template_to_qpbins(action, self.coarse)
        roi_fraction = float(expanded_roi(self.coarse).mean())
        obs, reward, terminated, truncated, info = self.env.step(categories)
        info['d3qn_template'] = int(action)
        info['semantic_roi_fraction'] = roi_fraction
        info['d3qn_categories'] = categories.tolist()
        return self._augment(obs), reward, terminated, truncated, info
