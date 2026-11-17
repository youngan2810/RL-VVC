"""V3 causal-state correction: represent previous ROI and background QP separately.

The native VVenC adapter, source YOLO ROI, action space, and clip reward stay
identical to the audited V2 pilot. This is a deliberately narrow ablation.
"""
from .d3qn_native_env import D3QNNativeEnv
from .d3qn_method import action_pair


class D3QNNativeEnvV3(D3QNNativeEnv):
    def _before_action(self):
        obs=super()._before_action()
        roi,bg=action_pair(self.prev_action)
        # V2 globals: index 5 duplicated 1-progress, index 19 encoded the
        # 25-action categorical ID as a scalar, incorrectly imposing ordinality.
        # V3 exposes the two physical QP offsets instead, same 25-dim shape.
        obs['global'][5]=roi/2.
        obs['global'][19]=bg/2.
        return obs
