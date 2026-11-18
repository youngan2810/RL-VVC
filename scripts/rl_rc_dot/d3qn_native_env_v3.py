"""V3 causal-state correction: represent previous ROI and background QP separately.

The native VVenC adapter, source YOLO ROI, action space, and clip reward stay
identical to the audited V2 pilot. This is a deliberately narrow ablation.
"""
import json
import numpy as np
from .d3qn_native_env import D3QNNativeEnv
from .d3qn_method import action_pair


class D3QNNativeEnvV3(D3QNNativeEnv):
    def reset(self, *, seed=None):
        obs,meta=super().reset(seed=seed)
        self._v3_roi_frames=0
        self._v3_no_roi_frames=0
        self._v3_roi_fraction_sum=0.
        self._v3_actions_roi=np.zeros(25,dtype=np.int64)
        self._v3_actions_no_roi=np.zeros(25,dtype=np.int64)
        return obs,meta

    def step(self,action):
        roi_fraction=float((self.importance>0).mean())
        has_roi=bool(roi_fraction>0)
        obs,reward,done,truncated,info=super().step(action)
        self._v3_roi_fraction_sum+=roi_fraction
        if has_roi:
            self._v3_roi_frames+=1
            self._v3_actions_roi[int(action)]+=1
        else:
            self._v3_no_roi_frames+=1
            self._v3_actions_no_roi[int(action)]+=1
        if done:
            info['episode']['roi_policy_diagnostics']={
                'roi_present_frames':self._v3_roi_frames,
                'roi_absent_frames':self._v3_no_roi_frames,
                'mean_roi_ctu_fraction':self._v3_roi_fraction_sum/self.completed,
                'roi_action_histogram':self._v3_actions_roi.tolist(),
                'no_roi_action_histogram':self._v3_actions_no_roi.tolist()}
            # Preserve the same final diagnostics in the per-episode evidence.
            (self.folder/'episode_summary.json').write_text(
                json.dumps(info['episode'],indent=2),encoding='utf-8')
        return obs,reward,done,truncated,info

    def _before_action(self):
        obs=super()._before_action()
        roi,bg=action_pair(self.prev_action)
        # V2 globals: index 5 duplicated 1-progress, index 19 encoded the
        # 25-action categorical ID as a scalar, incorrectly imposing ordinality.
        # V3 exposes the two physical QP offsets instead, same 25-dim shape.
        obs['global'][5]=roi/2.
        obs['global'][19]=bg/2.
        return obs
