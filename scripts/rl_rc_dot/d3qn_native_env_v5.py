"""V5 native VVC environment: V3 semantics + three causal cycle features.

No changes to native encoder interface, QP action maps, detector, reward or
terminal definition. Only the agent-visible global feature length changes.
"""
from .d3qn_native_env_v3 import D3QNNativeEnvV3
from .d3qn_state_v5 import extend_observation, terminal_observation

class D3QNNativeEnvV5(D3QNNativeEnvV3):
    def _before_action(self):
        obs=super()._before_action()
        if int(self.native.current["step"]) != self.completed:
            raise ValueError("V5 coding-step mismatch at decision boundary")
        return extend_observation(obs,self.completed)

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        if done:
            obs=terminal_observation(obs)
            self.current_obs=obs
        return obs,reward,done,truncated,info
