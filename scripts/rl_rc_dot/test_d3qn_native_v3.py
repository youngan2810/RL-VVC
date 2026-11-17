"""CPU-only V3 regression tests; real VVenC/VVdeC preflight remains separate."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.rl_rc_dot.d3qn_agent_v3 import StableMaskedD3QN, exploration_epsilon
from scripts.rl_rc_dot.d3qn_native_env import D3QNNativeEnv
from scripts.rl_rc_dot.d3qn_native_env_v3 import D3QNNativeEnvV3
from scripts.rl_rc_dot.d3qn_method import NEUTRAL_ACTION, action_pair
from scripts.rl_rc_dot.d3qn_native_runner_v3 import PROTOCOL,TRAIN_RATES


def synthetic_obs(step):
    g=np.zeros((25,),dtype=np.float32)
    g[1]=step/256.
    m=np.zeros((5,10,15),dtype=np.float32)
    legal=np.ones((25,),dtype=bool)
    return {'global':g,'maps':m,'legal':legal}


class StableD3QNContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_slow_exploration_and_unique_v3_protocol(self):
        self.assertEqual(PROTOCOL,'native-masked-d3qn-qp/v3')
        self.assertEqual(TRAIN_RATES,(50,80,120,160,200))
        self.assertAlmostEqual(exploration_epsilon(0),1.0)
        self.assertAlmostEqual(exploration_epsilon(5120),0.6)
        self.assertAlmostEqual(exploration_epsilon(10240),0.2)
        self.assertAlmostEqual(exploration_epsilon(20000),0.2)
        with self.assertRaises(ValueError):exploration_epsilon(-1)

    def test_three_step_returns_and_terminal_flush_are_exact(self):
        agent=StableMaskedD3QN(seed=7,device='cpu',n_step=3,
                                reward_scale=1.0,batch_size=4,warmup=64)
        for i in range(5):
            agent.observe(synthetic_obs(i),NEUTRAL_ACTION,float(i+1),
                          synthetic_obs(i+1),i==4)
        self.assertEqual(agent.steps,5)
        self.assertEqual(len(agent.replay),5)
        self.assertEqual(len(agent.pending),0)
        rows=list(agent.replay.rows)
        self.assertEqual([round(r[2],4) for r in rows],[6.,9.,12.,9.,5.])
        self.assertEqual([bool(r[4]) for r in rows],[False,False,True,True,True])
        self.assertEqual([round(float(r[3][0][1])*256) for r in rows],[3,4,5,5,5])

    def test_previous_action_has_two_physical_qp_features(self):
        base=synthetic_obs(0)
        for action in (0,9,12,19,24):
            env=object.__new__(D3QNNativeEnvV3)
            env.prev_action=action
            with patch.object(D3QNNativeEnv,'_before_action',return_value={
                'global':base['global'].copy(),'maps':base['maps'].copy(),
                'legal':base['legal'].copy()}):
                obs=env._before_action()
            roi,bg=action_pair(action)
            self.assertAlmostEqual(float(obs['global'][5]),roi/2.)
            self.assertAlmostEqual(float(obs['global'][19]),bg/2.)

    def test_training_gradients_polyak_and_checkpoint_resume(self):
        model=StableMaskedD3QN(seed=23,n_step=3,lr=5e-5,
                               batch_size=4,warmup=4,train_every=1)
        target_before=[p.detach().clone() for p in model.target.parameters()]
        changes=[]
        for i in range(10):
            obs=synthetic_obs(i)
            action=model.predict(obs,epsilon=1.0)
            out=model.observe(obs,action,0.003,synthetic_obs(i+1),i==9)
            if out:changes.append(out)
        self.assertTrue(changes)
        self.assertTrue(all(np.isfinite(x['loss']) and np.isfinite(x['q_gap_valid'])
                            and np.isfinite(x['grad_norm']) for x in changes))
        self.assertTrue(any(not torch.equal(old,p)
                            for old,p in zip(target_before,model.target.parameters())))
        self.assertEqual(len(model.pending),0)
        # Force the documented episode-boundary save constraint for a short CPU test.
        model.steps=256
        identity={'protocol':PROTOCOL,'unit':True}
        with tempfile.TemporaryDirectory() as td:
            ck=Path(td)/'v3_checkpoint.pt'
            model.save(ck,identity=identity,next_episode_seed=112233)
            restored,next_seed=StableMaskedD3QN.load(ck,identity=identity,device='cpu')
            self.assertEqual(next_seed,112233)
            self.assertEqual(restored.steps,model.steps)
            self.assertEqual(restored.updates,model.updates)
            self.assertEqual(len(restored.replay),len(model.replay))
            self.assertEqual(restored.predict(synthetic_obs(20)),
                             model.predict(synthetic_obs(20)))
            self.assertEqual(restored.n_step,3)
            self.assertAlmostEqual(restored.target_tau,.005)
            with self.assertRaises(ValueError):
                StableMaskedD3QN.load(ck,identity={'protocol':'wrong'},device='cpu')
            with self.assertRaises(ValueError):
                from scripts.rl_rc_dot.d3qn_agent_v2 import MaskedD3QN
                MaskedD3QN.load(ck,identity=identity,device='cpu')

    def test_finite_horizon_gamma_invariant(self):
        with self.assertRaises(ValueError):
            StableMaskedD3QN(gamma=.99)
        with self.assertRaises(ValueError):
            StableMaskedD3QN(n_step=0)
        model=StableMaskedD3QN(n_step=3)
        self.assertEqual(model.gamma,1.0)


if __name__=='__main__':
    unittest.main()
