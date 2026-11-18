"""CPU-only V5 coding-cycle ablation regression and compatibility checks.

These do NOT claim native codec parity or improved Val4 objective. Those remain
independent Kaggle preflight and experimental gates.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import numpy as np
import torch

from scripts.rl_rc_dot.codec import command
from scripts.rl_rc_dot.d3qn_method import NEUTRAL_ACTION
from scripts.rl_rc_dot.d3qn_state_v5 import (
    CODING_CYCLE, GLOBAL_DIM_V5, cycle_features,
    extend_observation, terminal_observation,
)
from scripts.rl_rc_dot.d3qn_agent_v4 import StableMaskedD3QN as V4Agent
from scripts.rl_rc_dot.d3qn_agent_v5 import (
    StableMaskedD3QN, DuelingQNetworkV5, pack_v5,
    CHECKPOINT_VERSION, STATE_SCHEMA,
)
from scripts.rl_rc_dot.d3qn_native_env_v3 import D3QNNativeEnvV3
from scripts.rl_rc_dot.d3qn_native_env_v5 import D3QNNativeEnvV5
from scripts.rl_rc_dot.d3qn_native_runner_v5 import PROTOCOL, parser


def v4_state(step, neutral_only=False):
    g=np.zeros(25,dtype=np.float32)
    g[1]=step/256.
    m=np.zeros((5,10,15),dtype=np.float32)
    legal=np.ones(25,dtype=np.bool_)
    if neutral_only:
        legal[:]=False
        legal[NEUTRAL_ACTION]=True
    return {'global':g,'maps':m,'legal':legal}


def v5_state(step, neutral_only=False):
    return extend_observation(v4_state(step,neutral_only),step)


class V5Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_coding_phase_values_and_encoder_cycle_contract(self):
        self.assertEqual(CODING_CYCLE,32)
        for step in [0,1,31,32,33,63,64,95,96,128,224,255]:
            a=cycle_features(step)
            self.assertEqual(a.shape,(3,))
            self.assertEqual(a.dtype,np.float32)
            self.assertAlmostEqual(float(a[0]),float(np.sin(2*np.pi*(step%32)/32)),places=5)
            self.assertAlmostEqual(float(a[1]),float(np.cos(2*np.pi*(step%32)/32)),places=5)
            self.assertEqual(float(a[2]),float(step%32==0))
        for step in (0,32,64,96,128,160,192,224):
            self.assertEqual(float(cycle_features(step)[2]),1.)
        # Cycle is based on coding-step, not POC or a presumed I/CRA type.
        argv=command(Path('encoder'),Path('source'),Path('out'),256,120)
        self.assertEqual(argv[argv.index('--intraperiod')+1],'32')
        self.assertIn('GOPSize=32',argv[argv.index('--additional')+1])
        with self.assertRaises(ValueError): cycle_features(-1)
        with self.assertRaises(ValueError): cycle_features(1.5)

    def test_v5_preserves_25_original_features_and_masks(self):
        for step in (0,31,32,127,255):
            initial=v4_state(step)
            modified=extend_observation(initial,step)
            self.assertEqual(modified['global'].shape,(GLOBAL_DIM_V5,))
            np.testing.assert_array_equal(modified['global'][:25],initial['global'])
            np.testing.assert_array_equal(modified['maps'],initial['maps'])
            np.testing.assert_array_equal(modified['legal'],initial['legal'])
            self.assertTrue(np.isfinite(modified['global']).all())
            self.assertEqual(pack_v5(modified)[0].shape,(28,))
            with self.assertRaises(ValueError):pack_v5(initial)
        with self.assertRaises(ValueError):extend_observation(v4_state(8),9)
        with self.assertRaises(ValueError):extend_observation(v5_state(8),8)

    def test_native_environment_phase_and_terminal_are_consistent(self):
        env=object.__new__(D3QNNativeEnvV5)
        for step in (0,1,32,255):
            env.completed=step
            env.native=SimpleNamespace(current={'step':step})
            with patch.object(D3QNNativeEnvV3,'_before_action',return_value=v4_state(step)):
                obs=env._before_action()
            self.assertEqual(obs['global'].shape,(28,))
            self.assertEqual(float(obs['global'][27]),float(step%32==0))
        env.completed=32
        env.native=SimpleNamespace(current={'step':31})
        with patch.object(D3QNNativeEnvV3,'_before_action',return_value=v4_state(32)):
            with self.assertRaises(ValueError):env._before_action()
        env.native=SimpleNamespace(current={'step':255})
        env.completed=255
        with patch.object(D3QNNativeEnvV3,'step',return_value=(
                v4_state(0),0.075,True,False,{'episode':{'objective':0.075}})):
            nxt,r,done,truncated,info=env.step(NEUTRAL_ACTION)
        self.assertTrue(done)
        self.assertEqual(nxt['global'].shape,(28,))
        self.assertEqual(env.current_obs['global'].shape,(28,))
        self.assertAlmostEqual(r,0.075)
        self.assertEqual(info['episode']['objective'],0.075)
        self.assertEqual(terminal_observation(v4_state(0))['global'].shape,(28,))

    def test_network_changes_only_first_fusion_input_width(self):
        net=DuelingQNetworkV5()
        self.assertEqual(net.fusion[0].in_features,32*25+28)
        self.assertEqual(net.fusion[0].out_features,128)
        a=v5_state(0)
        sample={'global':torch.tensor(a['global'][None]),
                'maps':torch.tensor(a['maps'][None]),
                'legal':torch.tensor(a['legal'][None])}
        with torch.no_grad():q=net(sample)
        self.assertEqual(tuple(q.shape),(1,25))
        mask=sample['legal'].clone()
        mask[:]=False
        mask[:,12]=True
        sample['legal']=mask
        with torch.no_grad():q=net(sample)
        self.assertEqual(tuple(q.shape),(1,25))
        with self.assertRaises(ValueError):net({'global':torch.zeros(1,25),'maps':sample['maps'],'legal':mask})

    def test_nstep_rewards_terminal_quota_and_legal_actions(self):
        model=StableMaskedD3QN(seed=7,n_step=3,terminal_quota=3,
                               reward_scale=1.,batch_size=4,warmup=64)
        for i in range(5):
            s=v5_state(i,neutral_only=True)
            e=v5_state(i+1,neutral_only=True)
            self.assertEqual(model.predict(s,epsilon=1.),NEUTRAL_ACTION)
            self.assertIsNone(model.observe(s,12,float(i+1),e,i==4))
        rows=list(model.replay.rows)
        self.assertEqual([round(x[2],6) for x in rows],[6.,9.,12.,9.,5.])
        self.assertEqual([bool(x[4]) for x in rows],[False,False,True,True,True])
        self.assertEqual(len(model.pending),0)
        self.assertEqual(rows[0][0][0].shape,(28,))
        model.replay.sample(4,'cpu')
        self.assertEqual(model.replay.last_sample_terminal_count,3)

    def test_training_smoke_and_gradients(self):
        model=StableMaskedD3QN(seed=77,n_step=3,terminal_quota=3,
                               reward_scale=1.,batch_size=4,warmup=4,train_every=1)
        outputs=[]
        for i in range(9):
            u=model.observe(v5_state(i,True),12,.003,v5_state(i+1,True),i==8)
            if u:outputs.append(u)
        self.assertTrue(outputs)
        self.assertEqual(len(model.replay),9)
        self.assertEqual(outputs[-1]['sampled_terminal'],3)
        self.assertTrue(np.isfinite(outputs[-1]['loss']))
        self.assertLess(abs(outputs[-1]['q_gap_valid']),1e-5)

    def test_v5_checkpoint_version_resume_and_v4_rejection(self):
        agent=StableMaskedD3QN(seed=13,n_step=3,terminal_quota=3,
                               batch_size=4,warmup=64,reward_scale=1.)
        for i in range(7):
            agent.observe(v5_state(i),12,.01,v5_state(i+1),i==6)
        agent.steps=256
        identity={'protocol':PROTOCOL,'learning':{'terminal_quota':3},'unit':True}
        with tempfile.TemporaryDirectory() as td:
            ck=Path(td)/'v5.pt'
            agent.save(ck,identity=identity,next_episode_seed=456)
            payload=torch.load(ck,map_location='cpu',weights_only=True)
            self.assertEqual(payload['version'],CHECKPOINT_VERSION)
            self.assertEqual(payload['state_schema'],STATE_SCHEMA)
            self.assertEqual(payload['config']['global_dim'],28)
            restored,seed=StableMaskedD3QN.load(ck,identity=identity)
            self.assertEqual(seed,456)
            self.assertEqual(restored.steps,agent.steps)
            self.assertEqual(restored.terminal_quota,3)
            a=agent.replay.sample(4,'cpu')
            b=restored.replay.sample(4,'cpu')
            self.assertTrue(torch.equal(a[0]['global'],b[0]['global']))
            self.assertEqual(restored.predict(v5_state(3),epsilon=0.),
                             agent.predict(v5_state(3),epsilon=0.))
            with self.assertRaises(ValueError): V4Agent.load(ck,identity=identity)
            with self.assertRaises(ValueError):
                StableMaskedD3QN.load(ck,identity={'protocol':'wrong'})
            with self.assertRaises(ValueError):
                StableMaskedD3QN.load(ck,identity={'protocol':PROTOCOL,'learning':{'terminal_quota':4},'unit':True})
            with self.assertRaises(ValueError):
                restored.replay.append(v4_state(1),12,.01,v5_state(2),False)

    def test_cli_and_protocol_locked_to_train16_val4(self):
        args=parser().parse_args(['--phase','train','--input-root','.',
            '--encoder','./enc','--decoder','./dec','--yolo-repo','./yolo',
            '--weights','./weights','--output','./out'])
        self.assertEqual(args.train_steps,5120)
        self.assertEqual(args.n_step,3)
        self.assertEqual(args.terminal_quota,4)
        self.assertAlmostEqual(args.learning_rate,5e-5)
        self.assertEqual(PROTOCOL,'native-masked-d3qn-qp/v5-coding-cycle')
        self.assertNotIn('test',parser()._option_string_actions['--phase'].choices)
        runner=Path(__file__).with_name('d3qn_native_runner_v5.py').read_text()
        self.assertIn('lr=args.learning_rate,terminal_quota=args.terminal_quota)',runner)
        self.assertIn("'coding_cycle_steps':32",runner)
        self.assertIn("'state_schema':'v5-coding-cycle-global28'",runner)
        self.assertIn('source_fingerprint()',runner)


if __name__=='__main__':
    unittest.main()
