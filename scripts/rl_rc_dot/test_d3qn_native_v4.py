"""CPU-only regression contracts for V4 terminal-aware replay ablation.

No codec, detector or BDD100K files required. Real VVC/VVdeC parity remains
the Kaggle preflight gate; these tests do not claim objective improvement.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.rl_rc_dot.d3qn_agent_v2 import DuelingQNetwork
from scripts.rl_rc_dot.d3qn_agent_v3 import StableMaskedD3QN as V3Agent
from scripts.rl_rc_dot.d3qn_agent_v4 import (
    CHECKPOINT_VERSION, StableMaskedD3QN, TerminalAwareReplayBuffer,
)
from scripts.rl_rc_dot.d3qn_native_runner_v4 import PROTOCOL, parser


def observation(i, *, neutral_only=False):
    g=np.zeros(25,dtype=np.float32)
    g[1]=i/256.
    m=np.zeros((5,10,15),dtype=np.float32)
    legal=np.ones(25,dtype=np.bool_)
    if neutral_only:
        legal[:]=False
        legal[12]=True
    return {'global':g,'maps':m,'legal':legal}


class V4TerminalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_buffer(self, *, terminal_ids, n_rows=75, quota=4, seed=19):
        buf=TerminalAwareReplayBuffer(128,seed,terminal_quota=quota)
        for i in range(n_rows):
            buf.append(observation(i),12,float(i),observation(i+1),i in terminal_ids)
        return buf

    def test_four_terminal_without_replacement(self):
        terminal_ids={4,19,37,55,74}
        buf=self.make_buffer(terminal_ids=terminal_ids)
        obs,actions,rewards,nxt,dones=buf.sample(16,'cpu')
        self.assertEqual(buf.last_sample_terminal_count,4)
        self.assertEqual(int(dones.sum()),4)
        self.assertEqual(len(torch.unique(obs['global'][:,1])),16)
        self.assertTrue((actions==12).all())
        self.assertTrue(torch.all(torch.isfinite(rewards)))
        sampled=set((obs['global'][:,1]*256).round().long().tolist())
        self.assertEqual(len(sampled & terminal_ids),4)

    def test_low_terminal_availability_and_no_terminal(self):
        for terminal_ids,expected in [(set(),0),({5,61},2),({5,8,61,70,74},4)]:
            buf=self.make_buffer(terminal_ids=terminal_ids,seed=43)
            obs,_,_,_,dones=buf.sample(16,'cpu')
            self.assertEqual(int(dones.sum()),expected)
            self.assertEqual(len(torch.unique(obs['global'][:,1])),16)

    def test_replay_rng_resume_and_sampling_reproducibility(self):
        base=self.make_buffer(terminal_ids={5,15,25,35,45,55,65})
        _=base.sample(16,'cpu')
        state=base.state_dict()
        duplicate=TerminalAwareReplayBuffer(128,77,terminal_quota=4)
        duplicate.load_state_dict(state)
        first=base.sample(16,'cpu')
        second=duplicate.sample(16,'cpu')
        for a,b in zip(first,second):
            self.assertTrue(torch.equal(a,b))
        self.assertEqual(base.last_sample_terminal_count,duplicate.last_sample_terminal_count)

    def test_nstep_terminal_flush_and_reward_unchanged(self):
        model=StableMaskedD3QN(seed=7,n_step=3,terminal_quota=4,
                               reward_scale=1.,batch_size=4,warmup=64)
        for i in range(5):
            model.observe(observation(i),12,float(i+1),observation(i+1),i==4)
        rows=list(model.replay.rows)
        self.assertEqual([round(r[2],4) for r in rows],[6.,9.,12.,9.,5.])
        self.assertEqual([r[4] for r in rows],[False,False,True,True,True])
        self.assertEqual(len(model.pending),0)
        model.replay.sample(4,'cpu')
        self.assertEqual(model.replay.last_sample_terminal_count,3)

    def test_optimizer_smoke_with_terminal_batch_and_legal_mask(self):
        model=StableMaskedD3QN(seed=23,n_step=3,terminal_quota=3,
                               batch_size=4,warmup=4,train_every=1,
                               reward_scale=1.)
        output=None
        for i in range(9):
            obs=observation(i,neutral_only=True)
            nex=observation(i+1,neutral_only=True)
            self.assertEqual(model.predict(obs,epsilon=1.),12)
            output=model.observe(obs,12,.003,nex,i==8)
        self.assertEqual(len(model.replay),9)
        self.assertIsNotNone(output)
        self.assertEqual(output['sampled_terminal'],3)
        self.assertTrue(np.isfinite(output['loss']))
        self.assertTrue(np.isfinite(output['mean_abs_td']))
        self.assertTrue(abs(output['q_gap_valid'])<1e-5)

    def test_checkpoint_version_strict_identity_and_resume(self):
        agent=StableMaskedD3QN(seed=13,n_step=3,terminal_quota=4,
                               batch_size=4,warmup=64,reward_scale=1.)
        for i in range(7):
            agent.observe(observation(i),12,.01,observation(i+1),i==6)
        agent.steps=256
        identity={'protocol':PROTOCOL,'terminal_quota':4,'unit':True}
        with tempfile.TemporaryDirectory() as td:
            ck=Path(td)/'v4.pt'
            agent.save(ck,identity=identity,next_episode_seed=456)
            payload=torch.load(ck,map_location='cpu',weights_only=True)
            self.assertEqual(payload['version'],CHECKPOINT_VERSION)
            self.assertEqual(payload['config']['terminal_quota'],4)
            restored,seed=StableMaskedD3QN.load(ck,identity=identity,device='cpu')
            self.assertEqual(seed,456)
            self.assertEqual(restored.steps,agent.steps)
            self.assertEqual(restored.terminal_quota,4)
            self.assertEqual(restored.replay.state_dict()['capacity'],10000)
            a=agent.replay.sample(4,'cpu')
            b=restored.replay.sample(4,'cpu')
            self.assertTrue(torch.equal(a[0]['global'],b[0]['global']))
            with self.assertRaises(ValueError):
                StableMaskedD3QN.load(ck,identity={'protocol':PROTOCOL,'terminal_quota':2,'unit':True})
            with self.assertRaises(ValueError):
                V3Agent.load(ck,identity=identity)
            v3=V3Agent(seed=15,n_step=3,batch_size=4,warmup=64,reward_scale=1.)
            for i in range(4):
                v3.observe(observation(i),12,.01,observation(i+1),i==3)
            v3.steps=256
            old=Path(td)/'v3.pt'
            v3.save(old,identity=identity,next_episode_seed=567)
            with self.assertRaises(ValueError):
                StableMaskedD3QN.load(old,identity=identity)

    def test_cli_protocol_and_fixed_train_parameters(self):
        args=parser().parse_args([
            '--phase','train','--input-root','.','--encoder','./enc',
            '--decoder','./dec','--yolo-repo','./yolo','--weights','./weights',
            '--output','./out'
        ])
        self.assertEqual(PROTOCOL,'native-masked-d3qn-qp/v4-terminal-aware')
        self.assertEqual(args.terminal_quota,4)
        self.assertEqual(args.n_step,3)
        self.assertAlmostEqual(args.learning_rate,5e-5)
        self.assertEqual(args.train_steps,5120)
        self.assertNotIn('test',parser()._option_string_actions['--phase'].choices)
        with self.assertRaises(ValueError):
            StableMaskedD3QN(batch_size=4,terminal_quota=5)
        with self.assertRaises(ValueError):
            StableMaskedD3QN(terminal_quota=0)


if __name__=='__main__':
    unittest.main()
