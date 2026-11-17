"""Unit contracts for the independent D3QN MDP and action-masked Double-Q model.

No VVenC binary needed; separate native/decoder validation remains mandatory.
Run: python -m unittest scripts.rl_rc_dot.test_d3qn_native_v2 -v
"""
import unittest
import numpy as np
import torch
from scripts.rl_rc_dot.d3qn_method import (
    semantic_importance,action_pair,qp_action_map,legal_actions,observation,
    reward_step,clip_objective,N_ACTIONS,NEUTRAL_ACTION)
from scripts.rl_rc_dot.d3qn_agent_v2 import DuelingQNetwork,MaskedD3QN


def dummy_native_event(step=0):
    maps=np.zeros((4,600),dtype=float)
    maps[:3]=np.linspace(1,40,600)
    maps[3]=.07
    g=np.zeros(16,dtype=float)
    g[0]=31;g[3]=31;g[4]=30;g[14]=1;g[15]=1500
    return {'global':g.tolist(),'maps':maps.tolist(),'step':step,
            'poc':step,'lookahead_last_poc':min(255,step+10)}


class IndependentActionTests(unittest.TestCase):
    def test_25_exact_integer_actions_and_neutral(self):
        empty=semantic_importance(np.empty((0,5)))
        self.assertEqual(empty.shape,(10,15))
        mask=legal_actions(empty)
        self.assertEqual(int(mask.sum()),5)
        self.assertTrue(mask[NEUTRAL_ACTION])
        self.assertTrue((qp_action_map(NEUTRAL_ACTION,empty)==0).all())
        self.assertEqual(action_pair(0),(-2,-2))
        self.assertEqual(action_pair(24),(2,2))
        for a in range(N_ACTIONS):
            m=qp_action_map(a,empty)
            self.assertEqual(m.shape,(20,30))
            self.assertTrue((m==m.astype(int)).all())
            for y in range(0,20,2):
                for x in range(0,30,2):
                    self.assertEqual(np.unique(m[y:y+2,x:x+2]).size,1)

    def test_roi_semantics_and_distinct_actions(self):
        boxes=np.array([[45.,55.,72.,89.,.9]])
        semantic=semantic_importance(boxes)
        self.assertGreater(float(semantic.sum()),0)
        self.assertLess(float(semantic.sum()),10)
        self.assertEqual(int(legal_actions(semantic).sum()),25)
        roi_bg=qp_action_map(9,semantic) # roi=-1, bg=+2
        mask=np.repeat(np.repeat(semantic>0,2,axis=0),2,axis=1)
        np.testing.assert_array_equal(roi_bg,np.where(mask,-1,2))
        with self.assertRaises(ValueError):semantic_importance([[0,0,600,10,.5]])
        with self.assertRaises(ValueError):qp_action_map(25,semantic)

    def test_state_is_not_ppo_state(self):
        boxes=np.array([[45.,55.,72.,89.,.9]])
        obs,sem=observation(dummy_native_event(),boxes,
                            frames=256,target_bps=120000,completed=0,cumulative_bits=0)
        self.assertEqual(set(obs),{'global','maps','legal'})
        self.assertEqual(obs['global'].shape,(25,))
        self.assertEqual(obs['maps'].shape,(5,10,15))
        self.assertEqual(obs['legal'].shape,(25,))
        np.testing.assert_array_equal(obs['maps'][4],sem)
        self.assertTrue(obs['legal'][NEUTRAL_ACTION])
        with self.assertRaises(ValueError):
            observation(dummy_native_event(step=3),boxes,frames=256,target_bps=120000,completed=0)

    def test_telescope_matches_final_clip_objective(self):
        n=256;target=120000.;rng=np.random.default_rng(12)
        scores=rng.random(n);actual=rng.integers(1500,5200,size=n)
        previous=0;rewards=[]
        for idx,(f,bits) in enumerate(zip(scores,actual)):
            value,_=reward_step(float(f),int(bits),previous,idx,frames=n,target_bps=target)
            rewards.append(value);previous+=int(bits)
        self.assertAlmostEqual(sum(rewards),clip_objective(scores,previous,frames=n,target_bps=target),places=11)


class MaskedD3QNTests(unittest.TestCase):
    def test_replay_and_double_q_update(self):
        torch.set_num_threads(1)
        agent=MaskedD3QN(seed=3,batch_size=4,warmup=4,train_every=1,target_sync_updates=2)
        obs,_=observation(dummy_native_event(),[[45,55,72,89,.9]],frames=256,target_bps=120000,completed=0)
        legal=np.flatnonzero(obs['legal'])
        self.assertEqual(len(legal),25)
        with torch.no_grad():
            q=agent.online({'global':torch.zeros(2,25),'maps':torch.zeros(2,5,10,15)})
        self.assertEqual(q.shape,(2,25))
        out=[]
        for i in range(7):
            action=agent.predict(obs,epsilon=1.)
            self.assertTrue(obs['legal'][action])
            info=agent.observe(obs,action,.005,obs,i==6)
            if info is not None:out.append(info)
        self.assertEqual(agent.updates,4)
        self.assertTrue(all(np.isfinite(x['loss']) and np.isfinite(x['grad_norm']) for x in out))
        illegal={**obs,'legal':np.zeros(25,dtype=bool)}
        with self.assertRaises(ValueError):agent.predict(illegal)
        empty=np.zeros((10,15),dtype=np.float32)
        obs_no,_=observation(dummy_native_event(),[],frames=256,target_bps=120000,completed=0)
        for _ in range(30):
            selected=agent.predict(obs_no,epsilon=1.)
            self.assertTrue(obs_no['legal'][selected])


if __name__=='__main__':
    unittest.main()
