"""Integer action semantics, exact PPO likelihood and native QG/RDO effects."""
import io,json,os,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.rl_rc_dot.qp_development import QP_BIN_ACTIONS,QP_BINS,map_qp_bins,neutral_action,map_action,ZeroTieCategoricals
from scripts.rl_rc_dot.action_audit import aggregate,rounded
from scripts.rl_rc_dot.environment import VVenCEnv
from scripts.rl_rc_dot.model import build,AuxPPO,AuxiliaryLabels
from scripts.rl_rc_dot.test_native_integrated import FixtureCatalog,FixtureDetector
from scripts.rl_rc_dot.codec import NativeEpisode
from scripts.rl_rc_dot.preflight import make_fixture,decode_audit
from scripts.rl_rc_dot.data import atomic_json
from scripts.rl_rc_dot.session import validate
from scripts.rl_rc_dot.artifacts import save_checkpoint,load_resume

class IntegerAction(unittest.TestCase):
    def test_zero_tie_break_does_not_introduce_a_logit_dead_zone(self):
        d=ZeroTieCategoricals([5]*36)
        logits=torch.zeros(1,180);d.proba_distribution(logits)
        torch.testing.assert_close(d.mode(),torch.full((1,36),2))
        torch.testing.assert_close(d.distribution[0].probs,torch.full((1,5),.2))
        logits[0,3]=1e-4;d.proba_distribution(logits)
        self.assertEqual(int(d.mode()[0,0]),3)
    def test_short_notebook_keeps_training_settings_and_new_profile(self):
        import ast,contextlib
        p=Path(__file__).resolve().parents[2]/'notebooks/VVC_RL_QPDev1_ActionFix_Short_Kaggle.ipynb'
        n=json.loads(p.read_text(encoding='utf-8'));codes=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        for s in codes:ast.parse(s)
        config=codes[0].split('\nimport torch')[0]
        with contextlib.redirect_stdout(io.StringIO()):ns={};exec(config,ns)
        self.assertEqual(ns['AGENT_PROFILE'],'qpdev1-qbins');self.assertEqual(ns['MODE'],'new')
        self.assertEqual(ns['ADDITIONAL_TIMESTEPS'],5120)
        self.assertEqual(ns['PPO_PROFILE'],{'learning_rate':3e-5,'n_epochs':3,'target_kl':.05,'aux_loss_coef':.1,'gamma':1})
        self.assertEqual((ns['WORKERS'],ns['ROLLOUT_STEPS'],ns['TRAIN_FRAME_LIMIT']),(2,256,256))
        self.assertEqual((ns['VAL_VIDEOS'],ns['VAL_RATES'],ns['VAL_FRAMES']),(4,'50,120,200',256))
        self.assertIn("'--agent-profile',AGENT_PROFILE",'\n'.join(codes))
    def test_uniform_choices_survive_every_legal_qg(self):
        for index,value in enumerate(QP_BINS):
            a=neutral_action(QP_BIN_ACTIONS);a[0]=index;out=map_qp_bins(a)
            np.testing.assert_array_equal(out,np.full((20,30),value))
            for shape in ((0,0,32,32),(16,0,16,32),(0,16,32,16),(4,4,8,8)):
                self.assertEqual(rounded(aggregate(out,dict(zip(('x','y','w','h'),shape)))),value)
        for bad in (np.full(36,.1),np.full(36,5),np.full(36,np.nan)):
            with self.assertRaises(ValueError):map_qp_bins(bad)

    def test_mixed_old_actions_cancel_but_new_tiles_do_not_cross_qgs(self):
        old=np.zeros((20,30));old[0,:2]=[2,-2]
        self.assertEqual(aggregate(old,{'x':0,'y':0,'w':32,'h':16}),0)
        rng=np.random.default_rng(7)
        for _ in range(20):
            d=map_qp_bins(rng.integers(0,5,36))
            self.assertLessEqual(abs(d).max(),6)
            self.assertTrue(np.array_equal(d,d.astype(int)))
            for y in range(0,20,2):
                for x in range(0,30,2):self.assertTrue(np.all(d[y:y+2,x:x+2]==d[y,x]))

    def test_old_gaussian_mapping_retains_recorded_dead_zone(self):
        a=np.zeros(36);a[0]=np.arctanh(.0803541839/2)
        d=map_action(a)
        self.assertLess(abs(d).max(),.5)
        self.assertEqual(rounded(aggregate(d,{'x':0,'y':0,'w':32,'h':32})),0)

@unittest.skipUnless(os.environ.get('RCDOT_TEST_ENCODER'),'Requires repaired native encoder')
class IntegerNative(unittest.TestCase):
    def test_sampling_log_probability_entropy_ratio_and_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);enc=os.environ['RCDOT_TEST_ENCODER']
            factory=lambda:VVenCEnv(FixtureCatalog(),FixtureDetector(),enc,root/'episodes',frame_limit=32,
                                   threads=2,agent_profile='qpdev1-qbins')
            env=DummyVecEnv([factory]);model=build(env,7,n_steps=32,batch_size=16,n_epochs=2,gamma=1,learning_rate=3e-5)
            try:
                obs=env.reset();tensor=model.policy.obs_to_tensor(obs)[0]
                with torch.no_grad():
                    distribution=model.policy.get_distribution(tensor)
                    a=distribution.sample();lp=distribution.log_prob(a)
                    manual=torch.stack([d.log_prob(a[:,i]) for i,d in enumerate(distribution.distribution)],dim=1).sum(dim=1)
                    torch.testing.assert_close(lp,manual)
                    manual_entropy=torch.stack([d.entropy() for d in distribution.distribution],dim=1).sum(dim=1)
                    torch.testing.assert_close(distribution.entropy(),manual_entropy)
                    _,new_lp,_=model.policy.evaluate_actions(tensor,a)
                    torch.testing.assert_close(torch.exp(new_lp-lp),torch.ones_like(lp))
                    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],np.full((1,36),2))
                    # A categorical mode can request +1, without forcing nonzero
                    # in the actual untrained policy or changing its logits permanently.
                    bias=model.policy.action_net.frame.bias.clone()
                    model.policy.action_net.frame.bias[3]=.1
                    probe=model.predict(obs,deterministic=True)[0][0]
                    np.testing.assert_array_equal(map_qp_bins(probe),np.ones((20,30)))
                    model.policy.action_net.frame.bias.copy_(bias)
                model.learn(64)
                self.assertEqual(model.completed_rollouts,2)
                self.assertTrue(np.isfinite(model.last_update_metrics['action_entropy_per_decision']))
                self.assertNotIn('policy_std',model.last_update_metrics)
                # Raw categories are stored; the wire map is checked separately.
                self.assertTrue(np.all(model.rollout_buffer.actions==np.floor(model.rollout_buffer.actions)))
                contract={'profile':'qpdev1-qbins','fixture':True};atomic_json(root/'method.json',contract)
                save_checkpoint(model,env,root,contract)
            finally:model.get_env().close()
            env2=DummyVecEnv([factory]);resumed,_=load_resume(root,contract,env2)
            try:
                resumed.learn(32,reset_num_timesteps=False)
                self.assertEqual(resumed.completed_rollouts,3)
                spec={'input_root':tmp,'ffmpeg':'unused','ffprobe':'unused','seed':7,'gpu_ids':[],
                    'yolo_repo':'fixture','weights':'fixture','output':tmp,'validation_videos':1,
                    'validation_rates':[120],'validation_frames':32,'encoder':enc,'encoder_threads':2,
                    'agent_profile':'qpdev1-qbins'}
                if os.environ.get('RCDOT_TEST_DECODER'):spec['decoder']=os.environ['RCDOT_TEST_DECODER']
                v=validate(resumed,spec,detector_factory=lambda *a:FixtureDetector(),
                    catalog_factory=lambda *a,**kw:FixtureCatalog())
                self.assertIn('random_score',v);self.assertIn('random',v['points'][0])
                self.assertGreater(v['points'][0]['random']['native_action_audit']['candidate_qgs'],0)
                self.assertEqual(v['points'][0]['zero']['requested_abs_delta'],0)
                if spec.get('decoder'):
                    self.assertTrue(all(p['decoded_reconstruction_verified'] for p in v['points'][0].values() if isinstance(p,dict)))
            finally:resumed.get_env().close()

if __name__=='__main__':unittest.main()
