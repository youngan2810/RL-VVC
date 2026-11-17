"""Controls and learning audit must not alter PPO gradients or hide weak modes."""
import unittest,torch,numpy as np
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.rl_rc_dot.qp_development import QP_BIN_ACTIONS,DEV_OBSERVATIONS,map_qp_bins,ZeroTieCategoricals
from scripts.rl_rc_dot.learning_diagnostics import categorical_metrics,gradient_metrics
from scripts.rl_rc_dot.session import fixed_minus_two
from scripts.rl_rc_dot.model import build
from scripts.rl_rc_dot.test_integrated_model import Fixture

class IntegerFixture(Fixture):
    action_space=QP_BIN_ACTIONS;observation_space=DEV_OBSERVATIONS
    def obs(self):return {'global':np.full(32,self.t/10,np.float32),'mb':np.full((6,20,30),self.t/10,np.float32)}
    def step(self,action):
        self.t+=1;mask=np.zeros((20,30),np.float32);mask[0,0]=1
        return self.obs(),float(self.t)/10,self.t==4,False,{'block_reward':mask,'block_mask':mask,'consumed_action':map_qp_bins(action).ravel().tolist()}

class LearningDiagnostics(unittest.TestCase):
    def test_screen_does_not_promote_policy_equal_to_fixed_control(self):
        import tempfile,json
        from pathlib import Path
        from scripts.rl_rc_dot.development_report import summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'validations').mkdir()
            a={'f1':.5,'mean_frame_f1':.5,'objective_score':.2,'actual_kbps':100,'mean_qp':30,'requested_abs_delta':0,'requested_spatial_std':0,'requested_nonzero_fraction':0,'psnr_y':35}
            b={**a,'f1':.6,'mean_frame_f1':.6,'objective_score':.4,'requested_abs_delta':2,'requested_nonzero_fraction':1}
            v={'score':.4,'anchor_score':.2,'completed_rollouts':10,'num_timesteps':5120,'random_score':.3,'fixed_m2_score':.4,'macro_delta_f1':.1,
               'points':[{'video_id':'fixture','rate_kbps':100,'anchor':a,'agent':b}]}
            (root/'validations/initial.json').write_text(json.dumps({**v,'score':.2,'completed_rollouts':0,'num_timesteps':0}))
            path=root/'validations/trained.json';path.write_text(json.dumps(v))
            (root/'progress.json').write_text(json.dumps({'best':{'validation':'validations/trained.json'}}))
            r=summarize(root)
            self.assertFalse(r['advance_to_long_training'])
            self.assertFalse(r['screening_gates']['objective_above_fixed_minus_two'])
            self.assertTrue(r['screening_gates']['macro_rate_increase_at_most_1percent'])
            v.pop('fixed_m2_score');path.write_text(json.dumps(v));r=summarize(root)
            self.assertFalse(r['screening_gates']['fixed_minus_two_control_present'])

    def test_learning_notebook_changes_only_declared_trial_settings(self):
        import ast,json,io,contextlib
        from pathlib import Path
        n=json.loads((Path(__file__).resolve().parents[2]/'notebooks/VVC_RL_QBins_LearningCheck_Kaggle.ipynb').read_text(encoding='utf-8'))
        cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        for c in cells:ast.parse(c)
        with contextlib.redirect_stdout(io.StringIO()):ns={};exec(cells[0].split('\nimport torch')[0],ns)
        self.assertEqual(ns['PPO_LEARNING_RATE'],3e-5);self.assertEqual(ns['VAL_EVERY_UPDATES'],5)
        self.assertEqual(ns['ADDITIONAL_TIMESTEPS'],5120);self.assertEqual(ns['MODE'],'new')
        self.assertEqual(ns['PPO_AUX_LOSS_COEF'],.1);self.assertEqual(ns['PPO_EPOCHS'],3)
        self.assertEqual(ns['PPO_ACTION_INIT_GAIN'],.01)
        self.assertIn("'--ppo-action-init-gain'",'\n'.join(cells))
        self.assertEqual(ns['AGENT_PROFILE'],'qpdev1-qbins');self.assertEqual(ns['VAL_RATES'],'50,120,200')

    def test_small_head_init_unblocks_feature_gradient_without_changing_other_branches(self):
        models=[build(DummyVecEnv([IntegerFixture]),7,n_steps=8,batch_size=4,n_epochs=1,action_init_gain=g) for g in (0,.01)]
        try:
            old,new=(m.policy for m in models)
            self.assertEqual({k:v.shape for k,v in old.state_dict().items()},{k:v.shape for k,v in new.state_dict().items()})
            for k,v in old.state_dict().items():
                if not k.startswith('action_net.'):torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
            norms=[]
            obs=models[0].get_env().reset();a=torch.full((1,36),2);a[:,0]=0
            for p in (old,new):
                o=p.obs_to_tensor(obs)[0];_,lp,_=p.evaluate_actions(o,a)
                gs=torch.autograd.grad(-lp.mean(),list(p.pi_features_extractor.parameters()),allow_unused=True)
                norms.append(float(sum((g*g).sum() for g in gs if g is not None).sqrt()))
            self.assertEqual(norms[0],0);self.assertGreater(norms[1],0)
        finally:
            for m in models:m.get_env().close()

    def test_nonzero_gain_roundtrip_keeps_likelihood_and_optimizer(self):
        import tempfile
        from pathlib import Path
        from scripts.rl_rc_dot.model import AuxPPO
        m=build(DummyVecEnv([IntegerFixture]),7,n_steps=8,batch_size=4,n_epochs=1,learning_rate=3e-4,action_init_gain=.01)
        try:
            m.learn(8);obs=m.get_env().reset();expected=m.predict(obs,deterministic=True)[0]
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'model.zip';m.save(path);loaded=AuxPPO.load(path,device='cpu')
                self.assertEqual(loaded.policy.action_init_gain,.01)
                np.testing.assert_array_equal(loaded.predict(obs,deterministic=True)[0],expected)
                self.assertEqual(len(loaded.policy.optimizer.state),len(m.policy.optimizer.state))
        finally:m.get_env().close()

    def test_fixed_control_is_minus_two_not_minus_two_spatial_residual(self):
        a=fixed_minus_two(QP_BIN_ACTIONS)
        self.assertEqual(int(a[0]),0);np.testing.assert_array_equal(a[1:],np.full(35,2))
        np.testing.assert_array_equal(map_qp_bins(a),np.full((20,30),-2))

    def test_near_uniform_global_mode_is_not_reported_as_confident(self):
        d=ZeroTieCategoricals([5]*36);logits=torch.zeros(3,180);logits[:,0]=.008
        d.proba_distribution(logits);r=categorical_metrics(d)
        self.assertEqual(r['global_action_mode_fraction_0'],1)
        self.assertLess(r['global_action_max_probability'],.202)
        self.assertLess(r['global_action_probability_margin'],.002)
        self.assertAlmostEqual(r['spatial_action_entropy'],float(np.log(5)),places=6)

    def test_gradient_audit_preserves_backprop_exactly(self):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__();self.pi_features_extractor=nn.Linear(2,3);self.action_net=nn.Linear(3,2);self.reward_head=nn.Linear(3,1)
        torch.manual_seed(1);p=Policy();x=torch.randn(4,2)
        def losses():
            f=p.pi_features_extractor(x);return p.action_net(f).square().mean(),p.reward_head(f).square().mean()*.1
        a,b=losses();(a+b).backward();expected=[v.grad.clone() for v in p.parameters()];p.zero_grad()
        a,b=losses();result=gradient_metrics(a,b,p)
        self.assertTrue(all(v.grad is None for v in p.parameters()))
        (a+b).backward()
        for got,want in zip(p.parameters(),expected):torch.testing.assert_close(got.grad,want,rtol=0,atol=0)
        self.assertGreater(result['audit_policy_head_gradient_norm'],0)

    def test_real_ppo_update_reports_post_step_kl_and_branch_gradients(self):
        torch.set_num_threads(2);m=build(DummyVecEnv([IntegerFixture]),7,n_steps=8,batch_size=4,n_epochs=2,learning_rate=3e-4)
        try:
            m.learn(8);r=m.last_update_metrics
            for key in ('post_update_joint_kl','post_update_abs_logprob_change','global_action_max_probability','audit_policy_head_gradient_norm','audit_auxiliary_feature_gradient_norm'):
                self.assertTrue(np.isfinite(r[key]),key)
            self.assertGreater(r['post_update_joint_kl'],0)
            self.assertGreater(r['post_update_abs_logprob_change'],0)
            self.assertEqual(r['optimizer_steps'],4)
        finally:m.get_env().close()

if __name__=='__main__':unittest.main()
