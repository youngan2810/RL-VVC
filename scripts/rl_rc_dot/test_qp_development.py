"""Development objective, causal states, spatial policy and real-codec contracts."""
import sys,os,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.rl_rc_dot.qp_development import (map_action,state,clip_objective,reward_increment,
    DEV_OBSERVATIONS,DEV_ACTIONS,SpatialActionHead,retention_supervision)
from scripts.rl_rc_dot.model import build,AuxPPO,AuxiliaryLabels
from scripts.rl_rc_dot.environment import VVenCEnv
from scripts.rl_rc_dot.test_native_integrated import FixtureCatalog,FixtureDetector
from scripts.rl_rc_dot.session import validate,contract_for,parser,ppo_options
from scripts.rl_rc_dot.artifacts import save_checkpoint,load_resume,export_run,import_run,policy_digest
from scripts.rl_rc_dot.data import atomic_json
from functools import partial

class Development(unittest.TestCase):
    def test_split_action_is_bounded_and_centered(self):
        rng=np.random.default_rng(7)
        for _ in range(20):
            a=rng.normal(0,20,36);d=map_action(a)
            self.assertLessEqual(abs(d).max(),6+1e-12)
            self.assertAlmostEqual(float(d.mean()),2*np.tanh(np.clip(a[0],-10,10)),places=10)
        np.testing.assert_array_equal(map_action(np.zeros(36)),np.zeros((20,30)))
        with self.assertRaises(ValueError):map_action(np.full(36,np.nan))

    def test_return_equals_final_clip_objective(self):
        frames=12;total=0;cost=0;bits=0;f1=[]
        for t in range(1,frames+1):
            m={'tp':t%3,'fp':1,'fn':2};bits+=1000+t*700
            r,cost=reward_increment(m,bits,t,frames,80000,cost);total+=r
            f1.append(2*m['tp']/(2*m['tp']+3))
        self.assertAlmostEqual(total,clip_objective(np.mean(f1),bits*30/frames,80000),places=12)
        # No TP, all FN must score lower than retaining the objects at identical rate.
        bad,_=reward_increment({'tp':0,'fp':0,'fn':3},4000,1,1,120000,0)
        good,_=reward_increment({'tp':3,'fp':0,'fn':0},4000,1,1,120000,0)
        self.assertGreater(good,bad)

    def test_state_keeps_absolute_complexity_and_native_qpa(self):
        g=[30,0,0,0,0,0,0,0,0,0,0,0,0,0,2,10]
        maps=np.stack([np.arange(600)+1.,np.arange(600)+2.,np.arange(600)+3.,np.full(600,.05)])
        event={'schema':'vvenc-rcdot-vvc-adapter/3','poc':0,'lookahead_last_poc':10,
               'history_poc':-1,'global':g,'maps':maps.tolist()}
        a=state(event,256,50000);event['maps']=(maps*np.array([2,2,2,1])[:,None]).tolist()
        b=state(event,256,200000)
        self.assertEqual(a['global'].shape,(32,));self.assertEqual(a['mb'].shape,(6,20,30))
        self.assertFalse(np.array_equal(a['global'][16:],b['global'][16:]))
        self.assertTrue(DEV_OBSERVATIONS.contains(a))

    def test_spatial_head_has_local_connectivity(self):
        head=SpatialActionHead()
        with torch.no_grad():head.region[-1].weight.fill_(.1)
        a=torch.zeros(1,64);b=a.clone();b[0,4]=1
        delta=(head(b)-head(a)).detach().numpy()[0]
        self.assertNotEqual(delta[5],0)
        self.assertEqual(np.count_nonzero(delta),1)

    def test_auxiliary_covers_missed_source_boxes_and_rejects_duplicates(self):
        raw=np.array([[0,0,32,32,.9],[64,64,80,80,.8]],float)
        pred=np.array([[0,0,32,32,.9],[0,0,32,32,.8]],float)
        target,mask=retention_supervision(raw,pred)
        self.assertAlmostEqual(float(target[0,0]),2/3,places=6)
        self.assertEqual(target[4,4],0);self.assertEqual(mask[4,4],1)
        self.assertEqual(int(mask.sum()),5)
        z,m=retention_supervision([],[])
        self.assertEqual(z.sum()+m.sum(),0)

    def test_profile_seals_gamma(self):
        a=parser().parse_args(['--input-root','x','--encoder','x','--decoder','x','--yolo-repo','x','--weights','x','--output','x','--agent-profile','qpdev1'])
        self.assertEqual(ppo_options(a)['gamma'],1.0)

    def test_notebook_run_and_resume_bind_development_profile(self):
        import ast,contextlib,io
        p=Path(__file__).resolve().parents[2]/'notebooks/VVC_RL_QPDev1_Train_Validation_Kaggle.ipynb'
        n=json.loads(p.read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
        self.assertIn("'--agent-profile',AGENT_PROFILE",code)
        self.assertIn("method.get('agent_profile')==AGENT_PROFILE",code)
        config=next(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code').split('\nimport torch')[0]
        for profile in ('paper','qpdev1','qpdev-state','qpdev-reward'):
            tree=ast.parse(config)
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='AGENT_PROFILE' for t in node.targets):
                    node.value=ast.Constant(profile)
            ast.fix_missing_locations(tree)
            with contextlib.redirect_stdout(io.StringIO()):
                ns={};exec(compile(tree,str(p),'exec'),ns)
            self.assertEqual(ns['ADDITIONAL_TIMESTEPS'],20480)
            self.assertEqual(ns['VAL_RATES'],'50,120,200')
            self.assertEqual(ns['PPO_PROFILE'].get('gamma',.99),1 if profile!='paper' else .99)
            self.assertIn(profile.replace('-','_'),str(ns['RUN']))
        for c in n['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))

@unittest.skipUnless(os.environ.get('RCDOT_TEST_ENCODER'),'Requires the repaired encoder')
class DevelopmentNative(unittest.TestCase):
    def test_native_update_resume_and_paired_validation(self):
        torch.set_num_threads(2)
        for profile in ('qpdev-reward','qpdev-state','qpdev1'):
            with self.subTest(profile=profile),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);enc=os.environ['RCDOT_TEST_ENCODER']
                spec={'output':str(root),'encoder':enc,'encoder_threads':2,'input_root':str(root),
                    'ffmpeg':'unused','ffprobe':'unused','yolo_repo':'fixture','weights':'fixture',
                    'gpu_ids':[],'validation_videos':1,'validation_rates':[120],'validation_frames':8,
                    'seed':7,'val_every_updates':25,'agent_profile':profile}
                factory=lambda:VVenCEnv(FixtureCatalog(),FixtureDetector(),enc,root/'episodes',frame_limit=8,
                                       threads=2,agent_profile=profile)
                env=DummyVecEnv([factory]);model=build(env,7,n_steps=8,batch_size=4,n_epochs=1,gamma=1)
                contract={'profile':profile,'fixture':True};atomic_json(root/'method.json',contract)
                try:
                    obs=env.reset()
                    if profile=='qpdev1':np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],np.zeros((1,36)))
                    before=policy_digest(model);model.learn(8)
                    self.assertNotEqual(before,policy_digest(model))
                    checkpoint_obs=env.reset();saved_prediction=model.predict(checkpoint_obs,deterministic=True)[0]
                    ref=save_checkpoint(model,env,root,contract)
                finally:model.get_env().close()
                env2=DummyVecEnv([factory]);resumed,_=load_resume(root,contract,env2)
                try:
                    self.assertEqual(resumed.gamma,1)
                    np.testing.assert_array_equal(resumed.predict(checkpoint_obs,deterministic=True)[0],saved_prediction)
                    resumed.learn(8,reset_num_timesteps=False)
                    v=validate(resumed,spec,detector_factory=lambda *args:FixtureDetector(),
                        catalog_factory=lambda *args,**kwargs:FixtureCatalog())
                    self.assertAlmostEqual(v['score'],v['points'][0]['agent']['objective_score'])
                    for name in ('agent','anchor','zero'):
                        e=v['points'][0][name]
                        self.assertAlmostEqual(e['r'],e['objective_score'],places=8)
                    self.assertTrue(list((root/'validation_media').rglob('cus.jsonl.gz')))
                    self.assertTrue(np.isfinite(resumed.last_update_metrics['requested_map_spatial_std']))
                    if profile=='qpdev1':
                        from scripts.rl_rc_dot.development_report import summarize
                        atomic_json(root/'validations/test.json',v)
                        ref=save_checkpoint(resumed,env2,root,contract)
                        save_checkpoint(resumed,env2,root,contract,best={**ref,'score':v['score'],'validation':'validations/test.json'})
                        report=summarize(root)
                        self.assertFalse(report['advance_to_long_training']) # no initial validation fixture
                        self.assertTrue((root/'reports/requested_and_actual_qp.png').is_file())
                        exported=export_run(root,root/'export.zip')
                        import_run(exported['safe_input'],root/'restored')
                        self.assertTrue((root/'restored/reports/development_summary.json').is_file())
                finally:resumed.get_env().close()

if __name__=='__main__':unittest.main()
