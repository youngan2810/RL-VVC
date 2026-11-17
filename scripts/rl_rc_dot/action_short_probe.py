"""Short native PPO probe with explicit synthetic task fixture, not BDD detection.

Reports Random/No-op/Trained separately. A constant fixture detector cannot
establish useful car-detection learning; this is deliberately reported.
"""
import argparse,json,os
from pathlib import Path
from functools import partial
import torch,numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from .environment import VVenCEnv
from .model import build
from .session import validate
from .preflight import make_fixture
from .data import atomic_json

class ProbeCatalog:
    lock={'splits':{'train':['fixture_train'],'val':['fixture_val'],'test':['unused_test']}}
    def prepare(self,video,path,limit=0):make_fixture(path,limit or 32);return limit or 32
class ConstantDetector:
    def __call__(self,frame):return np.array([[20.,30.,60.,70.,.9]])
    def close(self):pass

def run(encoder,output,steps=256,decoder=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    spec={'input_root':str(root),'ffmpeg':'unused','ffprobe':'unused','seed':20261111,'gpu_ids':[],
          'yolo_repo':'fixture','weights':'fixture','output':str(root),'validation_videos':1,
          'validation_rates':[50,120,200],'validation_frames':32,'encoder':str(encoder),'encoder_threads':2,
          'agent_profile':'qpdev1-qbins'}
    if decoder:spec['decoder']=str(decoder)
    factory=lambda:VVenCEnv(ProbeCatalog(),ConstantDetector(),encoder,root/'episodes',frame_limit=32,
                           threads=2,agent_profile='qpdev1-qbins')
    env=DummyVecEnv([factory]);model=build(env,20261111,n_steps=32,batch_size=32,n_epochs=3,gamma=1,
                                        learning_rate=3e-5,target_kl=.05,aux_loss_coef=.1)
    validator=partial(validate,detector_factory=lambda *a:ConstantDetector(),catalog_factory=lambda *a,**kw:ProbeCatalog())
    updates=[]
    try:
        initial=validator(model,spec);atomic_json(root/'initial.json',initial)
        for _ in range(steps//32):
            model.learn(32,reset_num_timesteps=False);updates.append(model.last_update_metrics.copy())
        trained=validator(model,spec);atomic_json(root/'trained.json',trained)
        model.save(root/'probe_model.zip');atomic_json(root/'updates.json',updates)
    finally:model.get_env().close()
    result={'scope':'SYNTHETIC NATIVE CODEC + CONSTANT DETECTOR FIXTURE',
            'interactions':model.num_timesteps,'updates':model.completed_rollouts,
            'noop_objective':trained['zero_score'],'random_objective':trained['random_score'],
            'initial_objective':initial['score'],'trained_objective':trained['score'],
            'trained_f1_delta_vs_noop':trained['macro_delta_f1'],
            'action_statistics':[{k:p['agent'].get(k) for k in ('mean_qp','actual_kbps','requested_nonzero_fraction','native_action_audit')}
                                 for p in trained['points']],
            'useful_car_detection_policy_established':False,
            'reason':'Detector is constant; task score does not depend on reconstructed image. Real BDD Mini short Validation is still required.'}
    atomic_json(root/'SHORT_PROBE.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--encoder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--decoder',type=Path);a=p.parse_args()
    if a.steps<32 or a.steps%32:p.error('steps must be a positive multiple of 32')
    print(json.dumps(run(a.encoder,a.output,a.steps,a.decoder)),flush=True)
if __name__=='__main__':main()
