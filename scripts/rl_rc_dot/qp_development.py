"""QP development v1: causal budget state, spatial residual control and clip objective.

This is an experimental proposal, not a claim of improved detection. Codec,
data lock and detector are unchanged. No detector output is a policy input.
"""
import math
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from .method import observation, action_map,detection_assignments

PROFILES=('paper','qpdev-reward','qpdev-state','qpdev1')
VERSION='vvc-qp-development/1'
RATE_WEIGHT=0.1
DEV_OBSERVATIONS=spaces.Dict({
    'global':spaces.Box(-np.inf,np.inf,(32,),dtype=np.float32),
    'mb':spaces.Box(-np.inf,np.inf,(6,20,30),dtype=np.float32)})
DEV_ACTIONS=spaces.Box(-10,10,(36,),dtype=np.float32)

def check_profile(profile):
    if profile not in PROFILES:raise ValueError('Unknown agent profile: '+str(profile))
    return profile

def augmented(profile):return check_profile(profile) in ('qpdev-state','qpdev1')

def map_action(action):
    """35 = historical action; 36 = frame offset + centered spatial residual.

    Spatial centering does not imply constant bitrate: native QG pooling,
    clipping and nonlinear RDO remain authoritative.
    """
    a=np.asarray(action,dtype=np.float64)
    if a.shape==(35,):return action_map(a)
    if a.shape!=(36,) or not np.isfinite(a).all():raise ValueError('Expected 35 or 36 finite actions')
    global_delta=2.0*np.tanh(np.clip(a[0],-10,10))
    spatial=action_map(a[1:])/3.0
    spatial-=spatial.mean()
    spatial*=min(1.0,4.0/max(float(np.abs(spatial).max()),1e-12))
    return global_delta+spatial

def state(event,frames,target_bps,previous_delta=None,previous_au_bits=0):
    base=observation(event,frames)
    if not math.isfinite(target_bps) or target_bps<=0:raise ValueError('Invalid target bitrate')
    g=np.asarray(event['global'],dtype=np.float64)
    maps=np.asarray(event['maps'],dtype=np.float64).reshape(4,20,30)
    transformed=maps.copy();transformed[:3]=np.log1p(np.maximum(transformed[:3],0))
    # Absolute summaries survive per-picture spatial z-score normalization.
    summaries=np.r_[transformed.mean(axis=(1,2)),transformed.std(axis=(1,2))]/16.0
    bits=max(float(g[2]),0.0);completed=float(g[12])*frames
    budget=target_bps*frames/30.0
    previous=np.zeros((20,30)) if previous_delta is None else np.asarray(previous_delta,dtype=float)
    if previous.shape!=(20,30) or not np.isfinite(previous).all():raise ValueError('Invalid action history')
    history=np.array([math.log(target_bps/100000.0),np.clip(bits/budget,0,8),
        np.clip((budget-bits)/budget,-8,1),np.clip(1-completed/frames,0,1),
        np.clip(previous_au_bits/(target_bps/30),0,32)/8,
        np.clip(g[13],-8,8)/4,g[3]/63.0,float(event['history_poc']!=-1)])
    qpa=(4-6*np.log2(np.maximum(maps[3],1e-12)))/63.0
    result={'global':np.r_[base['global'],summaries,history].astype(np.float32),
            'mb':np.concatenate([base['mb'],np.clip(qpa,0,1)[None],(previous/6)[None]]).astype(np.float32)}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite development state')
    return result

def task_f1(metrics):
    tp,fp,fn=(metrics[k] for k in ('tp','fp','fn'))
    return 2.0*tp/max(2*tp+fp+fn,1)

def retention_supervision(raw,prediction):
    """Dense box-area local F1 labels; misses are supervised over source boxes.

    These are task-retention targets, not estimates of a causal QP advantage.
    Area is computed geometrically, including fractional boundary cells.
    """
    raw,pred,correct,matched=detection_assignments(raw,prediction)
    tp=np.zeros((20,30));fp=tp.copy();fn=tp.copy()
    xs=np.arange(30)*16;ys=np.arange(20)*16
    def add(target,box):
        wx=np.maximum(0,np.minimum(xs+16,box[2])-np.maximum(xs,box[0]))/16
        wy=np.maximum(0,np.minimum(ys+16,box[3])-np.maximum(ys,box[1]))/16
        target+=wy[:,None]*wx[None,:]
    retained=set(matched[correct].tolist())
    for i,box in enumerate(raw):add(tp if i in retained else fn,box)
    for box,good in zip(pred,correct):
        if not good:add(fp,box)
    denominator=2*tp+fp+fn
    target=np.divide(2*tp,denominator,out=np.zeros_like(tp),where=denominator>0)
    return target.astype(np.float32),(denominator>0).astype(np.float32)

def clip_objective(mean_frame_f1,actual_bps,target_bps):
    if (not np.isfinite([mean_frame_f1,actual_bps,target_bps]).all()
            or not 0<=mean_frame_f1<=1 or actual_bps<=0 or target_bps<=0):
        raise ValueError('Invalid clip objective')
    return float(mean_frame_f1-RATE_WEIGHT*abs(math.log(actual_bps/target_bps)))

def reward_increment(metrics,cumulative_bits,decisions,frames,target_bps,previous_rate_cost):
    """With gamma=1, the episode sum exactly equals the final clip objective.

    Dense F1/N plus differences of prefix rate cost avoids repeatedly counting
    the same early rate error. An interrupted episode is not a complete score.
    """
    if frames<1 or not 1<=decisions<=frames or cumulative_bits<=0:raise ValueError('Invalid episode progress')
    cost=abs(math.log(cumulative_bits*30.0/decisions/target_bps))
    return task_f1(metrics)/frames-RATE_WEIGHT*(cost-previous_rate_cost),cost

class SpatialExtractor(BaseFeaturesExtractor):
    """35 spatial features + 29 content/budget features; no giant flatten FC."""
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=64)
        self.cnn=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.pool=nn.AdaptiveAvgPool2d((5,7))
        self.spatial=nn.Conv2d(32,1,1)
        self.context=nn.Sequential(nn.Linear(32+32,64),nn.ReLU(),nn.Linear(64,29),nn.Tanh())
    def forward(self,obs):
        features=self.cnn(obs['mb'])
        local=self.spatial(self.pool(features)).flatten(1)
        context=self.context(torch.cat([obs['global'],features.mean(dim=(2,3))],dim=1))
        return torch.cat([local,context],dim=1)

class SpatialActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame=nn.Linear(29,1)
        self.region=nn.Sequential(nn.Linear(30,16),nn.Tanh(),nn.Linear(16,1))
        # A deterministic untrained policy starts at native QPA (zero offset).
        nn.init.zeros_(self.frame.weight);nn.init.zeros_(self.frame.bias)
        nn.init.zeros_(self.region[-1].weight);nn.init.zeros_(self.region[-1].bias)
    def forward(self,features):
        local,context=features[:,:35],features[:,35:]
        fused=torch.cat([local.unsqueeze(-1),context.unsqueeze(1).expand(-1,35,-1)],dim=-1)
        return torch.cat([self.frame(context),self.region(fused).squeeze(-1)],dim=1)

class SpatialPolicy(MultiInputActorCriticPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net=SpatialActionHead().to(self.device)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

def definition(profile):
    check_profile(profile)
    return {'version':VERSION,'profile':profile,'reward':'mean-frame F1 - 0.1*abs(log(final_rate/target)); telescoping rate cost; gamma=1',
            'state':'32 globals + 6 maps, encoder-only causal history' if augmented(profile) else 'historical 16+4',
            'action':'Gaussian36: frame +/-2 + centered spatial +/-4; total +/-6; native QPA residual' if profile=='qpdev1' else 'historical Gaussian35',
            'validation':'macro final clip objective; paired anchor and fresh initial-policy controls',
            'auxiliary':'box-area local F1 retention, including missed source objects and false-positive regions',
            'empty_frame_f1':0.0,'rate_weight':RATE_WEIGHT}
