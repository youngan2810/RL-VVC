"""V5 28-global state extension of V4 terminal-aware masked Dueling Double DQN.

The *only training-time experimental variable relative to V4* is three
causal 32-step coding-cycle features in the observation. The network's first
linear input widens from 825 to 828; all remaining architecture/optimizer,
reward, replay quota, gamma=1, n-step=3, and target updates are preserved.

No V4 checkpoints can be loaded: 25-D V4 states are incompatible with V5.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .d3qn_method import N_ACTIONS
from .d3qn_state_v5 import GLOBAL_DIM_V5
from .d3qn_agent_v2 import DuelingQNetwork, pack as pack_v4
from .d3qn_agent_v4 import (
    StableMaskedD3QN as V4Agent,
    TerminalAwareReplayBuffer as V4Replay,
    exploration_epsilon,
)

CHECKPOINT_VERSION='d3qn-native-checkpoint/v5-coding-cycle'
STATE_SCHEMA='v5-coding-cycle-global28'


def pack_v5(obs):
    if set(obs)!={'global','maps','legal'}:
        raise ValueError('V5 state requires global, maps, legal')
    g=np.asarray(obs['global'],dtype=np.float32)
    maps=np.asarray(obs['maps'],dtype=np.float32)
    legal=np.asarray(obs['legal'])
    if (g.shape!=(GLOBAL_DIM_V5,) or maps.shape!=(5,10,15) or
            legal.shape!=(N_ACTIONS,) or legal.dtype!=np.bool_ or
            not legal.any() or not np.isfinite(g).all() or not np.isfinite(maps).all()):
        raise ValueError('Invalid V5 28-global state or legal action mask')
    return g.copy(),maps.astype(np.float16),legal.copy()


class DuelingQNetworkV5(DuelingQNetwork):
    """Same dueling CNN/MLP as V4; only the global input width is +3."""
    def __init__(self):
        super().__init__()
        self.fusion[0]=nn.Linear(32*25+GLOBAL_DIM_V5,128)

    def forward(self,obs):
        glob,maps=obs['global'],obs['maps']
        if (glob.ndim!=2 or tuple(glob.shape[1:])!=(GLOBAL_DIM_V5,)
                or maps.ndim!=4 or tuple(maps.shape[1:])!=(5,10,15)
                or glob.shape[0]!=maps.shape[0]):
            raise ValueError('V5 Dueling Q requires [B,28] globals and [B,5,10,15] maps')
        legal=obs.get('legal')
        if (legal is None or legal.ndim!=2
                or tuple(legal.shape)!=(glob.shape[0],N_ACTIONS)
                or legal.dtype!=torch.bool or not bool(legal.any(dim=1).all())):
            raise ValueError('V5 Dueling Q requires nonempty Boolean legal mask')
        context=self.fusion(torch.cat([glob,self.spatial(maps)],dim=1))
        value,advantage=self.value(context),self.advantage(context)
        mean_adv=(advantage*legal).sum(dim=1,keepdim=True)/legal.sum(dim=1,keepdim=True)
        return value+advantage-mean_adv


class TerminalAwareReplayBufferV5(V4Replay):
    """Inherit V4's exact unbiased-within-stratum batch selection policy.

    This sampler is *not* an importance-weighted estimator of uniform replay.
    Only the input-state schema changes.
    """
    def append(self,obs,action,reward,next_obs,done):
        a=int(action)
        pre=pack_v5(obs)
        post=pack_v5(next_obs)
        if (isinstance(action,(bool,np.bool_)) or
                not isinstance(action,(int,np.integer)) or
                not 0<=a<N_ACTIONS or not pre[2][a] or
                not math.isfinite(float(reward))):
            raise ValueError('Illegal action or nonfinite reward in V5 replay')
        self.rows.append((pre,a,float(reward),post,bool(done)))

    def load_state_dict(self,data):
        if (tuple(data['obs_global'].shape[1:])!=(GLOBAL_DIM_V5,) or
                tuple(data['next_global'].shape[1:])!=(GLOBAL_DIM_V5,)):
            raise ValueError('Cannot load V4 25-global replay into V5')
        super().load_state_dict(data)


class StableMaskedD3QN(V4Agent):
    """Preserve V4 optimizer update/Polyak/n-step logic, replace state width.

    Deliberately isolate checkpoint format and predict/replay validation.
    """
    def __init__(self,*,seed=20261111,device='cpu',lr=5e-5,gamma=1.0,
                 n_step=3,target_tau=0.005,batch_size=64,
                 replay_capacity=10000,warmup=512,train_every=4,
                 reward_scale=25.0,terminal_quota=4):
        super().__init__(seed=seed,device=device,lr=lr,gamma=gamma,
                         n_step=n_step,target_tau=target_tau,
                         batch_size=batch_size,replay_capacity=replay_capacity,
                         warmup=warmup,train_every=train_every,
                         reward_scale=reward_scale,terminal_quota=terminal_quota)
        # V4 constructor initializes 25-D networks. Replace both before ANY
        # transition/update, and reseed to make V5 fresh runs reproducible.
        torch.manual_seed(seed)
        self.online=DuelingQNetworkV5().to(self.device)
        self.target=copy.deepcopy(self.online).eval()
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.replay=TerminalAwareReplayBufferV5(replay_capacity,seed,terminal_quota)

    def predict(self,obs,epsilon=0.):
        g,m,legal=pack_v5(obs)
        if not 0<=epsilon<=1:
            raise ValueError('Epsilon outside [0,1]')
        if self.rng.random()<epsilon:
            return int(self.rng.choice(np.flatnonzero(legal)))
        self.online.eval()
        with torch.inference_mode():
            q=self.online({'global':torch.as_tensor(g[None],device=self.device),
                           'maps':torch.as_tensor(m[None].astype(np.float32),device=self.device),
                           'legal':torch.as_tensor(legal[None],device=self.device)})[0]
            q=q.masked_fill(~torch.as_tensor(legal,device=self.device),-torch.inf)
        return int(q.argmax().item())

    def observe(self,obs,action,reward,next_obs,done):
        _,_,legal=pack_v5(obs)
        pack_v5(next_obs)
        if (isinstance(action,(bool,np.bool_)) or
                not isinstance(action,(int,np.integer)) or
                not 0<=int(action)<N_ACTIONS or not legal[int(action)]
                or not math.isfinite(float(reward))):
            raise ValueError('Invalid V5 multi-step replay action/reward')
        self.pending.append((obs,int(action),float(reward)*self.reward_scale,
                             next_obs,bool(done)))
        if len(self.pending)>=self.n_step:
            self._emit_nstep()
        if done:
            while self.pending:
                self._emit_nstep()
        self.steps+=1
        if len(self.replay)<self.warmup or self.steps%self.train_every:
            return None
        return self._update()

    def save(self,path,*,identity,next_episode_seed=None):
        if self.steps<1 or self.steps%256 or self.pending:
            raise ValueError('V5 checkpoint requires clean completed 256-frame episode')
        if next_episode_seed is None or identity.get('protocol')!='native-masked-d3qn-qp/v5-coding-cycle':
            raise ValueError('V5 checkpoint identity/next seed missing')
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={
            'version':CHECKPOINT_VERSION,'state_schema':STATE_SCHEMA,
            'algorithm':'masked-dueling-double-dqn-nstep-polyak',
            'identity':identity,'next_episode_seed':int(next_episode_seed),
            'online':self.online.state_dict(),'target':self.target.state_dict(),
            'optimizer':self.optimizer.state_dict(),'steps':int(self.steps),
            'updates':int(self.updates),
            'config':{
                'gamma':self.gamma,'lr':self.optimizer.param_groups[0]['lr'],
                'batch_size':self.batch_size,'replay_capacity':self.replay.rows.maxlen,
                'warmup':self.warmup,'train_every':self.train_every,
                'reward_scale':self.reward_scale,'n_step':self.n_step,
                'target_tau':self.target_tau,'terminal_quota':self.terminal_quota,
                'global_dim':GLOBAL_DIM_V5},
            'replay':self.replay.state_dict(),
            'agent_rng_state':self.rng.bit_generator.state,
            'torch_cpu_rng_state':torch.get_rng_state(),
            'torch_cuda_rng_states':torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available() else []}
        tmp=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,tmp)
        tmp.replace(path)

    @classmethod
    def load(cls,path,*,identity,device='cpu'):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if (data.get('version')!=CHECKPOINT_VERSION
                or data.get('state_schema')!=STATE_SCHEMA
                or data.get('identity')!=identity
                or identity.get('protocol')!='native-masked-d3qn-qp/v5-coding-cycle'):
            raise ValueError('V5 source/state identity mismatch; V3/V4 checkpoints forbidden')
        cfg=data['config']
        if cfg.get('global_dim')!=GLOBAL_DIM_V5:
            raise ValueError('V5 checkpoint global dimension mismatch')
        if identity.get('learning',{}).get('terminal_quota',cfg['terminal_quota'])!=cfg['terminal_quota']:
            raise ValueError('Checkpoint terminal quota and experiment identity differ')
        model=cls(device=device,lr=cfg['lr'],gamma=cfg['gamma'],
                  n_step=cfg['n_step'],target_tau=cfg['target_tau'],
                  terminal_quota=cfg['terminal_quota'],
                  batch_size=cfg['batch_size'],replay_capacity=cfg['replay_capacity'],
                  warmup=cfg['warmup'],train_every=cfg['train_every'],
                  reward_scale=cfg['reward_scale'])
        model.online.load_state_dict(data['online'],strict=True)
        model.target.load_state_dict(data['target'],strict=True)
        model.target.eval()
        model.optimizer.load_state_dict(data['optimizer'])
        for state in model.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):
                    state[k]=v.to(model.device)
        model.steps=int(data['steps'])
        model.updates=int(data['updates'])
        model.replay.load_state_dict(data['replay'])
        model.rng.bit_generator.state=data['agent_rng_state']
        torch.set_rng_state(data['torch_cpu_rng_state'])
        if model.device.type=='cuda' and data['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(data['torch_cuda_rng_states'])
        return model,int(data['next_episode_seed'])
