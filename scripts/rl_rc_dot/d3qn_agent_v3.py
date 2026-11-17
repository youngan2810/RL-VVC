"""Research V3: finite-horizon 3-step masked Dueling Double DQN.

V2 and its experiment/checkpoints remain unchanged and reproducible.
This is an explicitly new protocol, not an implicit resume of V2.
Gamma remains 1.0 to preserve the VVC per-clip reward telescoping.
Three-step replay propagates codec/detection consequences across pictures.
Polyak target updates limit abrupt lagged-target changes.
"""
from __future__ import annotations
from collections import deque
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .d3qn_agent_v2 import MaskedD3QN, pack
from .d3qn_method import N_ACTIONS

CHECKPOINT_VERSION='d3qn-native-checkpoint/v3'


def exploration_epsilon(steps, *, start=1.0, floor=0.20, decay_steps=10240):
    """Slow schedule: eps=0.60 at 5120 frames, not eps=0.05."""
    if not isinstance(steps,(int,np.integer)) or steps<0 or not (0<=floor<=start<=1) or decay_steps<1:
        raise ValueError('Invalid D3QN V3 exploration schedule')
    return max(float(floor),float(start)-(float(start)-float(floor))*min(1.,steps/decay_steps))


class StableMaskedD3QN(MaskedD3QN):
    def __init__(self, *, seed=20261111, device='cpu', lr=5e-5, gamma=1.0,
                 n_step=3, target_tau=0.005, batch_size=64,
                 replay_capacity=10000, warmup=512, train_every=4,
                 reward_scale=25.0):
        if gamma!=1.0:
            raise ValueError('V3 gamma must stay 1 for exact telescoping clip reward')
        if not 1<=n_step<=8 or not 0<target_tau<=1:
            raise ValueError('Invalid multi-step horizon or target interpolation factor')
        super().__init__(seed=seed, device=device, gamma=gamma, lr=lr,
                         batch_size=batch_size,replay_capacity=replay_capacity,
                         warmup=warmup,train_every=train_every,
                         target_sync_updates=250,reward_scale=reward_scale)
        self.n_step=int(n_step)
        self.target_tau=float(target_tau)
        self.pending=deque()

    def _emit_nstep(self):
        if not self.pending:
            raise AssertionError('Cannot emit an empty n-step buffer')
        head=self.pending[0]
        window=list(self.pending)[:self.n_step]
        discounted=0.0
        last=None
        for obs,action,reward,next_obs,done in window:
            discounted+=reward # gamma = 1 exactly: no objective distortion
            last=(next_obs,done)
            if done:
                break
        if not math.isfinite(discounted):
            raise FloatingPointError('Nonfinite aggregated n-step reward')
        self.replay.append(head[0],head[1],discounted,last[0],last[1])
        self.pending.popleft()

    def observe(self,obs,action,reward,next_obs,done):
        # Validate every transition immediately, even while awaiting 3-step lookahead.
        _,_,legal=pack(obs)
        pack(next_obs)
        if (isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer))
                or not 0<=int(action)<N_ACTIONS or not legal[int(action)]
                or not math.isfinite(float(reward))):
            raise ValueError('Invalid multi-step replay action/reward')
        self.pending.append((obs,int(action),float(reward)*self.reward_scale,next_obs,bool(done)))
        if len(self.pending)>=self.n_step:
            self._emit_nstep()
        if done:
            while self.pending:
                self._emit_nstep()
        self.steps+=1
        if len(self.replay)<self.warmup or self.steps%self.train_every:
            return None
        return self._update()

    def _update(self):
        obs,actions,rewards,nxt,dones=self.replay.sample(self.batch_size,self.device)
        self.online.train()
        with torch.no_grad():
            # Select with online, evaluate with the target: true Double DQN.
            next_online=self.online(nxt).masked_fill(~nxt['legal'],float('-inf'))
            choice=next_online.argmax(dim=1,keepdim=True)
            next_target=self.target(nxt).gather(1,choice).squeeze(1)
            td_target=rewards+self.gamma*next_target*(~dones).float()
        predicted=self.online(obs).gather(1,actions[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(predicted,td_target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite V3 D3QN Huber objective')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=nn.utils.clip_grad_norm_(self.online.parameters(),5.,error_if_nonfinite=True)
        self.optimizer.step()
        self.updates+=1
        with torch.no_grad():
            for target_param,online_param in zip(self.target.parameters(),self.online.parameters()):
                target_param.lerp_(online_param,self.target_tau)
            values=self.online(obs).masked_fill(~obs['legal'],float('-inf'))
            qmax=values.max(dim=1).values
            qmin=values.min(dim=1).values
            gap=(qmax-qmin).mean()
            greedy=values.argmax(dim=1)
        if not torch.isfinite(gap) or not torch.isfinite(qmax).all():
            raise FloatingPointError('Nonfinite V3 D3QN Q-value diagnostics')
        return {'loss':float(loss.detach()),
                'mean_abs_td':float((td_target-predicted).abs().mean().detach()),
                'grad_norm':float(norm),
                'updates':int(self.updates),
                'q_gap_valid':float(gap),
                'q_max_valid':float(qmax.mean()),
                'unique_greedy_actions_in_batch':int(torch.unique(greedy).numel())}

    def save(self,path,*,identity,next_episode_seed=None):
        if self.steps<1 or self.steps%256 or self.pending:
            raise ValueError('V3 checkpoints require a clean 256-frame episode boundary')
        if next_episode_seed is None:
            raise ValueError('Missing next-episode seed for reproducible resume')
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={
            'version':CHECKPOINT_VERSION,
            'algorithm':'masked-dueling-double-dqn-nstep-polyak',
            'identity':identity,
            'next_episode_seed':int(next_episode_seed),
            'online':self.online.state_dict(),
            'target':self.target.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'steps':int(self.steps),'updates':int(self.updates),
            'config':{'gamma':self.gamma,
                      'lr':self.optimizer.param_groups[0]['lr'],
                      'batch_size':self.batch_size,'replay_capacity':self.replay.rows.maxlen,
                      'warmup':self.warmup,'train_every':self.train_every,
                      'reward_scale':self.reward_scale,
                      'n_step':self.n_step,'target_tau':self.target_tau},
            'replay':self.replay.state_dict(),
            'agent_rng_state':self.rng.bit_generator.state,
            'torch_cpu_rng_state':torch.get_rng_state(),
            'torch_cuda_rng_states':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        tmp=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,tmp)
        tmp.replace(path)

    @classmethod
    def load(cls,path,*,identity,device='cpu'):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('version')!=CHECKPOINT_VERSION or data.get('identity')!=identity:
            raise ValueError('V3 experiment/source identity mismatch: V2/PPO checkpoints are forbidden')
        cfg=data['config']
        model=cls(device=device,lr=cfg['lr'],gamma=cfg['gamma'],
                  n_step=cfg['n_step'],target_tau=cfg['target_tau'],
                  batch_size=cfg['batch_size'],replay_capacity=cfg['replay_capacity'],
                  warmup=cfg['warmup'],train_every=cfg['train_every'],
                  reward_scale=cfg['reward_scale'])
        model.online.load_state_dict(data['online'])
        model.target.load_state_dict(data['target'])
        model.target.eval()
        model.optimizer.load_state_dict(data['optimizer'])
        for state in model.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(model.device)
        model.steps=int(data['steps'])
        model.updates=int(data['updates'])
        model.replay.load_state_dict(data['replay'])
        model.rng.bit_generator.state=data['agent_rng_state']
        torch.set_rng_state(data['torch_cpu_rng_state'])
        if model.device.type=='cuda' and data['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(data['torch_cuda_rng_states'])
        return model,int(data['next_episode_seed'])
