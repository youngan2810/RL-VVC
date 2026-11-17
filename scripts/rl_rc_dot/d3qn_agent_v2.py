"""Masked Dueling Double DQN for the standalone 25-action VVC QP MDP.

Independent of the existing PPO actor, PPO rollout buffer and 36-bin action head.
The action mask is part of each replayed transition and Double-Q bootstrap.
"""
from __future__ import annotations
from collections import deque
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .d3qn_method import GLOBAL_DIM, MAP_CHANNELS, N_ACTIONS


class DuelingQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(
            nn.Conv2d(MAP_CHANNELS,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((5,5)),nn.Flatten())
        self.fusion=nn.Sequential(nn.Linear(32*25+GLOBAL_DIM,128),nn.ReLU(),
                                  nn.Linear(128,64),nn.ReLU())
        self.value=nn.Linear(64,1)
        self.advantage=nn.Linear(64,N_ACTIONS)
        for layer in (self.value,self.advantage):
            nn.init.orthogonal_(layer.weight,gain=.01)
            nn.init.zeros_(layer.bias)

    def forward(self,obs):
        glob, maps=obs['global'],obs['maps']
        if glob.ndim!=2 or tuple(glob.shape[1:])!=(GLOBAL_DIM,) or maps.ndim!=4 or tuple(maps.shape[1:])!=(MAP_CHANNELS,10,15) or glob.shape[0]!=maps.shape[0]:
            raise ValueError('Invalid D3QN state tensor dimensions')
        context=self.fusion(torch.cat([glob,self.spatial(maps)],dim=1))
        v,a=self.value(context),self.advantage(context)
        return v+a-a.mean(dim=1,keepdim=True)


def pack(obs):
    if set(obs)!={'global','maps','legal'}:
        raise ValueError('D3QN state requires global, maps, legal')
    g=np.asarray(obs['global'],dtype=np.float32)
    m=np.asarray(obs['maps'],dtype=np.float32)
    legal=np.asarray(obs['legal'])
    if (g.shape!=(GLOBAL_DIM,) or m.shape!=(MAP_CHANNELS,10,15) or legal.shape!=(N_ACTIONS,) or
            legal.dtype!=np.bool_ or not legal.any() or
            not np.isfinite(g).all() or not np.isfinite(m).all()):
        raise ValueError('Invalid D3QN state or legal-action mask')
    return g.copy(),m.astype(np.float16),legal.copy()


class ReplayBuffer:
    def __init__(self,capacity=10000,seed=20261111):
        if capacity<1: raise ValueError('Invalid replay capacity')
        self.rows=deque(maxlen=capacity)
        self.rng=np.random.default_rng(seed)

    def append(self,obs,action,reward,next_obs,done):
        a=int(action)
        pre=pack(obs);post=pack(next_obs)
        if (not isinstance(action,(int,np.integer)) or not 0<=a<N_ACTIONS or
                not pre[2][a] or not math.isfinite(reward)):
            raise ValueError('Illegal action or nonfinite reward')
        self.rows.append((pre,a,float(reward),post,bool(done)))

    def __len__(self):return len(self.rows)

    def sample(self,batch_size,device):
        if not 1<=batch_size<=len(self.rows):
            raise ValueError('Not enough samples for minibatch')
        batch=[self.rows[i] for i in self.rng.choice(len(self.rows),batch_size,replace=False)]
        def observations(slot):
            packed=[r[slot] for r in batch]
            return {'global':torch.as_tensor(np.stack([p[0] for p in packed]),device=device),
                    'maps':torch.as_tensor(np.stack([p[1] for p in packed]).astype(np.float32),device=device),
                    'legal':torch.as_tensor(np.stack([p[2] for p in packed]),device=device)}
        return (observations(0),
                torch.as_tensor([r[1] for r in batch],dtype=torch.int64,device=device),
                torch.as_tensor([r[2] for r in batch],dtype=torch.float32,device=device),
                observations(3),
                torch.as_tensor([r[4] for r in batch],dtype=torch.bool,device=device))


class MaskedD3QN:
    def __init__(self,*,seed=20261111,device='cpu',gamma=1.,lr=1e-4,
                 batch_size=64,replay_capacity=10000,warmup=512,
                 train_every=4,target_sync_updates=250,reward_scale=25.):
        if (not 0<=gamma<=1 or lr<=0 or batch_size<2 or warmup<batch_size or
                train_every<1 or target_sync_updates<1 or reward_scale<=0):
            raise ValueError('Invalid Dueling Double DQN configuration')
        torch.manual_seed(seed)
        self.device=torch.device(device)
        self.online=DuelingQNetwork().to(self.device)
        self.target=copy.deepcopy(self.online).eval()
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.replay=ReplayBuffer(replay_capacity,seed)
        self.rng=np.random.default_rng(seed)
        self.gamma=gamma;self.reward_scale=reward_scale
        self.batch_size=batch_size;self.warmup=warmup
        self.train_every=train_every;self.target_sync_updates=target_sync_updates
        self.steps=0;self.updates=0

    def predict(self,obs,epsilon=0.):
        glob,maps,legal=pack(obs)
        if not 0<=epsilon<=1:raise ValueError('Epsilon outside [0,1]')
        if self.rng.random()<epsilon:
            return int(self.rng.choice(np.flatnonzero(legal)))
        self.online.eval()
        with torch.no_grad():
            q=self.online({'global':torch.as_tensor(glob[None],device=self.device),
                           'maps':torch.as_tensor(maps[None].astype(np.float32),device=self.device)})[0]
            q=q.masked_fill(~torch.as_tensor(legal,device=self.device),float('-inf'))
        return int(q.argmax().item())

    def observe(self,obs,action,reward,next_obs,done):
        self.replay.append(obs,action,float(reward)*self.reward_scale,next_obs,done)
        self.steps+=1
        if len(self.replay)<self.warmup or self.steps%self.train_every:
            return None
        obs_b,acts,rewards,nxt,dones=self.replay.sample(self.batch_size,self.device)
        self.online.train()
        with torch.no_grad():
            available=self.online(nxt).masked_fill(~nxt['legal'],float('-inf'))
            choice=available.argmax(dim=1,keepdim=True)
            selected=self.target(nxt).gather(1,choice).squeeze(1)
            td=rewards+self.gamma*selected*(~dones).float()
        predicted=self.online(obs_b).gather(1,acts[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(predicted,td)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite D3QN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad=nn.utils.clip_grad_norm_(self.online.parameters(),5.,error_if_nonfinite=True)
        self.optimizer.step()
        self.updates+=1
        if self.updates%self.target_sync_updates==0:
            self.target.load_state_dict(self.online.state_dict())
        return {'loss':float(loss.detach()),'mean_abs_td':float((td-predicted).abs().mean().detach()),
                'grad_norm':float(grad),'updates':self.updates}

    def save(self,path):
        torch.save({'algorithm':'masked-dueling-double-dqn','online':self.online.state_dict(),
                    'target':self.target.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'steps':self.steps,'updates':self.updates,'gamma':self.gamma,
                    'reward_scale':self.reward_scale,
                    'rng_state':self.rng.bit_generator.state,
                    'replay_rng_state':self.replay.rng.bit_generator.state},path)
