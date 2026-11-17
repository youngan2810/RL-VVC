"""Masked Dueling Double DQN for the standalone 25-action VVC QP MDP.

Independent of the existing PPO actor, PPO rollout buffer and 36-bin action head.
The action mask is part of each replayed transition and Double-Q bootstrap.
"""
from __future__ import annotations
from collections import deque
import copy
import math
from pathlib import Path
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
        # Invalid action advantages must NOT change values of legal actions.
        # Dueling mean is computed over the currently legal actions only.
        legal=obs.get('legal')
        if legal is None or legal.ndim!=2 or tuple(legal.shape)!=(glob.shape[0],N_ACTIONS) or legal.dtype!=torch.bool:
            raise ValueError('Dueling Q network requires a Boolean legal-action mask')
        if not bool(legal.any(dim=1).all().item()):
            raise ValueError('Every D3QN state needs at least one legal action')
        count=legal.sum(dim=1,keepdim=True).clamp_min(1)
        mean_a=(a*legal).sum(dim=1,keepdim=True)/count
        return v+a-mean_a


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

    def state_dict(self):
        """Only tensors/primitives, compatible with torch.load(weights_only=True)."""
        if not self.rows:
            raise ValueError('Cannot snapshot an empty replay buffer')
        rows=list(self.rows)
        def stack_obs(which, part):
            return torch.as_tensor(np.stack([row[which][part] for row in rows]).copy())
        return {
            'capacity':self.rows.maxlen,
            'obs_global':stack_obs(0,0),'obs_maps':stack_obs(0,1),
            'obs_legal':stack_obs(0,2),
            'actions':torch.tensor([row[1] for row in rows],dtype=torch.int64),
            'rewards':torch.tensor([row[2] for row in rows],dtype=torch.float32),
            'next_global':stack_obs(3,0),'next_maps':stack_obs(3,1),
            'next_legal':stack_obs(3,2),
            'dones':torch.tensor([row[4] for row in rows],dtype=torch.bool),
            'rng_state':self.rng.bit_generator.state}

    def load_state_dict(self,data):
        n=len(data['actions'])
        if n<1 or n>data['capacity'] or data['capacity']!=self.rows.maxlen:
            raise ValueError('Replay capacity/size mismatch')
        self.rows.clear()
        keys=('obs_global','obs_maps','obs_legal','rewards','next_global','next_maps','next_legal','dones')
        if any(len(data[k])!=n for k in keys):
            raise ValueError('Inconsistent replay field length')
        for i in range(n):
            old=(data['obs_global'][i].numpy().copy(),
                 data['obs_maps'][i].numpy().copy(),
                 data['obs_legal'][i].numpy().copy())
            nxt=(data['next_global'][i].numpy().copy(),
                 data['next_maps'][i].numpy().copy(),
                 data['next_legal'][i].numpy().copy())
            self.rows.append((old,int(data['actions'][i]),float(data['rewards'][i]),nxt,bool(data['dones'][i])))
        self.rng.bit_generator.state=data['rng_state']

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
                           'maps':torch.as_tensor(maps[None].astype(np.float32),device=self.device),
                           'legal':torch.as_tensor(legal[None],device=self.device)})[0]
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
        # Do not hide a collapsed 1-action policy behind a decreasing TD loss.
        with torch.no_grad():
            values=self.online(obs_b)
            highest=values.masked_fill(~obs_b['legal'],float('-inf')).max(dim=1).values
            lowest=values.masked_fill(~obs_b['legal'],float('inf')).min(dim=1).values
            q_gap=(highest-lowest).mean()
            q_max=highest.mean()
            chosen_greedy=values.masked_fill(~obs_b['legal'],float('-inf')).argmax(dim=1)
            unique_greedy=int(torch.unique(chosen_greedy).numel())
        if not torch.isfinite(q_gap) or not torch.isfinite(q_max):
            raise FloatingPointError('Nonfinite D3QN Q-value diagnostics')
        return {'loss':float(loss.detach()),'mean_abs_td':float((td-predicted).abs().mean().detach()),
                'grad_norm':float(grad),'updates':self.updates,
                'q_gap_valid':float(q_gap),'q_max_valid':float(q_max),
                'unique_greedy_actions_in_batch':unique_greedy}

    def save(self,path,*,identity,next_episode_seed=None):
        """Atomic full replay + optimizer + RNG snapshot at an episode boundary."""
        if self.steps<1 or self.steps%256:
            raise ValueError('Save only after complete 256-picture training episodes')
        if next_episode_seed is None:
            raise ValueError('A resumable episode seed must be recorded')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload={
            'version':'d3qn-native-checkpoint/v2','algorithm':'masked-dueling-double-dqn',
            'identity':identity,'next_episode_seed':int(next_episode_seed),
            'online':self.online.state_dict(),'target':self.target.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'steps':self.steps,'updates':self.updates,
            'config':{'gamma':self.gamma,'batch_size':self.batch_size,
                      'warmup':self.warmup,'train_every':self.train_every,
                      'target_sync_updates':self.target_sync_updates,
                      'reward_scale':self.reward_scale,
                      'lr':self.optimizer.param_groups[0]['lr'],
                      'replay_capacity':self.replay.rows.maxlen},
            'replay':self.replay.state_dict(),
            'agent_rng_state':self.rng.bit_generator.state,
            'torch_cpu_rng_state':torch.get_rng_state(),
            'torch_cuda_rng_states':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        staging=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,staging)
        staging.replace(path)

    @classmethod
    def load(cls,path,*,identity,device='cpu'):
        """Load trusted tensor/primitives checkpoint; reject changed experiment."""
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('version')!='d3qn-native-checkpoint/v2' or data.get('identity')!=identity:
            raise ValueError('D3QN checkpoint identity/protocol mismatch; never use PPO checkpoint')
        cfg=data['config']
        agent=cls(device=device,gamma=cfg['gamma'],lr=cfg['lr'],
                  batch_size=cfg['batch_size'],replay_capacity=cfg['replay_capacity'],
                  warmup=cfg['warmup'],train_every=cfg['train_every'],
                  target_sync_updates=cfg['target_sync_updates'],
                  reward_scale=cfg['reward_scale'])
        agent.online.load_state_dict(data['online'])
        agent.target.load_state_dict(data['target'])
        agent.target.eval()
        agent.optimizer.load_state_dict(data['optimizer'])
        # Optimizer state needs to live on the same device as its parameters.
        for state in agent.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):
                    state[k]=v.to(agent.device)
        agent.steps=int(data['steps'])
        agent.updates=int(data['updates'])
        agent.replay.load_state_dict(data['replay'])
        agent.rng.bit_generator.state=data['agent_rng_state']
        torch.set_rng_state(data['torch_cpu_rng_state'])
        if agent.device.type=='cuda' and data['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(data['torch_cuda_rng_states'])
        return agent,int(data['next_episode_seed'])
