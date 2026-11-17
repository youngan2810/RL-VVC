"""Standalone VVenC D3QN environment. Reuses only the audited codec ABI.

Never imports the PPO environment, PPO state, PPO action mapper, or PPO reward.
The detector is intentionally used on SOURCE before the current native action.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np

from .codec import NativeEpisode
from .method import FRAME_BYTES, detection_assignments
from .d3qn_method import (NEUTRAL_ACTION, observation, qp_action_map,
                          f1_from_counts, reward_step, clip_objective)


class D3QNNativeEnv:
    def __init__(self,catalog,detector,encoder,output,*,split='train',frames=256,
                 rate=None,video=None,threads=2,mode='live',random_windows=False,retain_media=False):
        if split not in ('train','val','test') or mode not in ('anchor','live'):
            raise ValueError('Invalid split or native control mode')
        if frames<1 or threads<1:
            raise ValueError('Invalid frame/threads')
        self.ids=list(catalog.selected[split])
        if not self.ids or (video is not None and video not in self.ids):
            raise ValueError('Requested source not in selected split')
        self.catalog,self.detector=catalog,detector
        self.encoder=Path(encoder);self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.frames,self.threads,self.mode=frames,threads,mode
        self.rate,self.video=rate,video
        self.random_windows=bool(random_windows)
        self.retain_media=bool(retain_media)
        if self.random_windows and (split!='train' or frames!=256 or video is not None):
            raise ValueError('Random-window training only for mini train 256')
        self.native=None;self.source=None;self.trace=None;self.action_audit=None
        self.audit_enabled=(split=='val')
        self.np_random=np.random.default_rng(20261111)

    def _cleanup(self):
        if self.trace is not None:
            self.trace.close()
            self.trace=None
        if self.native is not None:
            self.native.close()
            self.native=None
        if self.source is not None:
            self.source._mmap.close()
            self.source=None
        if not self.retain_media and hasattr(self,'folder'):
            (self.folder/'source.yuv').unlink(missing_ok=True)

    def close(self):
        self._cleanup()

    def _before_action(self):
        poc=self.native.current['poc']
        if type(poc) is not int or not 0<=poc<self.frames:
            raise ValueError('Invalid current source POC')
        t0=time.perf_counter()
        self.source_boxes=np.asarray(self.detector(self.source[poc]),dtype=float)
        self.source_det_seconds+=time.perf_counter()-t0
        obs,self.importance=observation(
            self.native.current,self.source_boxes,frames=self.frames,
            target_bps=self.target_bps,completed=self.completed,
            previous_action=self.prev_action,previous_au_bits=self.prev_au_bits,
            previous_f1=self.prev_f1,cumulative_bits=self.native.bits)
        return obs

    def reset(self,*,seed=None):
        self._cleanup()
        if seed is not None:self.np_random=np.random.default_rng(seed)
        video=self.video or self.ids[int(self.np_random.integers(len(self.ids)))]
        rate=int(self.rate if self.rate is not None else self.np_random.integers(50,201))
        if rate<=0:raise ValueError('Invalid bitrate')
        folder=self.output/f'{video}_{rate}_{uuid.uuid4().hex[:8]}'
        folder.mkdir()
        source=folder/'source.yuv'
        start_frame=int(self.np_random.choice([0,128,256,384,512])) if self.random_windows else 0
        produced=self.catalog.prepare(video,source,self.frames,start_frame=start_frame) if start_frame else self.catalog.prepare(video,source,self.frames)
        if produced!=self.frames:raise ValueError('Unexpected length after YUV preprocessing')
        self.source=np.memmap(source,mode='r',dtype=np.uint8,shape=(self.frames,FRAME_BYTES))
        self.trace=gzip.open(folder/'trajectory.jsonl.gz','wt',encoding='utf-8')
        do_audit=self.audit_enabled and self.mode=='live'
        self.native=NativeEpisode(self.encoder,source,folder/'codec',self.frames,rate,mode=self.mode,threads=self.threads,
                                  trace_cus=do_audit,trace_qgs=do_audit)
        if do_audit:
            from .action_audit import ActionAudit
            self.action_audit=ActionAudit(self.native.output)
        else:self.action_audit=None
        self.target_bps=rate*1000.
        self.completed=0;self.prev_action=NEUTRAL_ACTION;self.prev_au_bits=0;self.prev_f1=0.
        self.frame_f1=[];self.rewards=[];self.action_history=np.zeros(25,dtype=np.int64)
        self.source_det_seconds=0.;self.recon_det_seconds=0.
        self.folder=folder;self.video_id=video;self.start_frame=start_frame;self.rate_kbps=rate
        obs=self._before_action()
        self.current_obs=obs
        return obs,{'video_id':video,'rate_kbps':rate,'start_frame':start_frame,'folder':str(folder)}

    def step(self,action):
        if self.native is None or self.native.current is None:
            raise RuntimeError('Call reset before action')
        if (isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer))
                or not 0<=action<25):
            raise ValueError('Expected integer D3QN action 0..24')
        if not self.current_obs['legal'][action]:
            raise ValueError('D3QN selected a masked action')
        poc=self.native.current['poc']
        before=self.native.bits
        delta=qp_action_map(action,self.importance)
        post,nxt=self.native.step(delta)
        if post['poc']!=poc or post['cumulative_bits']!=before+post['au_bits']:
            raise ValueError('Inconsistent native picture/bitstream outcome')
        audit=self.action_audit.consume(post,delta) if self.action_audit else None
        rec=(self.native.recon/f'{poc}.yuv').read_bytes()
        if len(rec)!=FRAME_BYTES*2:
            raise ValueError('Invalid reconstruction size')
        decoded=np.frombuffer(rec,dtype='<u2')
        decoded=np.minimum(255,(decoded.astype(np.uint32)+2)>>2).astype(np.uint8)
        t0=time.perf_counter()
        predicted=np.asarray(self.detector(decoded),dtype=float)
        self.recon_det_seconds+=time.perf_counter()-t0
        raw,pred,correct,_=detection_assignments(self.source_boxes,predicted)
        tp=int(correct.sum());fp=int(len(pred)-tp);fn=int(len(raw)-tp)
        f1=f1_from_counts(tp,fp,fn)
        reward,_=reward_step(f1,post['au_bits'],before,self.completed,
                             frames=self.frames,target_bps=self.target_bps)
        self.completed+=1
        self.frame_f1.append(f1);self.rewards.append(reward)
        self.action_history[int(action)]+=1
        row={'poc':poc,'action':int(action),'delta_roi':int(action)//5-2,
             'delta_background':int(action)%5-2,
             'roi_ctu_fraction':float((self.importance>0).mean()),
             'tp':tp,'fp':fp,'fn':fn,'frame_f1':f1,'reward':reward,
             'au_bits':post['au_bits'],'cumulative_bits':post['cumulative_bits'],
             'reconstruction_sha256':hashlib.sha256(rec).hexdigest()}
        if audit is not None:row['native_action_audit']=audit
        self.trace.write(json.dumps(row,allow_nan=False)+'\n')
        self.trace.flush()
        if not self.retain_media:
            (self.native.recon/f'{poc}.yuv').unlink(missing_ok=True)
            (self.native.actions/f'{poc}.txt').unlink(missing_ok=True)
        self.prev_action=int(action);self.prev_au_bits=post['au_bits'];self.prev_f1=f1
        done=nxt is None
        info={'poc':poc,'frame_f1':f1,'au_bits':post['au_bits'],
              'roi_ctu_fraction':row['roi_ctu_fraction'],'action':int(action),
              'delta_map':delta,'codec_post':post}
        if audit is not None:
            info['native_action_audit']=audit
        if done:
            actual_bps=self.native.bits*30/self.frames
            objective=clip_objective(self.frame_f1,self.native.bits,
                                     frames=self.frames,target_bps=self.target_bps)
            if abs(objective-sum(self.rewards))>1e-6:
                raise AssertionError('D3QN rewards do not sum to declared clip objective')
            info['episode']={'mean_frame_f1':float(np.mean(self.frame_f1)),
                             'objective':objective,'actual_kbps':actual_bps/1000.,
                             'rate_ratio':actual_bps/self.target_bps,
                             'actions':self.action_history.tolist(),
                             'source_detector_seconds':self.source_det_seconds,
                             'decoded_detector_seconds':self.recon_det_seconds}
            if self.action_audit:
                info['episode']['native_action_audit']=self.action_audit.summary()
            (self.folder/'episode_summary.json').write_text(json.dumps(info['episode'],indent=2),encoding='utf-8')
            self.trace.close();self.trace=None
            neutral=np.zeros(25,dtype=bool);neutral[NEUTRAL_ACTION]=True
            new_obs={'global':np.zeros(25,dtype=np.float32),'maps':np.zeros((5,10,15),dtype=np.float32),'legal':neutral}
        else:
            new_obs=self._before_action()
        self.current_obs=new_obs
        return new_obs,reward,done,False,info

    @property
    def current_obs(self):
        return self._current_obs

    @current_obs.setter
    def current_obs(self,value):
        self._current_obs=value
