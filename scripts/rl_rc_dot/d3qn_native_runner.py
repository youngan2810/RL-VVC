"""Native VVenC + YOLO D3QN experiment: isolated probe / train / validate.

- Mini BDD100K Train16 / Val4 only. No Test access.
- Full replay/optimizer/RNG checkpoint each complete 256-frame training episode.
- Training and the computationally costly validation are separate phases.
- Validations use independent VVdeC reconstruction and native QG/lambda audits.
"""
from __future__ import annotations
import argparse
import gzip
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .data import VideoCatalog, FrozenDetector, atomic_json, digest, ASSETS, source_fingerprint
from .d3qn_native_env import D3QNNativeEnv
from .d3qn_agent_v2 import MaskedD3QN
from .d3qn_method import NEUTRAL_ACTION, N_ACTIONS, qp_action_map, balanced_train_video
from .evaluate import verify_decoded

PROTOCOL = 'native-masked-d3qn-qp/v2'
DATASET_SPLIT_SEED = 20261111  # Fixed across ALL agent seeds; same Mini16/Val4 as PPO.
FIXED_CONTROLS = (
    ('qpa_anchor',NEUTRAL_ACTION),
    ('global_minus1',6),                # (-1,-1)
    ('global_plus1',18),                # (+1,+1)
    ('roi_minus1_bg_plus1',8),          # (-1,+1)
    ('roi_minus2_bg_plus2',4),          # (-2,+2)
    ('roi_zero_bg_plus2',14),           # (0,+2)
)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('probe','train','validate'),required=True)
    for name in ('input-root','encoder','decoder','yolo-repo','weights','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--checkpoint-in',type=Path)
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--ffprobe',default='ffprobe')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--train-steps',type=int,default=5120,
                   help='Additional 256-frame transitions THIS run; checkpoint remains resumable')
    p.add_argument('--encoder-threads',type=int,default=2)
    p.add_argument('--val-videos',type=int,default=4)
    p.add_argument('--val-start',type=int,default=0,
                   help='Start index in locked Mini Val4 (for chunked Kaggle validation)')
    p.add_argument('--val-rates',default='50,120,200')
    p.add_argument('--val-methods',default='all',
                   help='Comma-separated subset of qpa_anchor, global_minus1, global_plus1, '
                        'roi_minus1_bg_plus1, roi_minus2_bg_plus2, roi_zero_bg_plus2, trained_d3qn')
    p.add_argument('--max-hours',type=float,default=0.,
                   help='Optional wall clock budget including Kaggle setup')
    p.add_argument('--session-start-utc',type=float,default=0.)
    return p


def canonical_fixed_action(action,obs):
    """Equivalent legal action for no-object/all-object fixed-QP probes."""
    if not isinstance(action,(int,np.integer)) or not 0<=int(action)<N_ACTIONS:
        raise ValueError('Invalid fixed action')
    if obs['legal'][action]:
        return int(action)
    wanted=qp_action_map(action,obs['maps'][4])
    equivalent=[int(x) for x in np.flatnonzero(obs['legal'])
                if np.array_equal(qp_action_map(int(x),obs['maps'][4]),wanted)]
    if len(equivalent)!=1:raise ValueError('No unique legal equivalent for fixed-QP probe')
    return equivalent[0]


def run_episode(env,selector,seed):
    obs,metadata=env.reset(seed=seed)
    while True:
        action=int(selector(obs))
        obs,reward,done,truncated,info=env.step(action)
        if truncated:raise RuntimeError('Unexpected native truncated episode')
        if done:break
    return metadata,info['episode'],env.native.output/'stream.266',env.folder/'trajectory.jsonl.gz'


def compress_audit(output):
    """Avoid 72-84 large raw CU/QG traces, keep lossless evidence."""
    for name in ('events.jsonl','cus.jsonl','qgs.jsonl'):
        src=output/name
        if src.is_file():
            with src.open('rb') as inp,gzip.open(str(src)+'.gz','wb',compresslevel=4) as dst:
                shutil.copyfileobj(inp,dst)
            src.unlink()


def run_validation(args,catalog,detector,agent,identity,deadline):
    rates=[int(v) for v in args.val_rates.split(',')]
    results=[]
    root=args.output/'validation'
    root.mkdir(parents=True,exist_ok=True)
    def env_for(video,rate,name):
        return D3QNNativeEnv(catalog,detector,args.encoder,root/video/name,
                             split='val',frames=256,rate=rate,video=video,
                             threads=args.encoder_threads,
                             mode='anchor' if name=='qpa_anchor' else 'live')
    allowed=list(FIXED_CONTROLS)
    if args.phase!='probe':
        if agent is None:raise ValueError('Trained agent missing for validate')
        allowed.append(('trained_d3qn',None))
    names=[name for name,_ in allowed]
    chosen=names if args.val_methods=='all' else args.val_methods.split(',')
    if not chosen or len(set(chosen))!=len(chosen) or any(name not in names for name in chosen):
        raise ValueError('Invalid or duplicate validation methods for this phase')
    methods=[item for item in allowed if item[0] in chosen]
    for video in catalog.selected['val'][args.val_start:args.val_start+args.val_videos]:
        for rate in rates:
            for name,fixed in methods:
                if time.time()>=deadline:
                    atomic_json(args.output/'validation_partial.json',results)
                    print('VALIDATION_SOFT_STOP',len(results),flush=True)
                    return results,False
                env=env_for(video,rate,name)
                select=(lambda obs,fixed=fixed:canonical_fixed_action(fixed,obs)) if fixed is not None else (
                    lambda obs:agent.predict(obs,epsilon=0.))
                try:
                    meta,summary,stream,trajectory=run_episode(env,select,args.seed)
                    verify_decoded(stream,trajectory,256,args.ffmpeg,str(args.decoder))
                    result={'video':video,'rate_kbps':rate,'method':name,'summary':summary,
                            'decoder_parity':True,'bitstream_sha256':digest(stream),
                            'trajectory_sha256':digest(trajectory)}
                    results.append(result)
                    atomic_json(args.output/'validation_partial.json',results)
                    print('VALIDATION_POINT_PASS',video,rate,name,summary['actual_kbps'],
                          summary['mean_frame_f1'],flush=True)
                finally:
                    env.close()
                    if env.native is None and hasattr(env,'folder'):
                        compress_audit(env.folder/'codec')
    atomic_json(args.output/'validation.json',results)
    return results,True


def main():
    args=parser().parse_args()
    rates=[int(v) for v in args.val_rates.split(',')]
    if (args.train_steps<256 or args.train_steps%256 or not 1<=args.val_videos<=4
            or not 0<=args.val_start<=4 or args.val_start+args.val_videos>4
            or args.encoder_threads<1 or args.max_hours<0
            or not rates or len(set(rates))!=len(rates) or any(not 50<=r<=200 for r in rates)):
        raise ValueError('Require 256-multiple train steps, <=4 Val videos, 50..200 kbps rates')
    if args.phase=='probe' and args.checkpoint_in is not None:
        raise ValueError('Probe is fixed-action only; checkpoint is irrelevant')
    if args.phase=='validate' and args.checkpoint_in is None:
        raise ValueError('Validate needs --checkpoint-in from a completed D3QN episode')
    if args.output.exists():raise FileExistsError('Output directory must be new')
    args.output.mkdir(parents=True)
    session_start=args.session_start_utc if args.session_start_utc>0 else time.time()
    deadline=(session_start+args.max_hours*3600-900) if args.max_hours else float('inf')
    catalog=VideoCatalog(args.input_root,args.ffmpeg,args.ffprobe,mode='mini',seed=DATASET_SPLIT_SEED)
    catalog.check_available(('train','val'))
    identity={'protocol':PROTOCOL,'seed':args.seed,'data_mode':'mini',
              'dataset_split_seed':DATASET_SPLIT_SEED,
              'dataset_sha256':digest(ASSETS/'dataset_lock.json'),
              'source_fingerprint':source_fingerprint(),
              'encoder_sha256':digest(args.encoder),
              'detector_sha256':digest(args.weights),
              'frames':256,'qp_offsets':[-2,-1,0,1,2],'reward':'d3qn_v2_0.15_3.0_0.02'}
    contract={**identity,'phase':args.phase,'additional_steps':args.train_steps,
              'val_videos':args.val_videos,'val_start':args.val_start,
              'val_rates':rates,'val_methods':args.val_methods,
              'source_detector_before_action':True,'decoder_sha256':digest(args.decoder),
              'clock_budget_hours':args.max_hours,
              'epsilon_decay_steps':5120}
    atomic_json(args.output/'contract.json',contract)
    detector=FrozenDetector(args.yolo_repo,args.weights,args.device)
    status='not_started'
    try:
        if args.phase=='probe':
            results,finished=run_validation(args,catalog,detector,None,identity,deadline)
            status='probe_complete' if finished else 'probe_partial'
        else:
            if args.checkpoint_in:
                agent,next_seed=MaskedD3QN.load(args.checkpoint_in,identity=identity,device=args.device)
                print('D3QN_RESUME_PASS',agent.steps,agent.updates,'replay',len(agent.replay),flush=True)
            else:
                agent=MaskedD3QN(seed=args.seed,device=args.device)
                next_seed=args.seed
            if args.phase=='train':
                train=D3QNNativeEnv(catalog,detector,args.encoder,args.output/'training',
                                    split='train',frames=256,threads=args.encoder_threads,
                                    random_windows=True)
                logs=[];counts=np.zeros(N_ACTIONS,dtype=np.int64)
                completed_this_run=0
                checkpoint=args.output/'checkpoint.pt'
                try:
                    while completed_this_run<args.train_steps:
                        if time.time()>=deadline:
                            status='training_soft_stop'
                            print('TRAIN_SOFT_STOP',completed_this_run,'additional frames',flush=True)
                            break
                        train.video=balanced_train_video(catalog.selected['train'],args.seed,agent.steps//256)
                        obs,meta=train.reset(seed=next_seed)
                        total_reward=0.;last_update=None;episode_updates=[]
                        while True:
                            i=completed_this_run
                            epsilon=max(.05,1.-.95*min(1.,agent.steps/5120.))
                            action=agent.predict(obs,epsilon=epsilon)
                            nxt,reward,done,truncated,info=train.step(action)
                            if truncated:raise RuntimeError('Truncated native episode')
                            counts[action]+=1
                            update=agent.observe(obs,action,reward,nxt,done)
                            if update is not None:
                                last_update=update
                                episode_updates.append(update)
                            total_reward+=reward
                            completed_this_run+=1
                            if completed_this_run%32==0:
                                print('TRAIN_HEARTBEAT',agent.steps,'episode',meta['video_id'],
                                      'rate',meta['rate_kbps'],'epsilon',round(epsilon,4),flush=True)
                            if done:break
                            obs=nxt
                        next_seed=int(agent.rng.integers(1,2**31))
                        agent.save(checkpoint,identity=identity,next_episode_seed=next_seed)
                        averaged={k:(float(np.mean([v[k] for v in episode_updates])) if episode_updates else None)
                                  for k in ('loss','mean_abs_td','grad_norm','q_gap_valid',
                                            'q_max_valid','unique_greedy_actions_in_batch')}
                        row={'total_steps':agent.steps,'additional_steps':completed_this_run,
                             'episode':info['episode'],'episode_reward':total_reward,
                             'epsilon':epsilon,'train_video':meta['video_id'],
                             'train_rate':meta['rate_kbps'],
                             'training_video_index':(agent.steps//256-1)%16,
                             'training_epoch':(agent.steps//256-1)//16,
                             'last_update':last_update,
                             'mean_update':averaged,
                             'num_optimizer_updates_in_episode':len(episode_updates)}
                        logs.append(row)
                        atomic_json(args.output/'train_progress.json',
                                    {'status':'checkpointed','total_steps':agent.steps,
                                     'additional_steps':completed_this_run,'checkpoint':str(checkpoint),
                                     'last_episode':row,'action_counts':counts.tolist()})
                        atomic_json(args.output/'train_metrics.json',logs)
                        print('TRAIN_EPISODE_CHECKPOINT',agent.steps,meta['video_id'],
                              info['episode']['objective'],info['episode']['actual_kbps'],
                              flush=True)
                    if status!='training_soft_stop':
                        status='train_complete'
                finally:
                    train.close()
            else:
                results,finished=run_validation(args,catalog,detector,agent,identity,deadline)
                status='validation_complete' if finished else 'validation_partial'
        atomic_json(args.output/'result.json',{'status':status,'phase':args.phase,
                    'test_split_accessed':False,'trained_steps':agent.steps if args.phase!='probe' else 0})
        print('D3QN_RUN_FINISHED',status,flush=True)
    except BaseException as exc:
        atomic_json(args.output/'failure.json',{'error_type':type(exc).__name__,
                                               'error':str(exc),'status':status})
        raise
    finally:
        detector.close()


if __name__=='__main__':
    main()
