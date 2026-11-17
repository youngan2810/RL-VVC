"""Isolated native VVenC QP experiment: fixed-action probe, then D3QN pilot.

No PPO state/reward/action mapper; no Test split or published performance claim.
Run preflight and check codec/YOLO/VVdeC availability before using this runner.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np

from .data import VideoCatalog,FrozenDetector,atomic_json,digest,ASSETS,source_fingerprint
from .d3qn_native_env import D3QNNativeEnv
from .d3qn_agent_v2 import MaskedD3QN
from .d3qn_method import NEUTRAL_ACTION, N_ACTIONS, qp_action_map
from .evaluate import verify_decoded


def parser():
    p=argparse.ArgumentParser(description='Standalone masked Dueling Double DQN for VVC QP')
    p.add_argument('--phase',choices=('probe','train'),required=True)
    for x in ('input-root','encoder','decoder','yolo-repo','weights','output'):
        p.add_argument('--'+x,required=True,type=Path)
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--ffprobe',default='ffprobe')
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--train-steps',type=int,default=5120)
    p.add_argument('--frames',type=int,default=256)
    p.add_argument('--encoder-threads',type=int,default=2)
    p.add_argument('--val-videos',type=int,default=4)
    p.add_argument('--val-rates',default='50,120,200')
    return p


def canonical_fixed_action(action, obs):
    """Map a fixed policy to its legal equivalent when ROI is empty/full.

    Action masking removes duplicate QP maps, not the physical fixed policy.
    Without canonicalization a fixed ROI strategy would crash on no-car POCs.
    """
    if not isinstance(action,(int,np.integer)) or not 0<=int(action)<N_ACTIONS:
        raise ValueError('Invalid fixed action')
    if obs['legal'][action]:
        return int(action)
    wanted=qp_action_map(action,obs['maps'][4])
    alternatives=[int(a) for a in np.flatnonzero(obs['legal'])
                  if np.array_equal(qp_action_map(int(a),obs['maps'][4]),wanted)]
    if len(alternatives)!=1:
        raise ValueError('Masked fixed control has no unique legal QP-map equivalent')
    return alternatives[0]


def episode(env,act,seed):
    obs,metadata=env.reset(seed=seed)
    while True:
        chosen=int(act(obs))
        obs,reward,done,truncated,info=env.step(chosen)
        if truncated:raise RuntimeError('Unexpected native episode truncation')
        if done:break
    stream=env.native.output/'stream.266'
    trajectory=env.folder/'trajectory.jsonl.gz'
    return metadata,info['episode'],stream,trajectory


def main():
    args=parser().parse_args()
    rates=[int(v) for v in args.val_rates.split(',')]
    if (args.frames!=256 or args.train_steps<1 or args.val_videos<1 or
            not rates or len(set(rates))!=len(rates) or any(r<50 or r>200 for r in rates)):
        raise ValueError('Mini mode requires 256 frames and validation rates within [50,200]')
    if args.output.exists():
        raise FileExistsError('New unique output folder required for each experiment')
    args.output.mkdir(parents=True)
    catalog=VideoCatalog(args.input_root,args.ffmpeg,args.ffprobe,mode='mini',seed=args.seed)
    if args.val_videos>len(catalog.selected['val']):
        raise ValueError('Requested more validation videos than Mini Val4 contains')
    catalog.check_available(('train','val'))
    contract={
        'schema':'independent-native-masked-d3qn-qp/v1',
        'phase':args.phase,'frames':args.frames,'seed':args.seed,
        'train_steps':args.train_steps,'val_videos':args.val_videos,'val_rates':rates,
        'algorithm':'Dueling Double DQN, 25 legal ROI/BG offset pairs, replay, Double-Q target',
        'source_fingerprint':source_fingerprint(),
        'dataset_lock_sha256':digest(ASSETS/'dataset_lock.json'),
        'encoder_sha256':digest(args.encoder),'decoder_sha256':digest(args.decoder),
        'detector_weights_sha256':digest(args.weights),
        'source_detector_required_at_encoder':True,
        'interpretation':'Probe does not demonstrate bitrate-matched improvement. Test untouched.'
    }
    atomic_json(args.output/'contract.json',contract)
    detector=FrozenDetector(args.yolo_repo,args.weights,args.device)
    agent=MaskedD3QN(seed=args.seed,device=args.device)

    def env_for(split,path,*,video=None,rate=None,mode='live'):
        return D3QNNativeEnv(catalog,detector,args.encoder,path,split=split,
                             frames=256,rate=rate,video=video,mode=mode,
                             threads=args.encoder_threads,random_windows=(split=='train'))

    try:
        if args.phase=='train':
            train=env_for('train',args.output/'training')
            obs,meta=train.reset(seed=args.seed)
            logs=[];counter=np.zeros(N_ACTIONS,dtype=int)
            t0=time.perf_counter()
            try:
                for i in range(args.train_steps):
                    epsilon=max(.05,1.-.95*i/max(1,args.train_steps-1))
                    action=agent.predict(obs,epsilon=epsilon)
                    counter[action]+=1
                    nxt,r,done,truncated,info=train.step(action)
                    if truncated:raise RuntimeError('Unexpected native truncation')
                    update=agent.observe(obs,action,r,nxt,done)
                    if update is not None and (i+1)%256==0:
                        logs.append({'step':i+1,'epsilon':epsilon,**update})
                    if done:
                        logs.append({'step':i+1,'episode':info['episode']})
                        obs,meta=train.reset(seed=int(agent.rng.integers(1,2**31)))
                    else:obs=nxt
                agent.save(args.output/'checkpoint.pt')
                atomic_json(args.output/'train_metrics.json',logs)
                atomic_json(args.output/'train_summary.json',
                            {'interactions':agent.steps,'optimizer_updates':agent.updates,
                             'actions':counter.tolist(),'elapsed_seconds':time.perf_counter()-t0})
            finally:train.close()
        # The fixed controls are intentionally compact; expand rates/videos
        # and include full 25 fixed actions for rigorous headroom coverage.
        representative=[
            ('qpa_anchor',NEUTRAL_ACTION),
            ('global_minus1',6),  # roi=-1, background=-1
            ('global_plus1',18),  # roi=+1, background=+1
            ('roi_minus1_bg_plus1',8),
            ('roi_minus2_bg_plus2',4),
            ('roi_zero_bg_plus2',14)]
        results=[]
        videos=catalog.selected['val'][:args.val_videos]
        for video in videos:
            for rate in rates:
                entries=representative+([('trained_d3qn',None)] if args.phase=='train' else [])
                for name,constant in entries:
                    case=env_for('val',args.output/'validation'/video/name,
                                 video=video,rate=rate,
                                 mode='anchor' if name=='qpa_anchor' else 'live')
                    selector=(lambda obs,constant=constant:canonical_fixed_action(constant,obs)) if constant is not None else (
                        lambda obs:agent.predict(obs,epsilon=0.))
                    try:
                        meta,metrics,stream,trajectory=episode(case,selector,args.seed)
                        verify_decoded(stream,trajectory,256,str(args.ffmpeg),str(args.decoder))
                        results.append({'video':video,'rate_kbps':rate,'method':name,
                                        'summary':metrics,'vvdec_parity':True,
                                        'bitstream_sha256':digest(stream)})
                        atomic_json(args.output/'validation_partial.json',results)
                    finally:case.close()
        atomic_json(args.output/'validation.json',results)
        print(json.dumps({'result':'D3QN_NATIVE_EXPERIMENT_FINISHED',
                          'phase':args.phase,'trained_steps':agent.steps,
                          'validation_points':len(results),
                          'warning':'Same-target bitrate is not a matched-rate claim; no Test split used'},indent=2))
    finally:detector.close()


if __name__=='__main__':
    main()
