"""Isolated mini-dataset D3QN pilot. Never touches Test30/Test100.

Run after native preflight. Probe first and retain all native audit evidence;
comparison at the same *target* rate does not establish rate-matched gains.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .data import VideoCatalog, FrozenDetector, atomic_json, digest, ASSETS, source_fingerprint
from .environment import VVenCEnv
from .semantic_wrapper import SemanticQPWrapper
from .d3qn_qp import D3QN
from .evaluate import verify_decoded


def parser():
    p = argparse.ArgumentParser(description='Isolated semantic D3QN proof-of-concept; PPO is unchanged')
    p.add_argument('--phase', choices=('probe', 'train'), required=True)
    for name in ('input-root', 'encoder', 'decoder', 'yolo-repo', 'weights', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--device', default='cpu')
    p.add_argument('--target-kbps', type=int, default=120)
    p.add_argument('--train-steps', type=int, default=5120)
    p.add_argument('--val-videos', type=int, default=1)
    p.add_argument('--encoder-threads', type=int, default=2)
    return p


def run_episode(env, select, seed):
    obs, metadata = env.reset(seed=seed)
    last_info = None
    actions = np.zeros(5, dtype=int)
    while True:
        a = int(select(obs))
        actions[a] += 1
        obs, _, done, truncated, last_info = env.step(a)
        if truncated:
            raise RuntimeError('Unexpected truncated native VVC episode')
        if done:
            break
    native = env.unwrapped
    stream = native.native.output / 'stream.266'
    trajectory = native.native.output.parent / 'trajectory.jsonl.gz'
    if native.trajectory is not None:
        native.trajectory.close()
        native.trajectory = None
    # Close compresses the native trace, allowing independent VVdeC audit.
    native._close_episode()
    return metadata, last_info['episode'], actions, stream, trajectory


def main():
    args = parser().parse_args()
    if args.train_steps < 1 or args.val_videos < 1 or not 1 <= args.target_kbps <= 200:
        raise ValueError('Invalid pilot bounds')
    if args.output.exists():
        raise FileExistsError('Output must be a new empty directory to preserve provenance')
    args.output.mkdir(parents=True)
    dataset = VideoCatalog(args.input_root, args.ffmpeg, args.ffprobe, mode='mini', seed=args.seed)
    dataset.check_available(('train', 'val'))
    lock = {'schema':'d3qn-semantic-qp-pilot/v1','phase':args.phase,'seed':args.seed,
            'input':str(args.input_root),'encoder_sha256':digest(args.encoder),
            'decoder_sha256':digest(args.decoder),'weights_sha256':digest(args.weights),
            'source_fingerprint':source_fingerprint(),
            'dataset_lock_sha256':digest(ASSETS/'dataset_lock.json'),
            'notes':'One source-YOLO semantic prior per encoded frame; inference cost is not free. '
                    'Same-target-rate probes are NOT rate-matched tests.'}
    atomic_json(args.output/'contract.json',lock)
    detector = FrozenDetector(args.yolo_repo,args.weights,args.device)

    def make_env(split, folder, *, video=None, rate=None):
        base = VVenCEnv(dataset,detector,args.encoder,folder,split=split,frame_limit=256,
                        threads=args.encoder_threads,fixed_video=video,fixed_rate=rate,
                        random_windows=(split=='train'),agent_profile='qpdev1-qbins')
        return SemanticQPWrapper(base)

    try:
        agent = D3QN(seed=args.seed, device=args.device)
        if args.phase == 'train':
            env = make_env('train', args.output/'training')
            obs, _ = env.reset(seed=args.seed)
            rng = np.random.default_rng(args.seed)
            results = []
            for step in range(args.train_steps):
                # Broad exploration over costly native actions; no claims from 5k steps.
                epsilon = 0.1 + 0.9*max(0.,1-step/max(1,args.train_steps))
                act = agent.predict(obs,epsilon=epsilon)
                nxt,reward,done,truncated,info = env.step(act)
                if truncated:
                    raise RuntimeError('Native episode truncated')
                metrics = agent.observe(obs,act,reward,nxt,done)
                if metrics is not None and (step+1)%256 == 0:
                    results.append({'step':step+1,'epsilon':epsilon,**metrics})
                if done:
                    results.append({'step':step+1,'episode':info['episode']})
                    obs, _ = env.reset(seed=int(rng.integers(1,2**31)))
                else:
                    obs = nxt
            env.unwrapped._close_episode()
            agent.save(args.output/'d3qn_checkpoint.pt')
            atomic_json(args.output/'train_metrics.json',results)
        rows=[]
        for video in dataset.selected['val'][:args.val_videos]:
            for strategy, selection in [('qpa_anchor',lambda _:0),
                                        ('roi_mild',lambda _:1),
                                        ('roi_strong',lambda _:2),
                                        ('roi_protect',lambda _:3),
                                        ('roi_protect_bg',lambda _:4)]+([('d3qn',lambda obs:agent.predict(obs,epsilon=0.0))] if args.phase=='train' else []):
                folder=args.output/'validation'/video/strategy
                env=make_env('val',folder,video=video,rate=args.target_kbps)
                if strategy=='qpa_anchor':
                    # No-op template is byte-identical to original VVenC+QPA anchor.
                    env.unwrapped.mode='anchor'
                metadata,summary,action_hist,stream,trajectory=run_episode(env,selection,args.seed)
                verify_decoded(stream,trajectory,256,str(args.ffmpeg),str(args.decoder))
                rows.append({'video':video,'target_kbps':args.target_kbps,
                             'strategy':strategy,'summary':summary,
                             'action_histogram':action_hist.tolist(),
                             'decoder_verified':True,'stream_sha256':digest(stream)})
                atomic_json(args.output/'validation.json',rows)
        print(json.dumps({'status':'D3QN_EXPERIMENT_COLLECTED',
                          'phase':args.phase,'validation_points':len(rows),
                          'trained_steps':agent.steps,'warning':'Not rate matched; no Test data touched'},indent=2))
    finally:
        detector.close()


if __name__ == '__main__':
    main()
