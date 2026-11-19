"""D3QN V6 native VVC Train16/Val4 runner; no Test30 is accessible.

Reuses the already audited NativeEpisode, V3 25-D causal observation contract,
source-YOLO metric, locked VideoCatalog and VVdeC parity verifier. New learning
and action-conditioned network are in d3qn_v6_agent.py.
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
from .d3qn_native_env_v3 import D3QNNativeEnvV3
from .d3qn_method import NEUTRAL_ACTION, N_ACTIONS, qp_action_map, balanced_train_video
from .d3qn_verify_decoded import verify_decoded
from .d3qn_v6_agent import StableActionConditionedDQN, exploration_epsilon

PROTOCOL = 'native-masked-d3qn-qp/v6-action-conditioned'
TRAIN_RATES = (50, 80, 120, 160, 200)
SPLIT_SEED = 20261111
FIXED_CONTROLS = (
    ('qpa_anchor', NEUTRAL_ACTION),
    ('global_minus2', 0),
    ('roi_minus2_bg_zero', 2),
    ('roi_minus1_bg_plus2', 9),
    ('roi_plus1_bg_plus2', 19),
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('probe', 'train', 'validate'), required=True)
    for name in ('input-root', 'encoder', 'decoder', 'yolo-repo', 'weights', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--checkpoint-in', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--train-steps', type=int, default=1024)
    p.add_argument('--n-step', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--target-tau', type=float, default=.005)
    p.add_argument('--action-mix', type=float, default=.45)
    p.add_argument('--boundary-mix', type=float, default=.10)
    p.add_argument('--epsilon-floor', type=float, default=.20)
    p.add_argument('--epsilon-decay-steps', type=int, default=10240)
    p.add_argument('--encoder-threads', type=int, default=2)
    p.add_argument('--val-videos', type=int, default=4)
    p.add_argument('--val-start', type=int, default=0)
    p.add_argument('--val-rates', default='120')
    p.add_argument('--val-methods', default='all')
    p.add_argument('--max-hours', type=float, default=0.)
    p.add_argument('--session-start-utc', type=float, default=0.)
    return p


def canonical_fixed_action(action, obs):
    """Deduplicate legal representation for the same physical QP map."""
    if obs['legal'][action]:
        return int(action)
    wanted = qp_action_map(action, obs['maps'][4])
    matches = [int(i) for i in np.flatnonzero(obs['legal'])
               if np.array_equal(qp_action_map(int(i), obs['maps'][4]), wanted)]
    if len(matches) != 1:
        raise ValueError('Fixed action lacks a unique legal equivalent')
    return matches[0]


def run_episode(env, choose, seed):
    obs, meta = env.reset(seed=seed)
    while True:
        action = int(choose(obs))
        obs, reward, done, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError('Unexpected truncated native video')
        if done:
            return meta, info['episode'], env.native.output / 'stream.266', env.folder / 'trajectory.jsonl.gz'


def compress_native_audits(directory):
    """Atomically archive completed native audit streams; retain originals on failure."""
    directory = Path(directory)
    for name in ('events.jsonl', 'cus.jsonl', 'qgs.jsonl'):
        source = directory / name
        if not source.is_file():
            continue
        dest = directory / (name + '.gz')
        tmp = directory / (name + '.gz.tmp')
        try:
            with source.open('rb') as inp, tmp.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=4) as out:
                    shutil.copyfileobj(inp, out)
            tmp.replace(dest)
            source.unlink()
        finally:
            tmp.unlink(missing_ok=True)


def close_and_compress_native(env):
    """The environment clears env.native on close: capture its path FIRST."""
    native_dir = getattr(getattr(env, 'native', None), 'output', None)
    try:
        env.close()
    finally:
        if native_dir is not None and Path(native_dir).is_dir():
            compress_native_audits(native_dir)


def experiment_identity(args):
    return {'protocol': PROTOCOL, 'seed': args.seed, 'data_mode': 'mini',
            'dataset_split_seed': SPLIT_SEED,
            'dataset_sha256': digest(ASSETS / 'dataset_lock.json'),
            'source_fingerprint': source_fingerprint(),
            'encoder_sha256': digest(args.encoder),
            'detector_sha256': digest(args.weights),
            'frames': 256, 'qp_offsets': list(range(-2, 3)),
            'state': 'v3-causal-global25-plus-five-spatial-maps',
            'reward': 'd3qn_v2_0.15_3.0_0.02_unchanged',
            'learning': {'architecture': 'physical-action-conditioned-v6',
                         'n_step': args.n_step, 'target_tau': args.target_tau,
                         'learning_rate': args.learning_rate,
                         'action_mix': args.action_mix,
                         'boundary_mix': args.boundary_mix,
                         'epsilon_floor': args.epsilon_floor,
                         'epsilon_decay_steps': args.epsilon_decay_steps,
                         'gamma': 1.0}}


def main():
    args = parser().parse_args()
    if (args.train_steps < 256 or args.train_steps % 256 or
            not 0 <= args.val_start < 4 or not 1 <= args.val_videos <= 4 or
            args.val_start + args.val_videos > 4 or args.encoder_threads < 1 or
            args.max_hours < 0 or not 1 <= args.n_step <= 32 or
            not 0 < args.target_tau <= 1 or args.learning_rate <= 0 or
            args.action_mix < 0 or args.boundary_mix < 0 or
            args.action_mix + args.boundary_mix >= 1):
        raise ValueError('Invalid V6 runner configuration')
    rates = [int(x) for x in args.val_rates.split(',')]
    if not rates or len(set(rates)) != len(rates) or any(not 50 <= x <= 200 for x in rates):
        raise ValueError('Validation rates must lie in 50..200')
    if args.phase == 'validate' and args.checkpoint_in is None:
        raise ValueError('Validation requires V6 checkpoint')
    if args.phase == 'probe' and args.checkpoint_in is not None:
        raise ValueError('Fixed baseline probe does not load a checkpoint')
    if args.output.exists():
        raise FileExistsError('Each invocation must use a new output directory')
    args.output.mkdir(parents=True)
    deadline = float('inf') if args.max_hours == 0 else (
        (args.session_start_utc or time.time()) + args.max_hours * 3600 - 900)
    identity = experiment_identity(args)
    atomic_json(args.output / 'contract.json', {
        **identity, 'phase': args.phase, 'additional_steps': args.train_steps,
        'val_rates': rates, 'val_start': args.val_start,
        'val_videos': args.val_videos, 'val_methods': args.val_methods,
        'decoder_sha256': digest(args.decoder)})
    catalog = VideoCatalog(args.input_root, args.ffmpeg, args.ffprobe, mode='mini', seed=SPLIT_SEED)
    catalog.check_available(('train', 'val'))
    detector = FrozenDetector(args.yolo_repo, args.weights, args.device)
    try:
        agent = None
        next_seed = args.seed
        if args.phase != 'probe':
            if args.checkpoint_in:
                agent, next_seed = StableActionConditionedDQN.load(
                    args.checkpoint_in, identity=identity, device=args.device)
                print('V6_RESUME_PASS', agent.steps, agent.updates, len(agent.replay), flush=True)
            else:
                agent = StableActionConditionedDQN(
                    seed=args.seed, device=args.device, n_step=args.n_step,
                    target_tau=args.target_tau, lr=args.learning_rate,
                    action_mix=args.action_mix, boundary_mix=args.boundary_mix)
        if args.phase == 'train':
            train_env = D3QNNativeEnvV3(catalog, detector, args.encoder,
                                        args.output / 'training', split='train', frames=256,
                                        threads=args.encoder_threads, random_windows=True)
            results = []
            try:
                for _ in range(args.train_steps // 256):
                    if time.time() >= deadline:
                        atomic_json(args.output / 'training_partial.json', results)
                        print('V6_TRAIN_SOFT_STOP', agent.steps, flush=True)
                        break
                    train_env.video = balanced_train_video(
                        catalog.selected['train'], args.seed, agent.steps // 256)
                    train_env.rate = TRAIN_RATES[(agent.steps // 256) % 5]
                    obs, meta = train_env.reset(seed=next_seed)
                    updates = []
                    while True:
                        epsilon = exploration_epsilon(agent.steps,
                            floor=args.epsilon_floor, decay_steps=args.epsilon_decay_steps)
                        action = agent.predict(obs, epsilon=epsilon)
                        nxt, reward, done, truncated, info = train_env.step(action)
                        if truncated:
                            raise RuntimeError('Native training truncated')
                        upd = agent.observe(obs, action, reward, nxt, done)
                        if upd is not None:
                            updates.append(upd)
                        if agent.steps % 32 == 0:
                            print('V6_TRAIN_HEARTBEAT', agent.steps, meta['video_id'],
                                  meta['rate_kbps'], 'epsilon', round(epsilon, 3), flush=True)
                        if done:
                            break
                        obs = nxt
                    next_seed = int(agent.rng.integers(1, 2**31))
                    agent.save(args.output / 'checkpoint.pt', identity=identity,
                               next_episode_seed=next_seed)
                    row = {'total_steps': agent.steps, 'train_video': meta['video_id'],
                           'train_rate': meta['rate_kbps'], 'start_frame': meta['start_frame'],
                           'epsilon': epsilon, 'episode': info['episode'],
                           'updates': len(updates),
                           'mean_update': ({key: float(np.mean([u[key] for u in updates]))
                                           for key in updates[0] if key not in ('updates',)}
                                          if updates else {}),
                           'action_rate_counts': agent.coverage_counts.tolist()}
                    results.append(row)
                    atomic_json(args.output / 'training.json', results)
                    print('V6_TRAIN_EPISODE_PASS', agent.steps, row['episode']['objective'], flush=True)
            finally:
                train_env.close()
        else:
            allowed = list(FIXED_CONTROLS)
            if args.phase == 'validate':
                allowed.append(('trained_d3qn_v6', None))
            names = {name for name, _ in allowed}
            requested = names if args.val_methods == 'all' else set(args.val_methods.split(','))
            if not requested or not requested.issubset(names):
                raise ValueError('Invalid requested validation methods')
            results = []
            try:
                for vid in catalog.selected['val'][args.val_start:args.val_start+args.val_videos]:
                    for rate in rates:
                        for name, fixed in allowed:
                            if name not in requested:
                                continue
                            if time.time() >= deadline:
                                atomic_json(args.output / 'validation_partial.json', results)
                                print('V6_VALIDATION_SOFT_STOP', len(results), flush=True)
                                return
                            mode = 'anchor' if name == 'qpa_anchor' else 'live'
                            env = D3QNNativeEnvV3(catalog, detector, args.encoder,
                                args.output / 'validation' / vid / name,
                                split='val', frames=256, rate=rate, video=vid,
                                threads=args.encoder_threads, mode=mode)
                            try:
                                choose = (lambda obs, a=fixed: canonical_fixed_action(a, obs)) \
                                    if fixed is not None else (lambda obs: agent.predict(obs, epsilon=0.))
                                meta, summary, stream, trace = run_episode(env, choose, args.seed)
                                verify_decoded(stream, trace, 256, args.ffmpeg, str(args.decoder))
                                results.append({'video': vid, 'rate_kbps': rate, 'method': name,
                                    'summary': summary, 'decoder_parity': True,
                                    'bitstream_sha256': digest(stream),
                                    'trajectory_sha256': digest(trace)})
                                atomic_json(args.output / 'validation_partial.json', results)
                                print('V6_NATIVE_VALIDATION_PASS', vid, rate, name,
                                      summary['objective'], flush=True)
                            finally:
                                close_and_compress_native(env)
            finally:
                atomic_json(args.output / 'validation.json', results)
    finally:
        detector.close()


if __name__ == '__main__':
    main()
