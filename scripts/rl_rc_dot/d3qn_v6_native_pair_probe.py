"""Native on-policy action-2/9 counterfactual probe for frozen V5 or V6.

NOT training. Two independent VVenC encodes share an entire prefix generated
by the specified frozen greedy policy. They intervene once and encode a full
continuation. Results are policy-conditional realized return evidence, NOT Q*.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import VideoCatalog, FrozenDetector, atomic_json, digest
from .d3qn_method import qp_action_map
from .d3qn_native_env_v3 import D3QNNativeEnvV3
from .d3qn_native_env_v5 import D3QNNativeEnvV5
from .d3qn_verify_decoded import verify_decoded
from .d3qn_v6_agent import StableActionConditionedDQN, pack as pack_v6, batch_obs as batch_v6
from .d3qn_v6_diagnostics import compare_matched_pair


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'encoder', 'decoder', 'yolo-repo', 'weights',
                 'checkpoint-in', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--policy', choices=('v5', 'v6'), default='v5')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--video-indices', default='0,1')
    p.add_argument('--positions', default='0,16,32,33,64,96')
    p.add_argument('--rate', type=int, default=120)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--allow-encoder-mismatch', action='store_true',
                   help='Only with --reproduction-audit for an independently verified Fixed2 Val4 match')
    p.add_argument('--reproduction-audit', type=Path,
                   help='JSON Gate-E audit stating PASS_FIXED2_VAL4_REFERENCE_REPRODUCED')
    return p


def obs_fingerprint(obs):
    h = hashlib.sha256()
    for k, dtype in (('global', np.float32), ('maps', np.float32), ('legal', np.bool_)):
        v = np.asarray(obs[k], dtype=dtype)
        h.update(k.encode('ascii'))
        h.update(str(v.shape).encode('ascii'))
        h.update(v.tobytes())
    return h.hexdigest()


def load_frozen_policy(checkpoint, mode, device):
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    identity = raw['identity']
    if mode == 'v5':
        from .d3qn_agent_v5 import StableMaskedD3QN
        if raw.get('version') != 'd3qn-native-checkpoint/v5-coding-cycle':
            raise ValueError('Expected V5 checkpoint, do not use V4/V6 weights')
        agent, _ = StableMaskedD3QN.load(checkpoint, identity=identity, device=device)
        from .d3qn_agent_v5 import pack_v5
        def get_q(obs):
            g, m, l = pack_v5(obs)
            with torch.no_grad():
                return agent.online({'global': torch.as_tensor(g[None], device=agent.device),
                    'maps': torch.as_tensor(m[None].astype(np.float32), device=agent.device),
                    'legal': torch.as_tensor(l[None], device=agent.device)})[0].cpu().numpy()
        return agent, identity, D3QNNativeEnvV5, get_q
    if not str(raw.get('version', '')).startswith('d3qn-native-checkpoint/v6'):
        raise ValueError('Expected V6 checkpoint')
    agent, _ = StableActionConditionedDQN.load(checkpoint, identity=identity, device=device)
    def get_q(obs):
        with torch.no_grad():
            return agent.online(batch_v6([pack_v6(obs)], agent.device))[0].cpu().numpy()
    return agent, identity, D3QNNativeEnvV3, get_q


def one_branch(args, catalog, detector, agent, env_type, q_fn, video, pos, probe_action, dirname):
    env = env_type(catalog, detector, args.encoder, dirname, split='train', frames=256,
                   rate=args.rate, video=video, threads=args.threads,
                   mode='live', random_windows=False, retain_media=False)
    try:
        obs, meta = env.reset(seed=args.seed)
        probe_hash, qgap, two_legal, map_difference = None, None, None, None
        while True:
            coding_step = env.completed
            if coding_step == pos:
                legal = np.asarray(obs['legal'], dtype=bool)
                two_legal = bool(legal[2] and legal[9])
                map_difference = not np.array_equal(qp_action_map(2, env.importance),
                                                    qp_action_map(9, env.importance))
                if not two_legal or not map_difference:
                    raise ValueError(f'Action2/9 not distinct and legal at {video}:{pos}')
                probe_hash = obs_fingerprint(obs)
                q = q_fn(obs)
                qgap = float(q[2] - q[9])
                action = probe_action
            else:
                action = agent.predict(obs, epsilon=0.)
            obs, reward, done, truncated, info = env.step(int(action))
            if truncated:
                raise RuntimeError('Native episode truncated')
            if done:
                break
        stream = env.native.output / 'stream.266'
        trace = env.folder / 'trajectory.jsonl.gz'
        verify_decoded(stream, trace, 256, args.ffmpeg, str(args.decoder))
        with gzip.open(trace, 'rt', encoding='utf-8') as inp:
            frames = [json.loads(s) for s in inp if s.strip()]
        if len(frames) != 256:
            raise AssertionError('Native frame trace incomplete')
        return {'video': video, 'rate_kbps': args.rate, 'probe_step': pos,
                'probe_action': probe_action, 'q2_minus_q9': qgap,
                'pre_action_obs_hash': probe_hash,
                'two_actions_legal': two_legal, 'effective_maps_differ': map_difference,
                'objective': float(info['episode']['objective']),
                'mean_f1': float(info['episode']['mean_frame_f1']),
                'actual_kbps': float(info['episode']['actual_kbps']),
                'frames': frames, 'decoder_parity': True,
                'bitstream_sha256': digest(stream),
                'source_start_frame': meta['start_frame']}
    finally:
        env.close()


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError('Use a unique output directory')
    args.output.mkdir(parents=True)
    if args.rate < 50 or args.rate > 200:
        raise ValueError('Rate must be 50..200 kbps')
    agent, identity, env_type, q_fn = load_frozen_policy(args.checkpoint_in, args.policy, args.device)
    if digest(args.weights) != identity['detector_sha256']:
        raise ValueError('Detector weights do not match frozen checkpoint')
    same_encoder = digest(args.encoder) == identity['encoder_sha256']
    if not same_encoder:
        if not args.allow_encoder_mismatch or args.reproduction_audit is None:
            raise ValueError('Encoder hash mismatch requires a verified Fixed2 Val4 reproduction audit')
        native_audit = json.loads(args.reproduction_audit.read_text(encoding='utf-8'))
        if native_audit.get('status') != 'PASS_FIXED2_VAL4_REFERENCE_REPRODUCED':
            raise ValueError('Fixed2 Val4 reproduction audit did not PASS')
    catalog = VideoCatalog(args.input_root, args.ffmpeg, args.ffprobe, mode='mini', seed=20261111)
    catalog.check_available(('train',))
    indices = [int(x) for x in args.video_indices.split(',')]
    positions = [int(x) for x in args.positions.split(',')]
    if (not indices or len(set(indices)) != len(indices) or
            not positions or len(set(positions)) != len(positions) or
            any(not 0 <= x < 16 for x in indices) or
            any(not 0 <= x < 256 for x in positions)):
        raise ValueError('Invalid Train16 video/step selections')
    atomic_json(args.output / 'manifest.json', {'policy': args.policy, 'frozen_checkpoint': str(args.checkpoint_in),
        'frozen_identity': identity, 'encoder_sha_match': same_encoder,
        'encoder_mismatch_limitation': not same_encoder,
        'conditional_return_not_optimal_q': True,
        'test30_accessed': False, 'probe_positions': positions,
        'train_video_indices': indices})
    detector = FrozenDetector(args.yolo_repo, args.weights, args.device)
    results = []
    try:
        for vid_idx in indices:
            video = catalog.selected['train'][vid_idx]
            for pos in positions:
                pair = []
                for a in (2, 9):
                    print('NATIVE_ONPOLICY_START', video, pos, a, flush=True)
                    row = one_branch(args, catalog, detector, agent, env_type, q_fn,
                                     video, pos, a, args.output / 'native' / f'{vid_idx}_{pos}_a{a}')
                    pair.append(row)
                    atomic_json(args.output / f'native_{vid_idx}_{pos}_a{a}.json', row)
                if pair[0]['pre_action_obs_hash'] != pair[1]['pre_action_obs_hash']:
                    raise ValueError('Matched-prefix observation mismatch: invalid causal comparison')
                for row in pair:
                    row['same_pre_action_state'] = True
                summary = compare_matched_pair(pair[0], pair[1], action_step=pos)
                results.append(summary)
                atomic_json(args.output / 'onpolicy_pair_results.json', results)
                print('NATIVE_ONPOLICY_PAIR_PASS', video, pos, 'deltaJ',
                      round(summary['delta_J_2_minus_9'], 7),
                      'conflict', summary['ranking_conflict'], flush=True)
    finally:
        detector.close()
    atomic_json(args.output / 'summary.json', {'n_pairs': len(results),
         'n_conflicts': sum(x['ranking_conflict'] for x in results),
         'n_suffix_action_differences': sum(x['suffix_action_disagreements'] for x in results),
         'result_kind': 'frozen-on-policy-native-intervention-not-training',
         'encoder_sha_match': same_encoder})


if __name__ == '__main__':
    main()
