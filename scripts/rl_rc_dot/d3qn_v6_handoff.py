"""V6 handoff provenance checks. No training or codec work occurs here."""
from __future__ import annotations

from pathlib import Path
import hashlib

import torch

CHECKPOINT_VERSION = 'd3qn-native-checkpoint/v6-action-conditioned'
MANIFEST_SCHEMA = 'd3qn-v6-action-conditioned-mini16-val4/2'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def verify_checkpoint(path, expected_identity, *, expected_steps=None):
    """Reject different V6 protocol, source, weights, dataset or encoder bytes."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Incompatible V6 checkpoint version; V5/V4 checkpoints cannot resume V6')
    actual = payload.get('identity')
    if actual != expected_identity:
        different = sorted(set(actual or {}) | set(expected_identity))
        different = [key for key in different if (actual or {}).get(key) != expected_identity.get(key)]
        raise ValueError('V6 checkpoint provenance mismatch in ' + ', '.join(different) +
                         '; if encoder_sha256 differs, reuse the ORIGINAL encoder binary, not a rebuilt one')
    steps = int(payload['steps'])
    if steps < 256 or steps % 256:
        raise ValueError('Checkpoint not saved at completed 256-picture episode boundary')
    if expected_steps is not None and steps != expected_steps:
        raise ValueError(f'Checkpoint steps {steps} differ from expected {expected_steps}')
    config = payload['config']
    settings = expected_identity['learning']
    for setting in ('n_step', 'target_tau', 'learning_rate', 'action_mix',
                    'boundary_mix', 'epsilon_floor', 'epsilon_decay_steps'):
        key = 'lr' if setting == 'learning_rate' else setting
        if setting in ('epsilon_floor', 'epsilon_decay_steps'):
            continue  # recorded in identity, not model config
        if config.get(key) != settings[setting]:
            raise ValueError('V6 checkpoint learning configuration mismatch: ' + setting)
    for required in ('online', 'target', 'optimizer', 'replay', 'coverage_counts',
                     'agent_rng_state', 'torch_cpu_rng_state', 'next_episode_seed'):
        if required not in payload:
            raise ValueError('Incomplete V6 checkpoint: ' + required)
    return {'steps': steps, 'updates': int(payload['updates']),
            'encoder_sha256': actual['encoder_sha256']}


def verify_manifest(manifest, source_commit, identity, *, completed_episodes=None):
    if manifest.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('Unsupported or old V6 SAFE manifest schema; cannot safely resume')
    if (manifest.get('source_commit') != source_commit or
            manifest.get('protocol') != identity['protocol'] or
            manifest.get('seed') != identity['seed'] or
            manifest.get('test_used') is not False or
            manifest.get('identity') != identity):
        raise ValueError('V6 SAFE source/protocol/dataset/toolchain/learning identity mismatch')
    episodes = int(manifest['completed_episodes'])
    if manifest['train_steps'] != episodes * 256:
        raise ValueError('SAFE episode count and training steps inconsistent')
    if completed_episodes is not None and episodes != completed_episodes:
        raise ValueError('SAFE manifest and training history disagree')
    return True


def verify_checkpoint_digests(root, recorded):
    """Detect corrupted or stale checkpoint files before loading any state."""
    for name, expected in recorded.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not str(rel).startswith('checkpoints/'):
            raise ValueError('Unsafe checkpoint manifest path')
        file = Path(root) / rel
        if not file.is_file() or sha256(file) != expected:
            raise ValueError('SAFE checkpoint hash mismatch: ' + name)
    return True


def preflight_matches(report, contract, *, encoder_sha256, decoder_sha256):
    return bool(
        report.get('gate') == 'REAL_VVC_STATE_ACTION_AU_DECODE_PASS' and
        report.get('encoder_sha256') == encoder_sha256 and
        contract.get('encoder_sha256') == encoder_sha256 and
        contract.get('decoder_sha256') == decoder_sha256
    )
