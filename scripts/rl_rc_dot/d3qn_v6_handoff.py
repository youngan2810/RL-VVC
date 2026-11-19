"""V6 handoff provenance checks. No training or codec work occurs here."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import zipfile

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


def read_safe_manifest(source):
    """Find exactly one V6 SAFE manifest in an archive or extracted directory."""
    source = Path(source)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            members = [name for name in archive.namelist() if not name.endswith('/')]
            if len(members) != len(set(members)):
                raise ValueError('Duplicate SAFE ZIP entries')
            for name in members:
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or chr(92) in name:
                    raise ValueError('Unsafe SAFE member path')
            roots = [name[:-len('learning_manifest.json')] for name in members
                     if name.endswith('learning_manifest.json')]
            if len(roots) != 1:
                raise ValueError('Expected exactly one V6 manifest root')
            manifest = json.loads(archive.read(roots[0]+'learning_manifest.json'))
            return manifest, roots[0]
    if source.is_dir():
        candidates = ([source/'learning_manifest.json'] if
                      (source/'learning_manifest.json').is_file()
                      else list(source.rglob('learning_manifest.json')))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one extracted V6 manifest')
        return json.loads(candidates[0].read_text(encoding='utf-8')), candidates[0].parent
    raise FileNotFoundError('V6 SAFE artifact not found: '+str(source))


def restore_encoder_from_safe(source, expected_sha256, destination):
    """Recover the EXACT original native encoder, verifying bytes before executable use.

    This does not relax checkpoint identity: the recovered binary must match the
    recorded checkpoint/manifest SHA. The Kaggle native preflight is still required.
    """
    source = Path(source)
    manifest, root = read_safe_manifest(source)
    recorded = manifest.get('identity', {}).get('encoder_sha256')
    if manifest.get('schema') != MANIFEST_SCHEMA or recorded != expected_sha256:
        raise ValueError('SAFE encoder identity mismatch')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name+'.tmp')
    try:
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                member = str(root) + 'TOOLCHAIN/vvencapp'
                if member not in archive.namelist():
                    raise FileNotFoundError('SAFE ZIP has no bundled original VVenC executable')
                with archive.open(member, 'r') as inp, staging.open('wb') as dst:
                    shutil.copyfileobj(inp, dst)
        else:
            filename = Path(root) / 'TOOLCHAIN/vvencapp'
            if not filename.is_file() or not filename.resolve().is_relative_to(Path(root).resolve()):
                raise FileNotFoundError('Extracted SAFE has no original VVenC executable')
            shutil.copyfile(filename, staging)
        if sha256(staging) != expected_sha256:
            raise ValueError('Bundled VVenC SHA-256 mismatch; refusing unsafe resume')
        os.chmod(staging, 0o755)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    return destination
