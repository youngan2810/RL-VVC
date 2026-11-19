"""Source-pinned V6 Kaggle handoff and audit cleanup regressions."""
import gzip
import tempfile
import unittest
from pathlib import Path

import torch

from .d3qn_v6_handoff import (
    CHECKPOINT_VERSION, MANIFEST_SCHEMA,
    preflight_matches, verify_checkpoint, verify_checkpoint_digests, verify_manifest, sha256,
)
from .d3qn_v6_native_runner import close_and_compress_native


def identity():
    return {'protocol': 'native-masked-d3qn-qp/v6-action-conditioned', 'seed': 20261111,
            'source_fingerprint': 'source', 'encoder_sha256': 'encoder-A',
            'detector_sha256': 'weights', 'dataset_sha256': 'dataset',
            'learning': {'n_step': 8, 'target_tau': .005, 'learning_rate': 5e-5,
                         'action_mix': .45, 'boundary_mix': .10,
                         'epsilon_floor': .2, 'epsilon_decay_steps': 10240}}


def checkpoint_payload(ident):
    return {'version': CHECKPOINT_VERSION, 'identity': ident, 'steps': 256,
            'updates': 49, 'config': {'n_step': 8, 'target_tau': .005,
                                     'lr': 5e-5, 'action_mix': .45,
                                     'boundary_mix': .10},
            'online': {}, 'target': {}, 'optimizer': {}, 'replay': {},
            'coverage_counts': torch.zeros((5, 25)),
            'agent_rng_state': {}, 'torch_cpu_rng_state': torch.get_rng_state(),
            'next_episode_seed': 11}


class HandoffTests(unittest.TestCase):
    def test_close_clears_native_but_compression_still_happens(self):
        with tempfile.TemporaryDirectory() as folder:
            native_dir = Path(folder) / 'codec'
            native_dir.mkdir()
            for name in ('events.jsonl', 'cus.jsonl', 'qgs.jsonl'):
                (native_dir / name).write_bytes(b'{"picture":1}\n')
            class Native: output = native_dir
            class Env:
                native = Native()
                closed = False
                def close(self):
                    self.closed = True
                    self.native = None
            env = Env()
            close_and_compress_native(env)
            self.assertTrue(env.closed)
            self.assertIsNone(env.native)
            for name in ('events.jsonl', 'cus.jsonl', 'qgs.jsonl'):
                self.assertFalse((native_dir / name).exists())
                with gzip.open(native_dir / (name + '.gz'), 'rb') as f:
                    self.assertEqual(f.read(), b'{"picture":1}\n')
            close_and_compress_native(env)

    def test_checkpoint_identity_is_strict_including_encoder_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'checkpoint.pt'
            torch.save(checkpoint_payload(identity()), file)
            self.assertEqual(verify_checkpoint(file, identity(), expected_steps=256)['steps'], 256)
            mismatch = identity()
            mismatch['encoder_sha256'] = 'different-build'
            with self.assertRaisesRegex(ValueError, 'encoder_sha256'):
                verify_checkpoint(file, mismatch)
            with self.assertRaisesRegex(ValueError, 'steps'):
                verify_checkpoint(file, identity(), expected_steps=512)
            payload = checkpoint_payload(identity())
            payload['version'] = 'd3qn-native-checkpoint/v5-coding-cycle'
            torch.save(payload, file)
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                verify_checkpoint(file, identity())

    def test_safe_manifest_and_checkpoint_digests(self):
        ident = identity()
        source = 'a' * 40
        manifest = {'schema': MANIFEST_SCHEMA, 'identity': ident,
                    'source_commit': source, 'protocol': ident['protocol'],
                    'seed': ident['seed'], 'test_used': False,
                    'train_steps': 256, 'completed_episodes': 1}
        self.assertTrue(verify_manifest(manifest, source, ident, completed_episodes=1))
        with self.assertRaisesRegex(ValueError, 'identity'):
            verify_manifest(manifest, source, {**ident, 'encoder_sha256': 'modified'})
        with self.assertRaisesRegex(ValueError, 'schema'):
            verify_manifest({**manifest, 'schema': 'd3qn-v6-action-conditioned-mini16-val4/1'},
                            source, ident)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'checkpoints/latest_checkpoint.pt'
            file.parent.mkdir()
            file.write_bytes(b'ckpt')
            recorded = {'checkpoints/latest_checkpoint.pt': sha256(file)}
            self.assertTrue(verify_checkpoint_digests(folder, recorded))
            file.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_checkpoint_digests(folder, recorded)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                verify_checkpoint_digests(folder, {'../escape.pt': 'hash'})

    def test_preflight_cache_binds_both_encoder_and_decoder(self):
        report = {'gate': 'REAL_VVC_STATE_ACTION_AU_DECODE_PASS', 'encoder_sha256': 'a'}
        contract = {'encoder_sha256': 'a', 'decoder_sha256': 'b'}
        self.assertTrue(preflight_matches(report, contract, encoder_sha256='a', decoder_sha256='b'))
        self.assertFalse(preflight_matches(report, contract, encoder_sha256='a', decoder_sha256='new'))
        self.assertFalse(preflight_matches(report, {}, encoder_sha256='a', decoder_sha256='b'))


if __name__ == '__main__':
    unittest.main()
