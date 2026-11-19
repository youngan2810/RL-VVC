"""Source-pinned V6 Kaggle handoff and audit cleanup regressions."""
import gzip
import json
import zipfile
import tempfile
import unittest
from pathlib import Path

import torch

from .d3qn_v6_handoff import (
    CHECKPOINT_VERSION, MANIFEST_SCHEMA,
    preflight_matches, verify_checkpoint, verify_checkpoint_digests, verify_manifest, sha256,
    read_safe_manifest, restore_encoder_from_safe,
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

    def test_restore_exact_encoder_from_zip_and_extracted_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            original = b'fake-vvenc-binary-for-integrity-test'
            original_file = directory/'original'
            original_file.write_bytes(original)
            expected_hash = sha256(original_file)
            ident = identity()
            ident['encoder_sha256'] = expected_hash
            manifest = {'schema': MANIFEST_SCHEMA, 'identity': ident}
            zip_path = directory/'safe.zip'
            with zipfile.ZipFile(zip_path, 'w') as archive:
                archive.writestr('outer/learning_manifest.json', json.dumps(manifest))
                archive.writestr('outer/TOOLCHAIN/vvencapp', original)
            recovered = directory/'restored/vvencapp'
            self.assertEqual(restore_encoder_from_safe(zip_path, expected_hash, recovered), recovered)
            self.assertEqual(recovered.read_bytes(), original)
            self.assertTrue(recovered.stat().st_mode & 0o111)
            with self.assertRaisesRegex(ValueError, 'identity'):
                restore_encoder_from_safe(zip_path, 'incorrect', recovered)
            extracted = directory/'extracted'
            (extracted/'TOOLCHAIN').mkdir(parents=True)
            (extracted/'learning_manifest.json').write_text(json.dumps(manifest))
            (extracted/'TOOLCHAIN/vvencapp').write_bytes(original)
            self.assertEqual(read_safe_manifest(extracted)[0]['identity'], ident)
            self.assertEqual(restore_encoder_from_safe(extracted, expected_hash, recovered), recovered)
            (extracted/'TOOLCHAIN/vvencapp').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                restore_encoder_from_safe(extracted, expected_hash, recovered)
            self.assertEqual(recovered.read_bytes(), original)

    def test_preflight_cache_binds_both_encoder_and_decoder(self):
        report = {'gate': 'REAL_VVC_STATE_ACTION_AU_DECODE_PASS', 'encoder_sha256': 'a'}
        contract = {'encoder_sha256': 'a', 'decoder_sha256': 'b'}
        self.assertTrue(preflight_matches(report, contract, encoder_sha256='a', decoder_sha256='b'))
        self.assertFalse(preflight_matches(report, contract, encoder_sha256='a', decoder_sha256='new'))
        self.assertFalse(preflight_matches(report, {}, encoder_sha256='a', decoder_sha256='b'))


    def test_native_validation_complete_partial_failed_and_dedup_contract(self):
        """Simulates deadline after two cases and a failure after one case."""
        from .d3qn_v6_native_runner import write_validation_artifacts
        from .d3qn_v6_handoff import read_validation_artifacts
        expected = [('vid0', 120, 'fixed2'), ('vid0', 120, 'fixed9'),
                    ('vid0', 120, 'qpa_anchor')]
        def row(case):
            return {'video': case[0], 'rate_kbps': case[1], 'method': case[2],
                    'summary': {'objective': .5}, 'decoder_parity': True}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = [row(c) for c in expected[:2]]
            write_validation_artifacts(output, expected, first, 'PARTIAL')
            self.assertFalse((output / 'validation.json').exists())
            status, completed, error = read_validation_artifacts(output, expected)
            self.assertEqual(status, 'PARTIAL')
            self.assertEqual(len(completed), 2)
            self.assertIsNone(error)
            missing = [case for case in expected if case not in
                       {(r['video'], r['rate_kbps'], r['method']) for r in completed}]
            self.assertEqual(missing, expected[2:])
            with self.assertRaisesRegex(ValueError, 'COMPLETE requires'):
                write_validation_artifacts(output, expected, first, 'COMPLETE')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                write_validation_artifacts(output, expected, first + first[:1], 'FAILED')
            with self.assertRaisesRegex(ValueError, 'unrequested'):
                write_validation_artifacts(output, expected, first + [row(('other', 120, 'fixed2'))], 'FAILED')
            write_validation_artifacts(output, missing, [], 'FAILED', 'simulated decoder exception')
            self.assertFalse((output / 'validation.json').exists())
            status, rows, error = read_validation_artifacts(output, missing)
            self.assertEqual((status, rows, error), ('FAILED', [], 'simulated decoder exception'))
            write_validation_artifacts(output, missing, [row(missing[0])], 'COMPLETE')
            self.assertTrue((output / 'validation.json').is_file())
            status, rows, _ = read_validation_artifacts(output, missing)
            self.assertEqual(status, 'COMPLETE')
            self.assertEqual(first + rows, [row(c) for c in expected])
            # Corruption must be reported, never treated as a completed case.
            (output / 'validation_partial.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Invalid|differs'):
                read_validation_artifacts(output, missing)


if __name__ == '__main__':
    unittest.main()
