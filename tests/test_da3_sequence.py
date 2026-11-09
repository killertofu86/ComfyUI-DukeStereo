"""Run with the ComfyUI Python environment: python -m unittest discover -s tests."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('duke_test')
package.__path__ = [str(ROOT)]
sys.modules['duke_test'] = package
from duke_test.da3_sequence import SequenceBuilder, DukeDA3Prepare, SequenceReplay, DukeStereoPrepared


@dataclass
class Result:
    depth: np.ndarray
    conf: np.ndarray
    sky: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray


class SequenceTests(unittest.TestCase):
    def test_arbitrary_batches_and_tail(self):
        for count in (1, 8, 64, 65, 120, 121, 723):
            for batch in (7, 16, 64, 80):
                with self.subTest(count=count, batch=batch), tempfile.TemporaryDirectory() as tmp:
                    state = SequenceBuilder(tmp, 64, 8, {})
                    calls = []
                    def infer(images):
                        depth = images[..., 0].numpy()
                        calls.append(len(images))
                        return Result(depth, depth, depth, depth, depth)
                    for start in range(0, count, batch):
                        end = min(start + batch, count)
                        images = torch.arange(start, end).float()[:, None, None, None].expand(-1, 2, 2, 3)
                        state.append(images, infer, lambda a, b: (1., None, None), end == count)
                    self.assertLessEqual(max(calls), 64)
                    self.assertEqual(len(list(state.path.glob('*.npy'))), count)
                    for i in range(count):
                        np.testing.assert_allclose(np.load(state.path / f'{i:08d}.npy'), i, atol=1e-4)
                    manifest = json.loads((state.path / 'manifest.json').read_text())
                    self.assertEqual((manifest['minimum'], manifest['maximum']), (0, count-1))

    def test_bad_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            for overlap in (0, 64, 33):
                with self.assertRaises(ValueError):
                    SequenceBuilder(tmp, 64, overlap, {})


class AutomaticPassTests(unittest.TestCase):
    def test_complete_prepare_rewinds_and_replays_then_clears_state(self):
        class Manager:
            def __init__(self):
                self.outputs = {}
                self.has_closed_inputs = False
                self.resets = 0
            def reset(self):
                self.outputs.clear()
                self.has_closed_inputs = False
                self.resets += 1
        class Blocker:
            def __init__(self, message):
                self.message = message
        comfy = types.ModuleType('comfy')
        mm = types.ModuleType('comfy.model_management')
        comfy.model_management = mm
        graph = types.ModuleType('comfy_execution.graph')
        graph.ExecutionBlocker = Blocker
        def append(builder, images, infer, align, final=False):
            builder.count += len(images)
            if final:
                (builder.path / 'manifest.json').write_text('{}')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                'comfy': comfy, 'comfy.model_management': mm,
                'comfy_execution.graph': graph}), patch.object(SequenceBuilder, 'append', append), \
                patch('duke_test.da3_sequence.requeue') as queue:
            manager = Manager()
            node = DukeDA3Prepare()
            args = dict(images=torch.zeros(2, 2, 2, 3), da3_model={}, cache_directory=tmp,
                        source_key='test', fps=48, start_frame=0, frame_stride=1,
                        chunk_size=64, overlap=8, meta_batch=manager, unique_id='prepare')
            first = node.execute(**args)
            self.assertIsInstance(first['result'][0], Blocker)
            manager.has_closed_inputs = True
            last = node.execute(**args)
            self.assertIsInstance(last['result'][0], Blocker)
            self.assertEqual(manager.resets, 1)
            replay = manager.outputs['prepare'][0]
            self.assertIsInstance(replay, SequenceReplay)
            # A new node instance still finds phase state on the manager.
            replay_node = DukeDA3Prepare()
            self.assertEqual(replay_node.execute(**args)['result'], (replay.path,))
            manager.has_closed_inputs = True
            self.assertEqual(replay_node.execute(**args)['result'], (replay.path,))
            self.assertNotIn('prepare', manager.outputs)
            self.assertEqual(queue.call_count, 2)

    def test_stereo_images_are_lazy(self):
        node = DukeStereoPrepared()
        self.assertTrue(node.INPUT_TYPES()['required']['images'][1]['lazy'])
        self.assertEqual(node.check_lazy_status(), ['images'])
        self.assertEqual(node.check_lazy_status(images=torch.zeros(1)), [])


if __name__ == '__main__':
    unittest.main()
