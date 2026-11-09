"""Exercise directory pinning, both passes, successful cleanup, and failed mux."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# Reuse the lightweight package loader without importing ComfyUI's GPU runtime.
from test_da3_sequence import SequenceBuilder, DukeDA3Prepare, DukeStereoPrepared
from duke_test.directory_queue import DukeDA3DirectoryQueue, directory_run, DIRECTORY_KEY


class Manager:
    def __init__(self):
        self.outputs = {}
        self.inputs = {}
        self.has_closed_inputs = False
        self.total_frames = 3
        self.resets = 0

    def reset(self):
        for entry in list(self.outputs.values()):
            entry[-1].close()
        self.outputs = {}
        self.inputs = {}
        self.has_closed_inputs = False
        self.total_frames = float('inf')
        self.resets += 1


class Process:
    def __init__(self, *args, **kwargs):
        import io
        self.stdin = io.BytesIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self):
        self.returncode = 0
        return 0


class DirectoryTests(unittest.TestCase):
    def test_two_videos_two_passes_and_temporary_cleanup(self):
        self.run_directory(False)

    def test_mux_failure_keeps_cache_and_does_not_advance(self):
        self.run_directory(True)

    def run_directory(self, mux_failure):
        comfy = types.ModuleType('comfy')
        mm = types.ModuleType('comfy.model_management')
        mm.throw_exception_if_processing_interrupted = lambda: None
        utils = types.ModuleType('comfy.utils')
        class Progress:
            def __init__(self, total): pass
            def update_absolute(self, value): pass
        utils.ProgressBar = Progress
        comfy.model_management = mm
        comfy.utils = utils
        graph = types.ModuleType('comfy_execution.graph')
        class Blocker:
            def __init__(self, message): pass
        graph.ExecutionBlocker = Blocker
        def append(builder, images, infer, align, final=False):
            for _ in images:
                np.save(builder.path / f'{builder.count:08d}.npy', np.ones((2, 2), dtype=np.float32))
                builder.count += 1
            if final:
                builder.finish()
        modules = {'comfy': comfy, 'comfy.model_management': mm, 'comfy.utils': utils,
                   'comfy_execution.graph': graph}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(SequenceBuilder, 'append', append), \
                patch('duke_test.da3_sequence.requeue') as requeue, \
                patch('duke_test.duke_stereo_streaming.mux_audio',
                      side_effect=RuntimeError('mux failed') if mux_failure else None) as mux, \
                patch('subprocess.Popen', Process), \
                patch('duke_test.stereoimage_generation.apply_stereo_divergence',
                      side_effect=lambda rgb, *a, **k: rgb):
            root = Path(tmp) / 'inputs'; root.mkdir()
            for name in ('a.mp4', 'b.mp4'): (root / name).touch()
            manager = Manager(); queue = DukeDA3DirectoryQueue()
            prepare = DukeDA3Prepare(); stereo = DukeStereoPrepared()
            keys = []
            for video_index in range(2):
                selection = queue.execute(str(root), manager, unique_id='directory')
                keys.append(selection[1])
                self.assertEqual(selection[2:], (video_index + 1, 2))
                for final, count in ((False, 2), (True, 1)):
                    manager.has_closed_inputs = final
                    result = prepare.execute(torch.zeros(count, 2, 2, 3), {}, tmp, 'ignored',
                                             48, 0, 1, 64, 8, manager, 'prepare')
                    self.assertIsInstance(result['result'][0], Blocker)
                    self.assertEqual(directory_run(manager).index, video_index)
                path = Path(manager.outputs['prepare'][0].path)
                meta = json.loads((path / 'manifest.json').read_text())
                self.assertEqual(meta['source_key'], selection[1])
                self.assertTrue(meta['temporary_cache'])
                for final, count in ((False, 2), (True, 1)):
                    manager.has_closed_inputs = final
                    self.assertEqual(queue.execute(str(root), manager, unique_id='directory')[:2], selection[:2])
                    prepared = prepare.execute(torch.zeros(count, 2, 2, 3), {}, tmp, 'ignored',
                                               48, 0, 1, 64, 8, manager, 'prepare')
                    self.assertEqual(prepared['result'], (str(path),))
                    args = dict(images=torch.zeros(count, 2, 2, 3), depth_sequence=str(path),
                                source_key='ignored', start_frame=0, frame_stride=1,
                                output_path=str(Path(tmp) / 'stereo.mp4'), fps=48, divergence=-2.5,
                                separation=0, convergence_point=1, stereo_offset_exponent=2,
                                fill_technique='none', depth_blur=0, depth_blur_edge_threshold=6,
                                crf=19, meta_batch=manager, unique_id='stereo')
                    if mux_failure and final:
                        with self.assertRaisesRegex(RuntimeError, 'mux failed'):
                            stereo.execute(**args)
                        self.assertTrue(path.exists())
                        self.assertEqual(directory_run(manager).index, 0)
                        return
                    stereo.execute(**args)
                self.assertFalse(path.exists())
            self.assertNotEqual(keys[0], keys[1])
            self.assertFalse(manager.outputs)
            self.assertEqual(mux.call_count, 2)
            # Three continuations per clip plus one to move to the second clip.
            self.assertEqual(requeue.call_count, 7)

    def test_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'a.mp4'; source.touch()
            queue = DukeDA3DirectoryQueue(); manager = Manager()
            queue.execute(tmp, manager, unique_id='q')
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Source video changed'):
                queue.execute(tmp, manager, unique_id='q')
            manager.reset()
            self.assertNotIn(DIRECTORY_KEY, manager.outputs)


if __name__ == '__main__':
    unittest.main()
