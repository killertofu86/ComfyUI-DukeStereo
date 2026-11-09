"""Bounded DA3 processing. Uses the installed DA3 extension's private pipeline API."""
import importlib
import logging
import math
import json
import sys
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from .duke_stereo_streaming import DukeStereoVideoBatch
from .directory_queue import DIRECTORY_KEY, DukeDA3DirectoryQueue, directory_run, rewind, finish_video

logger = logging.getLogger(__name__)

def report_progress(stage, current, total, detail=""):
    logger.info("[Duke DA3] %s: %s/%s %s", stage, current, total or "?", detail)
    if total:
        try:
            from comfy.utils import ProgressBar
        except ImportError:
            return
        ProgressBar(total).update_absolute(min(current, total))

def remove_temporary_sequence(path, metadata):
    # Only delete a cache explicitly created and marked by this implementation.
    if (metadata.get("temporary_cache") and metadata.get("producer") == "DukeStereo-DA3-v1"
            and path.name.startswith("duke-depth-") and not path.is_symlink()):
        shutil.rmtree(path)



def da3_module(name):
    # ComfyUI may load custom nodes under a path-derived module name.
    for key, module in list(sys.modules.items()):
        if key.endswith('.nodes.streaming.pipeline') and 'DepthAnything' in str(getattr(module, '__file__', '')):
            root = key.rsplit('.nodes.', 1)[0]
            return importlib.import_module(root + '.nodes.' + name)
    import folder_paths
    for base in folder_paths.get_folder_paths('custom_nodes'):
        if (Path(base) / 'ComfyUI-DepthAnythingV3').is_dir():
            if base not in sys.path:
                sys.path.insert(0, base)
            return importlib.import_module('ComfyUI-DepthAnythingV3.nodes.' + name)
    raise RuntimeError('Installed ComfyUI-DepthAnythingV3 extension not found')


def requeue():
    try:
        from videohelpersuite.utils import requeue_workflow_unchecked
    except ModuleNotFoundError:
        import folder_paths
        for base in folder_paths.get_folder_paths('custom_nodes'):
            candidate = str(Path(base) / 'ComfyUI-VideoHelperSuite')
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        from videohelpersuite.utils import requeue_workflow_unchecked
    requeue_workflow_unchecked()


class SequenceBuilder:
    def __init__(self, parent, chunk_size, overlap, metadata):
        if not 0 < overlap <= chunk_size // 2:
            raise ValueError('Require 0 < overlap <= chunk_size / 2')
        Path(parent).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix='duke-depth-', dir=parent))
        self.size, self.overlap = chunk_size, overlap
        self.meta = metadata
        self.count = self.start = 0
        self.previous = None
        self.scale = 1.0
        self.pending = []
        self.progress = None
        self.sections = 0

    def append(self, images, infer, align, final=False):
        # Only chunk_size source frames and two inference results are resident.
        for image in images:
            self.pending.append(image.detach().cpu().clone())
            self.count += 1
            if len(self.pending) == self.size:
                self.process(infer, align)
        if final:
            if self.pending and (self.previous is None or len(self.pending) > self.overlap):
                self.process(infer, align)
            self.finish()

    def process(self, infer, align):
        self.sections += 1
        if self.progress:
            self.progress("Inference", self.start, f"section {self.sections}, {len(self.pending)} frames")
        result = infer(torch.stack(self.pending))
        if self.previous is not None:
            if self.progress:
                self.progress("Alignment", self.start, f"section {self.sections}")
            scale, _, _ = align(self.previous, result)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid Sim(3) alignment scale')
            self.scale *= float(scale)
        depth = result.depth.astype(np.float32) * self.scale
        if not np.isfinite(depth).all():
            raise ValueError('DA3 produced non-finite depth')
        for i, frame in enumerate(depth):
            target = self.path / f'{self.start+i:08d}.npy'
            if self.previous is not None and i < self.overlap:
                weight = (i + 1) / (self.overlap + 1)
                frame = np.load(target) * (1 - weight) + frame * weight
            np.save(target, frame.astype(np.float32), allow_pickle=False)
        # Retain raw overlap for pairwise local Sim(3), cumulative scale separately.
        from dataclasses import replace
        self.previous = replace(result, **{k: getattr(result, k)[-self.overlap:].copy()
                                          for k in ('depth', 'conf', 'sky', 'extrinsics', 'intrinsics')})
        self.start += len(self.pending) - self.overlap
        self.pending = self.pending[-self.overlap:]

    def finish(self):
        if not self.count:
            raise ValueError('Empty sequence')
        lo, hi = float('inf'), -float('inf')
        for i in range(self.count):
            if self.progress and (i % 16 == 0 or i + 1 == self.count):
                self.progress('Global depth range', i + 1, f'{self.count} frames')
            depth = np.load(self.path / f'{i:08d}.npy', mmap_mode='r')
            lo, hi = min(lo, float(depth.min())), max(hi, float(depth.max()))
        # Store lossless float32 raw depth and one global Standard normalization.
        manifest = dict(self.meta, version=1, frames=self.count, minimum=lo, maximum=hi,
                        normalization='global_standard_near_zero', chunk_size=self.size,
                        overlap=self.overlap)
        tmp = self.path / 'manifest.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(self.path / 'manifest.json')
        self.pending.clear()
        self.previous = None


@dataclass
class SequenceReplay:
    path: str


def idle_guard():
    # VHS recognizes and closes generator outputs when a run is reset.
    yield


class DukeDA3Prepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'da3_model': ('DA3MODEL',),
                'cache_directory': ('STRING', {'default': 'output/duke-depth'}),
                'source_key': ('STRING', {'default': 'clip-1'}),
                'fps': ('FLOAT', {'default': 48.0, 'min': 0.01}),
                'start_frame': ('INT', {'default': 0, 'min': 0}),
                'frame_stride': ('INT', {'default': 1, 'min': 1}),
                'chunk_size': ('INT', {'default': 64, 'min': 2, 'max': 128}),
                'overlap': ('INT', {'default': 8, 'min': 1, 'max': 64})},
                'optional': {'meta_batch': ('VHS_BatchManager',),
                             'workflow_mode': (['prepare_and_stereo', 'prepare_only'],
                                               {'default': 'prepare_and_stereo'}),
                             'cache_policy': (['keep', 'delete_after_stereo'], {'default': 'keep'})},
                'hidden': {'unique_id': 'UNIQUE_ID'}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('depth_sequence',)
    FUNCTION = 'execute'
    CATEGORY = 'Stereo/DA3'
    OUTPUT_NODE = True

    def execute(self, images, da3_model, cache_directory, source_key, fps,
                start_frame, frame_stride, chunk_size, overlap, meta_batch=None, unique_id=None,
                workflow_mode="prepare_and_stereo", cache_policy="keep"):
        import comfy.model_management as mm
        state_key = str(unique_id)
        run = directory_run(meta_batch)
        if run is not None:
            if workflow_mode != 'prepare_and_stereo':
                raise ValueError('Directory queue requires prepare_and_stereo mode')
            source_key = run.source_key
        total = getattr(meta_batch, 'total_frames', 0)
        total = int(total) if total and math.isfinite(total) else 0
        clip = f'video {run.index + 1}/{len(run.files)}' if run else ''
        if meta_batch is not None and state_key in meta_batch.outputs:
            state = meta_batch.outputs[state_key][0]
            if isinstance(state, SequenceReplay):
                if meta_batch.has_closed_inputs:
                    _, guard = meta_batch.outputs.pop(state_key)
                    guard.close()
                report_progress('Stereo pass', 0, 0, clip)
                return {'ui': {'text': ['Stereo pass: prepared depth loaded']},
                        'result': (state.path,)}
        else:
            state = SequenceBuilder(cache_directory, chunk_size, overlap,
                                    dict(source_key=source_key, fps=fps, start_frame=start_frame,
                                         frame_stride=frame_stride, producer='DukeStereo-DA3-v1',
                                         temporary_cache=run is not None or cache_policy == 'delete_after_stereo'))
            if meta_batch is not None:
                # Generator protocol is recognized by VHS reset on cancellation/restart.
                def cleanup():
                    try:
                        yield
                    finally:
                        state.pending.clear()
                        state.previous = None
                        if state.path.exists() and not (state.path / 'manifest.json').exists():
                            shutil.rmtree(state.path)
                guard = cleanup()
                next(guard)
                meta_batch.outputs[state_key] = (state, guard)
        state.progress = lambda stage, current, detail: report_progress(
            stage, current, state.count if stage == "Global depth range" else total, f"{clip} {detail}")
        report_progress("Prepare buffer", state.count, total,
                        f"{clip} {len(state.pending)}/{chunk_size} buffered")
        def infer(frames):
            pipeline_mod = da3_module('streaming.pipeline')
            utils = da3_module('utils')
            mm.throw_exception_if_processing_interrupted()
            if not Path(da3_model['model_path']).is_file():
                raise ValueError('DA3 weights must already exist locally')
            patcher = da3_module('load_model')._get_or_build_da3_model(da3_model)
            if not hasattr(patcher.model, '__call__'):
                raise RuntimeError('Incompatible DA3 model')
            dtype = patcher.model_options.get('da3_dtype', torch.float16)
            mm.load_models_gpu([patcher])
            pipeline = pipeline_mod.StreamingPipeline(patcher.model,
                       pipeline_mod.StreamingConfig(chunk_size=chunk_size, overlap=overlap,
                                                   align_lib='torch'), mm.get_torch_device(), dtype)
            self.pipeline = pipeline
            frames, _, _ = utils.resize_to_patch_multiple(frames.permute(0, 3, 1, 2))
            frames = utils.imagenet_normalize(frames).to(dtype=dtype).unsqueeze(0)
            with torch.inference_mode(), torch.autocast(device_type=mm.get_torch_device().type, dtype=dtype):
                return pipeline._process_chunk(frames, 0, frames.shape[1])
        final = meta_batch is None or meta_batch.has_closed_inputs
        try:
            state.append(images, infer, lambda a, b: self.pipeline._align_chunks(a, b), final)
        except BaseException:
            state.pending.clear()
            state.previous = None
            self.pipeline = None
            shutil.rmtree(state.path)
            if meta_batch is not None:
                entry = meta_batch.outputs.pop(state_key, None)
                if entry:
                    entry[-1].close()
            raise
        self.pipeline = None
        path = str(state.path)
        if not final:
            from comfy_execution.graph import ExecutionBlocker
            requeue()
            return {'ui': {'unfinished_batch': [True]},
                    'result': (ExecutionBlocker(None),)}
        if meta_batch is not None:
            entry = meta_batch.outputs.pop(state_key, None)
            if entry:
                entry[-1].close()
            if workflow_mode == 'prepare_and_stereo':
                if set(meta_batch.outputs) - {DIRECTORY_KEY}:
                    raise ValueError('Automatic two-pass mode requires its own VHS Batch Manager.')
                # VHS reopens its source generators on the next queued batch.
                # Keep the completed sequence in manager state so manual restart/reset
                # clears the phase as well; no stale state lives on the node instance.
                rewind(meta_batch)
                report_progress("Depth complete; starting Stereo", state.count, state.count, clip)
                guard = idle_guard()
                next(guard)
                meta_batch.outputs[state_key] = (SequenceReplay(path), guard)
                from comfy_execution.graph import ExecutionBlocker
                requeue()
                return {'ui': {'text': ['Depth complete; restarting source for Stereo'],
                               'unfinished_batch': [True]},
                        'result': (ExecutionBlocker(None),)}
            if not meta_batch.outputs:
                meta_batch.reset()
        return {'ui': {'text': [path]}, 'result': (path,)}


class DukeStereoPrepared(DukeStereoVideoBatch):
    normalize_depth = False

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        del inputs['required']['depth_maps']
        inputs['required']['fps'] = ('FLOAT', {'default': 48.0, 'min': 0.01})
        # A socket-only input stays with IMAGE/AUDIO instead of below widgets.
        inputs['required'].pop('images')
        inputs['required'] = {
            'images': ('IMAGE', {'lazy': True}),
            'depth_sequence': ('STRING', {'forceInput': True}),
            **inputs['required'],
        }
        inputs['required'].update(source_key=('STRING', {'default': 'clip-1'}),
                                  start_frame=('INT', {'default': 0, 'min': 0}),
                                  frame_stride=('INT', {'default': 1, 'min': 1}))
        return inputs

    def check_lazy_status(self, images=None, **kwargs):
        # The prepared path is blocked during pass one, so upstream upscaling
        # is requested only once the complete depth sequence is available.
        return ['images'] if images is None else []

    def execute(self, images, depth_sequence, source_key, start_frame, frame_stride, **kwargs):
        path = Path(depth_sequence)
        manifest = path / 'manifest.json'
        if not manifest.is_file():
            raise ValueError(
                'Depth sequence is incomplete or the directory is incorrect. '
                'Use prepare_and_stereo on Prepare with a shared VHS Batch Manager, '
                'or select an already completed sequence directory.')
        meta = json.loads(manifest.read_text())
        run = directory_run(kwargs.get('meta_batch'))
        if run is not None:
            source_key = run.source_key
        for key, value in dict(source_key=source_key, start_frame=start_frame,
                               frame_stride=frame_stride, fps=kwargs['fps']).items():
            if meta[key] != value:
                raise ValueError(f'Depth sequence mismatch: {key}')
        manager, uid = kwargs.get('meta_batch'), kwargs.get('unique_id')
        import subprocess
        from .duke_stereo_streaming import resolve_output_path, mux_audio
        from .stereoimage_generation import apply_stereo_divergence
        import comfy.model_management as mm
        h, w = images.shape[1:3]
        if manager is not None and uid in manager.outputs:
            state = manager.outputs[uid][0]
            if state['sequence'] != str(path.resolve()) or state['shape'] != (h, w):
                raise ValueError('Sequence or image dimensions changed during batch')
        else:
            output = resolve_output_path(kwargs['output_path'], dict(kwargs, source_key=source_key, width=w, height=h, frames=meta['frames']))
            proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{w*2}x{h}', '-r', str(kwargs['fps']),
                '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf',
                str(kwargs['crf']), '-preset', 'medium', output], stdin=subprocess.PIPE)
            state = dict(cursor=0, proc=proc, output=output, audio=kwargs.get('audio'),
                         sequence=str(path.resolve()), shape=(h, w))
            def cleanup():
                try:
                    yield
                finally:
                    try:
                        if not proc.stdin.closed:
                            proc.stdin.close()
                    except BrokenPipeError:
                        pass
                    finally:
                        if proc.poll() is None:
                            proc.wait()
            guard = cleanup()
            next(guard)
            state['guard'] = guard
            if manager is not None:
                manager.outputs[uid] = (state, guard)
        end = state['cursor'] + len(images)
        final = manager is None or manager.has_closed_inputs
        try:
            if end > meta['frames'] or (final and end != meta['frames']):
                raise ValueError('Source and prepared depth frame counts differ')
            for i, image in enumerate(images, state['cursor']):
                mm.throw_exception_if_processing_interrupted()
                depth = np.load(path / f'{i:08d}.npy', allow_pickle=False)
                depth = (depth - meta['minimum']) / max(meta['maximum'] - meta['minimum'], 1e-8)
                depth = torch.nn.functional.interpolate(torch.from_numpy(depth.copy())[None, None],
                        size=(h, w), mode='bilinear', align_corners=True)[0, 0].numpy()
                if kwargs['depth_blur'] > 0:
                    from scipy.ndimage import gaussian_filter
                    blurred = gaussian_filter(depth, sigma=kwargs['depth_blur'])
                    threshold = kwargs['depth_blur_edge_threshold'] / 255
                    if threshold > 0:
                        gy, gx = np.gradient(depth)
                        depth = np.where(np.hypot(gx, gy) > threshold, depth, blurred)
                    else:
                        depth = blurred
                rgb = (image.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                views = [apply_stereo_divergence(rgb, depth, sign * kwargs['divergence'],
                         -sign * kwargs['separation'], kwargs['stereo_offset_exponent'],
                         kwargs['fill_technique'], kwargs['convergence_point'], normalize_depth=False)
                         for sign in (1, -1)]
                state['proc'].stdin.write(np.concatenate(views, axis=1).tobytes())
                if (i + 1) % 8 == 0 or i + 1 == meta['frames']:
                    from comfy.utils import ProgressBar
                    ProgressBar(meta['frames']).update_absolute(i + 1)
            state['cursor'] = end
            report_progress('Stereo frames', end, meta['frames'],
                            f'video {run.index + 1}/{len(run.files)}' if run else '')
            if not final:
                requeue()
                return {'ui': {'unfinished_batch': [True]}, 'result': (state['output'],)}
            state['guard'].close()
            if state['proc'].returncode:
                raise RuntimeError('Stereo ffmpeg failed')
            report_progress('Finalize video and audio', end, meta['frames'])
            mux_audio(state['output'], state['audio'])
            remove_temporary_sequence(path, meta)
        except BaseException:
            state['guard'].close()
            if manager is not None:
                manager.outputs.pop(uid, None)
            raise
        if manager is not None:
            manager.outputs.pop(uid, None)
            if finish_video(manager):
                requeue()
                return {'ui': {'unfinished_batch': [True]}, 'result': (state['output'],)}
        return (state['output'],)


NODE_CLASS_MAPPINGS = {'Duke DA3 Directory Queue': DukeDA3DirectoryQueue,
                       'Duke DA3 Prepare Sequence': DukeDA3Prepare,
                       'Duke Stereo Prepared Sequence': DukeStereoPrepared}
NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in NODE_CLASS_MAPPINGS}
