# ComfyUI-DukeStereo

A ComfyUI custom node that converts video frames to side-by-side (SBS) stereo format for VR viewing.

## Features

- **Polylines Fill** - High-quality gap filling during stereo warping (from StereoImage Node)
- **Edge-aware depth blur** - Smooths depth while preserving edges
- **Streaming output** - RAM-efficient, writes frames directly to video
- **H.264/yuv420p** - VR-player compatible output
- **Format-Strings** - `%date:yyyy-MM-dd%`, `%time:HH-mm-ss%`, `%divergence%`, `%width%` etc.
- **Home-Expansion** - `~/video/...` paths supported
- **Auto-Increment** - Existing files get `_001`, `_002` suffix

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/killertofu86/ComfyUI-DukeStereo.git
pip install -r ComfyUI-DukeStereo/requirements.txt
```

## Requirements

- torch, transformers, numpy, Pillow, numba, scipy, ffmpeg

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| output_path | stereo_output.mp4 | Output path with format strings |
| divergence | 2.5 | 3D strength (higher = more depth) |
| separation | 0.0 | Additional horizontal offset |
| stereo_offset_exponent | 2.0 | Non-linear depth curve |
| convergence_point | 1.0 | Zero-parallax plane (0.0-1.0) |
| fill_technique | polylines_soft | Gap filling: polylines_soft, polylines_sharp, none |
| depth_blur | 0 | Depth blur kernel (0 = off) |
| depth_blur_edge_threshold | 6.0 | Edge preservation threshold |
| fps | 24 | Output framerate |
| crf | 19 | Quality (lower = better) |

## Inputs

- **images** - Video frames (IMAGE)
- **depth_maps** - Depth maps from LeReS or similar (IMAGE)
- **audio** - Optional ComfyUI AUDIO, e.g. the audio output of VHS Load Video

Both stereo output nodes support audio. Connect the video loader's `audio`
output directly to the stereo node; audio bypasses image upscaling and depth
processing. Without this connection, output remains silent.

The batch node retains the first batch's audio and muxes it once, after the last
meta-batch. Supply the complete source audio, as VHS Load Video does, rather than
individual audio chunks. Audio is encoded as AAC; the H.264 video is copied
without re-encoding. Longer audio is trimmed to the video; shorter audio is
padded with silence. Temporary audio files use the output directory and are
removed after muxing. If muxing fails, the silent video is retained.

For a directory workflow, expose the path loader's AUDIO output from its group
and route audio from the same source as the selected images. Output fps must
match the loader's effective frame rate for synchronization. After updating,
restart ComfyUI before starting a new meta-batch run.

## Output Path Examples

```
~/video/stereo_%date:yyyy-MM-dd%_%time:HH-mm-ss%.mp4
~/video/test_div%divergence%.mp4
/output/stereo_%width%x%height%.mp4
```

## License

MIT
