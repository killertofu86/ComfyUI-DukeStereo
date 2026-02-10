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

## Output Path Examples

```
~/video/stereo_%date:yyyy-MM-dd%_%time:HH-mm-ss%.mp4
~/video/test_div%divergence%.mp4
/output/stereo_%width%x%height%.mp4
```

## License

MIT
