# ComfyUI-DukeStereo

A ComfyUI custom node that converts images/video frames to side-by-side (SBS) stereo format using depth estimation.

## Features

- GPU-accelerated depth estimation using Depth-Anything-V2
- Batch processing for efficient video frame handling
- Configurable depth and stereo parameters

## Installation

Clone into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/killertofu86/ComfyUI-DukeStereo.git
pip install -r ComfyUI-DukeStereo/requirements.txt
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| model | Large | Depth model: Large or Base |
| depth_size | 518 | Depth map resolution |
| depth_batch_size | 64 | Batch size for depth estimation |
| warp_batch_size | 64 | Batch size for stereo warping |
| divergence | 2.5 | 3D strength |
| depth_blur | 6 | Blur kernel for depth smoothing |

## License

MIT
