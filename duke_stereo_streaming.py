# duke_stereo_streaming.py - Wrapper für stereoimage_generation mit Video-Streaming
import subprocess
import os
import re
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import torch
from PIL import Image

# Import der originalen Stereo-Funktionen
from .stereoimage_generation import apply_stereo_divergence


def resolve_output_path(path_template, params=None):
    """
    Resolve format strings and auto-increment if file exists.
    Supports: 
    - %date:FORMAT%, %time:FORMAT% (Java-style format converted to Python)
    - %param_name% for any node parameter (e.g. %divergence%, %model%)
    
    Args:
        path_template: Path with format strings
        params: Dict of node parameters to substitute
    """
    now = datetime.now()
    params = params or {}
    
    # Expand ~ to home directory
    path = str(Path(path_template).expanduser())
    
    # Convert %date:FORMAT% and %time:FORMAT% patterns
    def replace_format(match):
        fmt_type, fmt = match.group(1), match.group(2)
        # Convert Java-style format to Python strftime
        fmt = fmt.replace('yyyy', '%Y').replace('yy', '%y')
        fmt = fmt.replace('MM', '%m').replace('dd', '%d')
        fmt = fmt.replace('HH', '%H').replace('mm', '%M').replace('ss', '%S')
        return now.strftime(fmt)
    
    path = re.sub(r'%(date|time):([^%]+)%', replace_format, path)
    
    # Replace %param_name% with parameter values
    for key, val in params.items():
        path = path.replace(f'%{key}%', str(val))
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-increment if file exists
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base}_{counter:03d}{ext}"):
            counter += 1
        path = f"{base}_{counter:03d}{ext}"
    
    return path


def writer_thread(out_path, fps, width, height, queue, crf=19):
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', str(crf),
        '-preset', 'medium',
        out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while True:
        item = queue.get()
        if item is None:
            break
        proc.stdin.write(item.tobytes())
    proc.stdin.close()
    proc.wait()


class DukeStereoVideo:
    """
    ComfyUI Node für Stereo SBS Video mit Streaming (RAM-effizient).
    Verwendet die bewährte stereoimage_generation Logik.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "depth_maps": ("IMAGE",),
                "output_path": ("STRING", {"default": "/ComfyUI/output/stereo_%date:yyyy-MM-dd%_%time:HH-mm-ss%.mp4"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "divergence": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "separation": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "convergence_point": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stereo_offset_exponent": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "fill_technique": (["polylines_soft", "polylines_sharp", "naive", "none"],),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Stereo"

    def execute(self, images, depth_maps, output_path, fps, divergence, separation,
                convergence_point, stereo_offset_exponent, fill_technique, crf):
        
        import comfy.utils
        
        n_frames = images.shape[0]
        h, w = images.shape[1], images.shape[2]
        
        # Build params dict for path formatting
        path_params = {
            'divergence': divergence,
            'separation': separation,
            'stereo_offset_exponent': stereo_offset_exponent,
            'convergence_point': convergence_point,
            'fill_technique': fill_technique,
            'depth_blur': depth_blur,
            'depth_blur_edge_threshold': depth_blur_edge_threshold,
            'fps': fps,
            'crf': crf,
            'width': w,
            'height': h,
            'frames': n_frames,
        }
        
        # Resolve output path and validate
        output_path = resolve_output_path(output_path, path_params)
        output_dir = Path(output_path).parent
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise Exception(f"[DukeStereo] Cannot create directory: {output_dir}")
        
        test_file = output_dir / ".duke_stereo_write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise Exception(f"[DukeStereo] No write permission for: {output_dir}")
        
        print(f"[DukeStereo] Writing {n_frames} frames to: {output_path}")
        
        # Start writer thread (SBS = doppelte Breite)
        write_queue = Queue(maxsize=16)
        writer = Thread(target=writer_thread, args=(output_path, fps, w * 2, h, write_queue, crf))
        writer.start()
        
        pbar = comfy.utils.ProgressBar(n_frames)
        
        for i in range(n_frames):
            # Image und Depth extrahieren
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            depth_np = depth_maps[i].cpu().numpy()
            
            # Falls Depth 3-Kanal ist, auf 1 Kanal reduzieren
            if depth_np.ndim == 3:
                depth_np = depth_np[:, :, 0]
            
            # Depth Map auf Frame-Größe skalieren falls nötig
            if depth_np.shape[:2] != (h, w):
                depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
                depth_t = torch.nn.functional.interpolate(depth_t, size=(h, w), mode="bilinear", align_corners=True)
                depth_np = depth_t.squeeze().numpy()
            
            # Falls Depth 0-1 ist, auf 0-255 skalieren
            if depth_np.max() <= 1.0:
                depth_np = (depth_np * 255).astype(np.float32)
            
            # Stereo-Warp mit originaler Funktion
            left = apply_stereo_divergence(
                img_np, depth_np, divergence, -separation,
                stereo_offset_exponent, fill_technique, convergence_point
            )
            right = apply_stereo_divergence(
                img_np, depth_np, -divergence, separation,
                stereo_offset_exponent, fill_technique, convergence_point
            )
            
            # SBS zusammenfügen
            sbs = np.concatenate([left, right], axis=1)
            write_queue.put(sbs)
            pbar.update(1)
        
        # Writer beenden
        write_queue.put(None)
        writer.join()
        
        print(f"[DukeStereo] Done: {output_path}")
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "Duke Stereo Video": DukeStereoVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Duke Stereo Video": "Duke Stereo Video (SBS)"
}
