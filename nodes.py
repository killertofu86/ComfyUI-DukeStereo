# duke_stereo_nodes.py - DukeStereoSBS with Polylines Fill
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from time import time
from queue import Queue
from threading import Thread
import subprocess

try:
    from numba import njit, prange
except Exception as e:
    print(f"WARNING! Numba failed to import! Stereo generation will be much slower! ({str(e)})")
    from builtins import range as prange
    def njit(parallel=False):
        def Inner(func): return lambda *args, **kwargs: func(*args, **kwargs)
        return Inner


def writer_thread(out_path, fps, width, height, queue, crf=19):
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width*2}x{height}',
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


@njit(parallel=True)
def apply_stereo_divergence_polylines(original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float, fill_technique: str):
    """
    Polylines-basierter Stereo-Warp mit intelligentem Lücken-Füllen.
    Aus stereoimage_generation.py übernommen.
    """
    EPSILON = 1e-7
    PIXEL_HALF_WIDTH = 0.45 if fill_technique == 'polylines_sharp' else 0.0
    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    
    for row in prange(h):
        pt = np.zeros((5 + 2 * w, 3), dtype=np.float32)
        pt_end: int = 0
        pt[pt_end] = [-1.0 * w, 0.0, 0.0]
        pt_end += 1
        
        for col in range(0, w):
            coord_d = (normalized_depth[row][col] ** stereo_offset_exponent) * divergence_px
            coord_x = col + 0.5 + coord_d + separation_px
            if PIXEL_HALF_WIDTH < EPSILON:
                pt[pt_end] = [coord_x, abs(coord_d), col]
                pt_end += 1
            else:
                pt[pt_end] = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt[pt_end + 1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt_end += 2
        
        pt[pt_end] = [2.0 * w, 0.0, w - 1]
        pt_end += 1
        sg_end: int = pt_end - 1
        sg = np.zeros((sg_end, 6), dtype=np.float32)
        
        for i in range(sg_end):
            sg[i] += np.concatenate((pt[i], pt[i + 1]))
        
        for i in range(1, sg_end):
            u = i - 1
            while pt[u][0] > pt[u + 1][0] and 0 <= u:
                pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
                sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
                u -= 1
        
        csg = np.zeros((5 * int(abs(divergence_px)) + 25, 6), dtype=np.float32)
        csg_end: int = 0
        sg_pointer: int = 0
        pt_i: int = 0
        
        for col in range(w):
            color = np.full(c, 0.5, dtype=np.float32)
            while pt[pt_i][0] < col:
                pt_i += 1
            pt_i -= 1
            
            while pt[pt_i][0] < col + 1:
                coord_from = max(col, pt[pt_i][0]) + EPSILON
                coord_to = min(col + 1, pt[pt_i + 1][0]) - EPSILON
                significance = coord_to - coord_from
                coord_center = coord_from + 0.5 * significance
                
                while sg_pointer < sg_end and sg[sg_pointer][0] < coord_center:
                    csg[csg_end] = sg[sg_pointer]
                    sg_pointer += 1
                    csg_end += 1
                
                csg_i = 0
                while csg_i < csg_end:
                    if csg[csg_i][3] < coord_center:
                        csg[csg_i] = csg[csg_end - 1]
                        csg_end -= 1
                    else:
                        csg_i += 1
                
                best_csg_i: int = 0
                if csg_end != 1:
                    best_csg_closeness: float = -EPSILON
                    for csg_i in range(csg_end):
                        ip_k = (coord_center - csg[csg_i][0]) / (csg[csg_i][3] - csg[csg_i][0])
                        closeness = (1.0 - ip_k) * csg[csg_i][1] + ip_k * csg[csg_i][4]
                        if best_csg_closeness < closeness and 0.0 < ip_k < 1.0:
                            best_csg_closeness = closeness
                            best_csg_i = csg_i
                
                col_l: int = int(csg[best_csg_i][2] + EPSILON)
                col_r: int = int(csg[best_csg_i][5] + EPSILON)
                if col_l == col_r:
                    color += original_image[row][col_l] * significance
                else:
                    ip_k = (coord_center - csg[best_csg_i][0]) / (csg[best_csg_i][3] - csg[best_csg_i][0])
                    color += (original_image[row][col_l] * (1.0 - ip_k) +
                              original_image[row][col_r] * ip_k
                              ) * significance
                pt_i += 1
            
            derived_image[row][col] = np.asarray(color, dtype=np.uint8)
    
    return derived_image


class DukeStereoSBS:
    """
    Node für GPU-beschleunigte SBS Stereo-Bilder mit Polylines Fill-Technik.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_path": ("STRING", {"default": "/ComfyUI/output/stereo_output.mp4"}),
                "images": ("IMAGE",),
            },
            "optional": {
                "model": (["Large", "Base"],),
                "depth_size": ("INT", {"default": 518, "min": 128, "max": 1024}),
                "depth_batch_size": ("INT", {"default": 64, "min": 4, "max": 256}),
                "warp_batch_size": ("INT", {"default": 64, "min": 4, "max": 256}),
                "depth_blur": ("INT", {"default": 6, "min": 0, "max": 10}),
                "divergence": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "fill_technique": (["polylines_soft", "polylines_sharp", "none"],),
                "stereo_offset_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "convergence_point": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute"
    CATEGORY = "Stereo"

    @torch.no_grad()
    def depth_batch(self, imgs, depth_batch_size, depth_size):
        depths = []
        for i in range(0, len(imgs), depth_batch_size):
            batch = imgs[i:i + depth_batch_size]
            inputs = self.proc(images=batch, return_tensors="pt", size=depth_size).to("cuda")
            out = self.model(**inputs).predicted_depth
            depths.extend([d.cpu().numpy() for d in out])
        return depths

    def normalize_depth(self, depth, h, w, convergence_point):
        """Depth auf Bildgröße interpolieren und normalisieren mit Convergence Point."""
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
        depth_t = torch.nn.functional.interpolate(depth_t, size=(h, w), mode="bilinear", align_corners=True)
        depth_np = depth_t.squeeze().numpy()
        
        # Normalisieren auf 0-1
        d_min, d_max = depth_np.min(), depth_np.max()
        normalized = (depth_np - d_min) / (d_max - d_min + 1e-7)
        
        # Convergence point anwenden (verschiebt wo Null-Parallax liegt)
        normalized = normalized - convergence_point
        
        return normalized

    def stereo_warp_polylines(self, imgs, depths, divergence, depth_blur, warp_batch_size, 
                               fill_technique, stereo_offset_exponent, convergence_point):
        """Stereo-Warp mit Polylines Fill-Technik."""
        
        divergence_px = divergence * (self.w / 100)
        
        for i in range(0, len(imgs), warp_batch_size):
            batch_imgs = imgs[i:i + warp_batch_size]
            batch_depths = depths[i:i + warp_batch_size]
            
            for img, depth in zip(batch_imgs, batch_depths):
                img_np = np.array(img, dtype=np.uint8)
                depth_norm = self.normalize_depth(depth, self.h, self.w, convergence_point)
                
                # Optional: Depth Blur
                if depth_blur > 0:
                    kernel = depth_blur if depth_blur % 2 == 1 else depth_blur + 1
                    from scipy.ndimage import uniform_filter
                    depth_norm = uniform_filter(depth_norm, size=kernel)
                
                if fill_technique == "none":
                    # Fallback: einfacher grid_sample (alter Code)
                    left = self._simple_warp(img_np, depth_norm, divergence_px)
                    right = self._simple_warp(img_np, depth_norm, -divergence_px)
                else:
                    # Polylines-basierter Warp
                    left = apply_stereo_divergence_polylines(
                        img_np, depth_norm, divergence_px, 0.0, stereo_offset_exponent, fill_technique
                    )
                    right = apply_stereo_divergence_polylines(
                        img_np, depth_norm, -divergence_px, 0.0, stereo_offset_exponent, fill_technique
                    )
                
                # SBS zusammenfügen (RGB für ffmpeg)
                sbs = np.concatenate([left, right], axis=1)
                self.write_queue.put(sbs)

    def _simple_warp(self, img_np, depth_norm, divergence_px):
        """Einfacher GPU-Warp als Fallback (alte Methode)."""
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
        depth_t = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).float().cuda()
        
        disp = depth_t * divergence_px * 2 / self.w
        grid = self.base_grid.clone()
        grid[..., 0] += disp.squeeze()
        
        warped = torch.nn.functional.grid_sample(img_t, grid, padding_mode='border', align_corners=True)
        warped_np = (warped.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return warped_np

    def load_model(self, model):
        model_id = f"depth-anything/Depth-Anything-V2-{model}-hf"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).cuda().half()

    def execute(self, images, output_path, model="Large", fps=24, depth_size=518, 
                divergence=2.5, depth_blur=6, warp_batch_size=64, depth_batch_size=64,
                fill_technique="polylines_soft", stereo_offset_exponent=1.0, convergence_point=0.5, crf=19):
        
        self.load_model(model)
        
        self.h, self.w = images.shape[1], images.shape[2]
        
        # base_grid für Fallback-Warp
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, self.h), torch.linspace(-1, 1, self.w), indexing='ij')
        self.base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).cuda()
        
        self.write_queue = Queue(maxsize=16)
        writer = Thread(target=writer_thread, args=(output_path, fps, self.w, self.h, self.write_queue, crf))
        writer.start()
        
        imgs = [Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)) for img in images]
        depths = self.depth_batch(imgs, depth_batch_size, depth_size)
        
        self.stereo_warp_polylines(imgs, depths, divergence, depth_blur, warp_batch_size,
                                    fill_technique, stereo_offset_exponent, convergence_point)
        
        self.write_queue.put(None)
        writer.join()
        
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "DukeStereoSBS": DukeStereoSBS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DukeStereoSBS": "Duke SBS"
}
