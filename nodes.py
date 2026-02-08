# my_node.py
import torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from time import time
from queue import Queue
from threading import Thread
import cv2

def writer_thread(out_path, fps, width, height, queue):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width*2, height))
    while True:
        item = queue.get()
        if item is None: break
        out.write(item)
    out.release()

class DukeStereoSBS:
    """
    Ein Node, der aus Bilder hauptsächlich auf der GPU SBS Stero bilder der selben auflösung (breite mal 2) macht.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_path": ("STRING", {"default": "/ComfyUI/output/stereo_output.mp4"}),
                "images": ("IMAGE",),                           # Bild-Input von anderem Node
            },
            "optional": {
                "model": (["Large", "Base"],),                  # COMBO/Dropdown
                "depth_size": ("INT", {"default": 518, "min": 128, "max": 1024}),
                "depth_batch_size": ("INT", {"default": 64, "min": 4, "max": 256}),
                "warp_batch_size": ("INT", {"default": 64, "min": 4, "max": 256}),
                "depth_blur": ("INT", {"default": 6, "min": 0, "max": 10}),
                "divergence": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute"
    CATEGORY = "Stereo"

    @torch.no_grad()
    def depth_batch(self,imgs, depth_batch_size,depth_size):
        depths = []
        for i in range(0, len(imgs), depth_batch_size):
            batch = imgs[i:i+depth_batch_size]
            inputs = self.proc(images=batch, return_tensors="pt",size=depth_size).to("cuda")
            out = self.model(**inputs).predicted_depth
            depths.extend([d.cpu() for d in out])
        return depths

    @torch.no_grad()
    def stereo_warp_streaming(self,imgs, depths, div, warp_batch_size,depth_blur):
        for i in range(0, len(imgs), warp_batch_size):
            batch_imgs = imgs[i:i+warp_batch_size]
            batch_depths = depths[i:i+warp_batch_size]
            img_t = torch.stack([torch.from_numpy(np.array(im)).permute(2,0,1) for im in batch_imgs]).float().cuda() / 255
            depth_t = torch.stack([d.cuda() for d in batch_depths]).unsqueeze(1).float()
            depth_t = torch.nn.functional.interpolate(depth_t, size=(self.h, self.w), mode="bilinear", align_corners=True)
            if depth_blur > 0:
                kernel = depth_blur if depth_blur % 2 == 1 else depth_blur + 1  # ungerade machen
                pad = kernel // 2
                depth_t = torch.nn.functional.avg_pool2d(
                    torch.nn.functional.pad(depth_t, (pad, pad, pad, pad), mode='replicate'),
                    kernel, stride=1, padding=0
                )
            depth_t = depth_t / depth_t.amax(dim=(2,3), keepdim=True)
            disp = (depth_t - 0.5) * div * (self.w / 100)
            grid = self.base_grid.expand(len(batch_imgs), -1, -1, -1).clone()
            gl, gr = grid.clone(), grid.clone()
            gl[..., 0] += disp.squeeze(1) * 2 / self.w
            gr[..., 0] -= disp.squeeze(1) * 2 / self.w
            left = torch.nn.functional.grid_sample(img_t, gl, padding_mode='border', align_corners=True)
            right = torch.nn.functional.grid_sample(img_t, gr, padding_mode='border', align_corners=True)
            # Optimiert: Alles auf GPU
            sbs = torch.cat([left, right], dim=3)           # SBS auf GPU
            sbs = sbs.flip(dims=[1])
            sbs = (sbs.permute(0,2,3,1) * 255).to(torch.uint8)  # NCHW→NHWC, uint8   
            sbs_np = sbs.cpu().numpy()                       # Ein Transfer pro Batch
            for frame in sbs_np:
                self.write_queue.put(frame)

    def load_model(self, model):
        model_id = f"depth-anything/Depth-Anything-V2-{model}-hf"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).cuda().half()

        
    def execute(self, images,output_path, model="Large",fps=24, depth_size=518, divergence=2.5, depth_blur=6,warp_batch_size=64,depth_batch_size=64):
        self.load_model(model)

        # h, w aus erstem Bild
        self.h, self.w = images.shape[1], images.shape[2]
        # base_grid für Warping
        gy, gx = torch.meshgrid(torch.linspace(-1,1,self.h), torch.linspace(-1,1,self.w), indexing='ij')

        self.base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).cuda()

        self.write_queue = Queue(maxsize=16)
        writer = Thread(target=writer_thread, args=(output_path, fps, self.w, self.h, self.write_queue))
        writer.start()
        imgs = [Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)) for img in images]
        depths = self.depth_batch(imgs, depth_batch_size, depth_size)
        self.stereo_warp_streaming(imgs, depths, divergence, warp_batch_size, depth_blur)
        self.write_queue.put(None)
        writer.join()

        return (output_path,)


# Mappings für die Registrierung der Node
NODE_CLASS_MAPPINGS = {
    "DukeStereoSBS": DukeStereoSBS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DukeStereoSBS": "Duke SBS"
}
