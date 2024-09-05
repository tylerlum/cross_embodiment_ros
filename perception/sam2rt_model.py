import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from segment-anything-2-real-time.sam2.build_sam import build_sam2_camera_predictor

# Adapted from: github.com/Gy920/segment-anything-2-real-time/blob/main/demo/demo.py

class SAM2Model():

    def __init__(self):
        self.checkpoint = "segment-anything-2-real-time/checkpoints/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
        device = torch.device("cuda")
        self.predictor = build_sam2_camera_predictor(self.model_cfg, self., device=device)

    
    def predict(self, image, prompts=None, first=True, viz=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            width, height = image.shape[:2][::-1]
            if first:
                self.predictor.load_first_frame(image)
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

                if prompts == None:
                    prompts = self.generate_prompts()
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=prompts['points'],
                    labels=prompts['labels'],
                    # box=prompts['box'],
                )

                if viz: # show the prompt on the first frame
                    plt.figure(figsize=(9, 6))
                    plt.title(f"frame {ann_frame_idx}")
                    plt.imshow(Image.open(os.path.join(img_dir, frame_names[ann_frame_idx])))
                    self.show_points(points, labels, plt.gca())
                    self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

            else:
                out_obj_ids, out_mask_logits = predictor.track(image)
                if len(out_obj_ids) > 1:
                    all_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    for i in range(len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy(0.astype(np.uint8)) * 255
                        all_mask = cv2.bitwise_or(all_mask, out_mask)
                    mask_img = np.stack((all_mask, all_mask, all_mask), dim=2)
                else:
                    out_mask_logits = out_mask_logits.squeeze(dim=0).squeeze(dim=0)
                    out_mask_logits = out_mask_logits.cpu().numpy()
                    mask_img[out_mask_logits[0] > 0] = [255, 255, 255] # object is in white
                    
            return mask_img
    

    def generate_prompts(self):
        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = np.array([[240, 320]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)
        # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
        box = np.array([322, 415, 387, 480], dtype=np.float32)

        return {'points': points, 'labels': labels, 'box': box}
        

    # Visualization Utils
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    
    def visualize_first_frame(self, img_dir):
        # take a look the first video frame
        frame_idx = 0
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(img_dir, frame_names[frame_idx])))

