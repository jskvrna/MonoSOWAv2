import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2

image = Image.open('/path/to/MonoDETR/data/KITTIDataset/training/image_2/000001.png')
image = np.array(image.convert("RGB"))

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

sam2_checkpoint = "/path/to/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')

input_box = np.array([[391, 181, 421, 200]])

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )