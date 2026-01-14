from segment_anything import SamPredictor, sam_model_registry
import cv2, time, glob, torch, os
import numpy as np
import matplotlib.pyplot as plt
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Load the lazyconfig, we are using this, not the classic, because this regnety learned with different idea is used
cfg = LazyConfig.load("/path/to/detectron2/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py")
# cfg = LazyConfig.load("/path/to/detectron2/model_zoo/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py")  # (redacted)
cfg.train.init_checkpoint = 'https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ/42045954/model_final_ef3a80.pkl'  # replace with the path were you have your model

# Init the model
model = instantiate(cfg.model)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
model.eval()

sam = sam_model_registry["vit_h"](checkpoint="/path/to/models/sam.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

index = 0
pic_num_min = 0
pic_num_max = 100
pics = glob.glob('/path/to/data/kitti/training/image_2/*.png')
pics = sorted(pics, key=os.path.basename)
for pic_index in range(pic_num_min, pic_num_max):
    # Open the image, convert
    pic = pics[pic_index]
    img = cv2.imread(pic)
    img = np.array(img, dtype=np.uint8)
    img_det = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        outputs = model([{'image': torch.from_numpy(img_det)}])

    predictor.set_image(img)

    out_det = outputs[0]["instances"]

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    for z in range(len(out_det)):
        # It has score big enough and also it is a car then we want to adjust the mask According to SAM
        if out_det.scores[z] > 0.7 and out_det.pred_classes[z] == 2:
            box_instance = out_det.pred_boxes[z].tensor[0].cpu()
            box = np.array([box_instance[0], box_instance[1], box_instance[2], box_instance[3]]).astype(int)
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            show_mask(masks[0], plt.gca())
            show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig("/path/to/output/SAM/" + str(pic_index) + "_SAM" + ".png")

    plt.figure(figsize=(7.5,2.5))
    plt.imshow(img)
    for z in range(len(out_det)):
        # It has score big enough and also it is a car then we want to adjust the mask According to SAM
        if out_det.scores[z] > 0.7 and out_det.pred_classes[z] == 2:
            box_instance = out_det.pred_boxes[z].tensor[0].cpu()
            box = np.array([box_instance[0], box_instance[1], box_instance[2], box_instance[3]]).astype(int)
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            show_mask(out_det.pred_masks[z], plt.gca())
            show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig("/path/to/output/SAM/" + str(pic_index) + "_DET" + ".png")

    print(pic_index)

