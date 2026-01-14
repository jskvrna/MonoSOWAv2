from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import DatasetEnum
import mmcv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import cv2

def generate_random_colors(num_colors):
    """Generate distinct RGB colors with improved contrast"""
    return [(random.randint(100, 255),  # Avoid dark colors
             random.randint(100, 255),
             random.randint(100, 255)) 
            for _ in range(num_colors)]

def apply_boolean_masks(image_path, masks, output_path, alpha=0.5):
    """
    Apply boolean masks with automatically generated colors
    
    Args:
        image_path: Path to base image
        masks: List of 2D boolean arrays
        output_path: Result image path
        alpha: Transparency level (0-1)
    """
    # Generate colors equal to number of masks
    colors = generate_random_colors(len(masks))
    img_orig = cv2.imread(image_path)
    alpha_value = 0.75
    
    for i, mask in enumerate(masks):
        img_orig[mask == 1, 2] = (1 - alpha_value) * img_orig[mask == 1, 2] + alpha_value * colors[i][0]
        img_orig[mask == 1, 1] = (1 - alpha_value) * img_orig[mask == 1, 1] + alpha_value * colors[i][1]
        img_orig[mask == 1, 0] = (1 - alpha_value) * img_orig[mask == 1, 0] + alpha_value * colors[i][2]
    
    cv2.imwrite(output_path, img_orig)

config_file = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py'
checkpoint_file = 'pytorch_model.pth'

#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
#checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')

img_path = 'demo.jpg'
result = inference_detector(model, img_path)

results = result[1] #We want masks only with scores

classes = len(results[0])

for class_id in range(classes):
    if class_id == 2:
        bboxes = result[0][class_id]
        detections = results[0][class_id]
        scores = results[1][class_id]
        
        new_masks = []
        for i in range(len(detections)):
            if scores[i] > 0.25:
                new_masks.append(detections[i])
        detections = np.array(new_masks)
        print(len(detections))
        
        apply_boolean_masks(image_path=img_path, masks=detections, output_path="output_image_masks.png", alpha=0.5)
        
        output_path = 'output_image.jpg'
        colors = ['red', 'green', 'blue', 'yellow', 'purple']  # Color cycle

        # Load image and setup drawing
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        # Use a basic font (PIL's default)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        # Process each bounding box
        for i, bbox in enumerate(bboxes):
            # Convert coordinates and extract score
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            score = bbox[4]
            
            if score < 0.25:
                continue
            
            # Select color (cycle through list)
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], 
                          outline=color, width=2)
            
            # Create text background
            text = f"{score:.2f}"
            text_width, text_height = font.getsize(text)
            draw.rectangle([x_min, y_min - text_height - 2,
                           x_min + text_width + 2, y_min],
                           fill=color)
            
            # Draw score text
            draw.text((x_min + 1, y_min - text_height - 1),
                     text, fill='white', font=font)

        # Save result
        img.save(output_path)
        print(f"Success: Saved annotated image to {output_path}")
        print(f"Boxes drawn: {len(bboxes)}")
        print(f"Colors used: {colors[:len(bboxes)]}")

