import numpy as np
import cv2

# detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# Detectron2 config
cfg = get_cfg()
cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE='cpu'

predictor = DefaultPredictor(cfg)

def align_document(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    center = np.mean(box, axis=0)
    corners = sorted(box, key=lambda corner: np.arctan2(corner[1] - center[1], corner[0] - center[0]))

    width = int(max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[2] - corners[3])))
    height = int(max(np.linalg.norm(corners[1] - corners[2]), np.linalg.norm(corners[3] - corners[0])))
    aligned_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(np.float32(corners), aligned_corners)

    aligned_image = cv2.warpPerspective(image, M, (width, height))

    return aligned_image

def rotate_image_if_needed(image):
    height, width = image.shape[:2]
    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def get_card(image):
    result = []
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    classes = instances.pred_classes
    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy()
    book_class_index = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index('book')
    book_indices = np.where(classes == book_class_index)[0]

    if len(book_indices) > 0:
        book_boxes = boxes[book_indices]
        book_masks = masks[book_indices]

        for box, mask in zip(book_boxes, book_masks):
            x1, y1, x2, y2 = box.astype(int)
            mask = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
            aligned_image = align_document(image[y1:y2, x1:x2], mask)
            result.append(rotate_image_if_needed(aligned_image))
    return result
