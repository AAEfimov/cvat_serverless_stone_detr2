import json
import base64
import io
from PIL import Image

import torch
from detectron2.model_zoo import get_config
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2        import model_zoo
from detectron2.model_zoo.model_zoo import _ModelZooUrls
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

CONFIG_OPTS = ["MODEL.WEIGHTS", "model_final.pth_3"]
CONFIDENCE_THRESHOLD = 0.5

def init_context(context):
    context.logger.info("Init context...  0%")

    if torch.cuda.is_available():
        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cuda'])
    else:
        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cpu'])

    cfg = get_cfg()

    model_config = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" # model - model_final.pth_3

    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.merge_from_list(CONFIG_OPTS)

    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 400    # 1000 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    predictor = DefaultPredictor(cfg)

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run stone segmentation model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    predictions = context.user_data.model_handler(image)

    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes

    # print(f"instances: {instances}")
    # print(f"pred_boxes: {pred_boxes}")
    # print(f"scores: {scores}")
    # print(f"pred_classes: {pred_classes}")

    results = []
    for box, score, label in zip(pred_boxes, scores, pred_classes):
        context.logger.info("iter")
        label = COCO_CATEGORIES[int(label)]["name"]
        if score >= threshold:
            results.append({
                "confidence": str(float(score)),
                "label": label,
                "points": box.tolist(),
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
