import cv2

from unilm.dit.object_detection.ditod import add_vit_config

import torch

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from huggingface_hub import hf_hub_download

import gradio as gr

import PIL

from numpy import asarray

# Step 1: instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("cascade_dit_base.yml")

# # Step 2: add model weights URL to config
cfg.MODEL.WEIGHTS = ("publaynet_dit-b_cascade.pth")

# Step 3: set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 4: define model
predictor = DefaultPredictor(cfg)

def analyze_image(img):
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["text","title","list","table","figure"])
    
    output = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output["instances"].to("cpu"))
    result_image = result.get_image()[:, :, ::-1]
    
    return result_image

iface = gr.Interface(fn=analyze_image, 
                     inputs=gr.Image(type="numpy", label="document image"), 
                     outputs=gr.Image(type="numpy", label="annotated document")
                    )
iface.launch(debug=True)

