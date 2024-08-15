
import argparse, os
import wandb
from ultralytics import YOLO
#from wandb.integration.ultralytics import add_wandb_callback
from pathlib import Path
import pandas as pd
from types import SimpleNamespace



# defaults
default_config = SimpleNamespace(

    batch_size=16, #16 keep small in Colab to be manageable
    hsv_h = 0.1, 
    hsv_v = 0.1, 
    scale = 0.3, 
    epochs=70, # for brevity, increase for better results :)
    lr=1e-2,
    pretrained=True,  # whether to use pretrained encoder,
    seed=42,
    dropout = 0.2,
    optimizer = "AdamW",
    close_mosaic = 10,
    mosaic = 1,
    lrf = 0.1,
    imgsz = 640
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--cos_lr', type=bool, default=True, help='cos_lr')
    argparser.add_argument('--pretrained', type=bool, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--dropout', type=float, default=default_config.dropout, help='use fp16')
    argparser.add_argument('--optimizer', type=str, default=default_config.optimizer, help='optimizer')
    argparser.add_argument('--imgsz', type=str, default=default_config.imgsz, help='imgsz')
    argparser.add_argument('--lrf', type=str, default=default_config.lrf, help='lrf')
    argparser.add_argument('--mosaic', type=str, default=default_config.mosaic, help='mosaic')
    argparser.add_argument('--model', type=str, default="yolov8n", help='model')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def train(config):
    
    
    run = wandb.init(project="conference_ROI", entity="elbadawy990-sejong-university", job_type="training", config=config)
        
    # good practice to inject params using sweeps
    config = run.config
    

    if config.model == 'yolov8m':
        
        model = YOLO('yolov8m.yaml')

    elif config.model == 'yolov9m':

        model = YOLO('yolov9m.pt')
        

    elif config.model == 'yolov10m':
  
        model = YOLO('yolov10m.pt')
    
    
    #Add W&B Callback for Ultralytics
    #add_wandb_callback(model, enable_model_checkpointing=True)
    
    results = model.train(project="conference_ROI", data='datasets/data.yaml', epochs=default_config.epochs, imgsz=config.imgsz, batch=config.batch_size, \
                        lr0=config.lr, lrf = config.lrf , workers=8, plots = True, dropout = config.dropout, seed = default_config.seed, pretrained = config.pretrained, cos_lr = config.cos_lr,\
                        hsv_h = default_config.hsv_h, hsv_s = 0, hsv_v = default_config.hsv_v, degrees = 5,  translate = 0, single_cls = True ,scale = 0.1, mosaic = config.mosaic, erasing = 0, \
                        device = 0, optimizer = config.optimizer, name = 'ROI_esxtraction_ex')
    
    
    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(default_config)

