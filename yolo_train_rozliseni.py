#Příklad spuštění skriptu
# python yolo_train_rozliseni.py \
#   --seed 42 \
#   --dataset datasets/flickr32/data.yaml \
#   --project Resolution_flickr32 \
#   --models yolov8n.pt yolov8x.pt


import sys
import os
import gc
import torch
import argparse
import random
import numpy as np
from ultralytics import YOLO, settings

# ==============================================================================
# TŘÍDA PRO PŘESMĚROVÁNÍ VÝSTUPU
# ==============================================================================
class FileLogger:
    def __init__(self, filename):
        self.out_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.out_file.write(message)
        self.out_file.flush()

    def flush(self):
        self.out_file.flush()

    def close(self):
        self.out_file.close()

# ==============================================================================
# 1. PARSOVÁNÍ ARGUMENTŮ
# ==============================================================================
parser = argparse.ArgumentParser(description="Univerzální trénovací skript pro YOLO.")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--dataset', type=str, required=True, help='Cesta k data.yaml (např. /tmp/qmul_yolo/data.yaml)')
parser.add_argument('--project', type=str, required=True, help='Název W&B projektu (např. Resolution_qmul)')
parser.add_argument('--models', nargs='+', default=['yolov8n.pt', 'yolo11m.pt'], help='Seznam modelů k trénování')
args = parser.parse_args()

# ==============================================================================
# 2. UZAMČENÍ NÁHODY
# ==============================================================================
os.environ["PYTHONHASHSEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================================================================
# 3. KONFIGURACE W&B A YOLO
# ==============================================================================
# POZOR: Do přílohy BP vkládej skript výhradně bez reálného API klíče!
os.environ["WANDB_API_KEY"] = "TVUJ_API_KLIC"
os.environ["WANDB_PROJECT"] = args.project

settings.update({
    "wandb": True,
    "tensorboard": False,
    "sync": False,
    "datasets_dir": "/home/jovyan"
})

# ==============================================================================
# 4. HLAVNÍ SMYČKA EXPERIMENTŮ
# ==============================================================================
EPOCHS = 100

for model_name in args.models:
    clean_model_name = model_name.replace('.pt', '')
    run_name = f"{clean_model_name}_img1024_seed{args.seed}"
    os.environ["WANDB_RUN_GROUP"] = f"{clean_model_name}_img1024"
    
    log_filename = f"log_{run_name}.txt"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    logger = FileLogger(log_filename)
    sys.stdout, sys.stderr = logger, logger
    
    try:
        model = YOLO(model_name)
        model.train(
            data=args.dataset,
            epochs=EPOCHS,                
            imgsz=1024,
            cache=False,
            batch=8,
            workers=2,
            seed=args.seed,
            deterministic=True,
            project=args.project,
            name=run_name,
            plots=True,
            save=True,
            exist_ok=True,
            amp=True
        )
    except Exception as e:
        print(f"KRITICKÁ CHYBA u {run_name}: {e}")
    finally:
        import wandb
        try: wandb.finish()
        except: pass

        if 'model' in locals(): del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        sys.stdout, sys.stderr = original_stdout, original_stderr
        logger.close()
        print(f"Běh {run_name} dokončen. Log uložen do {log_filename}")