#Příklad spuštění skriptu
# python rfdetr_train_rozliseni.py \
#   --seed 42 \
#   --dataset datasets/qmul_yolo \
#   --project Resolution_qmul

import sys
import os
import gc
import torch
import argparse
import logging
import random
import numpy as np

# ==============================================================================
# TŘÍDA PRO PŘESMĚROVÁNÍ VÝSTUPU
# ==============================================================================
class FileLogger:
    def __init__(self, filename):
        self.out_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.out_file.write(message)
        self.out_file.flush()

    def flush(self): self.out_file.flush()
    def close(self): self.out_file.close()

# ==============================================================================
# 1. PARSOVÁNÍ ARGUMENTŮ
# ==============================================================================
parser = argparse.ArgumentParser(description="Univerzální trénovací skript pro RF-DETR.")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--dataset', type=str, required=True, help='Cesta ke složce datasetu')
parser.add_argument('--project', type=str, required=True, help='Název W&B projektu')
args = parser.parse_args()

# ==============================================================================
# 2. UZAMČENÍ NÁHODY (Opraveno!)
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
# 3. IMPORTY A W&B
# ==============================================================================
try:
    import wandb
except ImportError:
    print("WandB chybí."); sys.exit()

from rfdetr import RFDETRSmall, RFDETRLarge

os.environ["WANDB_API_KEY"] = "TVUJ_API_KLIC"
os.environ["WANDB_PROJECT"] = args.project
os.environ["WANDB_MODE"] = "online" 

# ==============================================================================
# 4. HLAVNÍ SMYČKA
# ==============================================================================
models_to_test = {
    # "rfdetr_small": RFDETRSmall,
    "rfdetr_large": RFDETRLarge
}

EPOCHS = 100

for model_name, ModelClass in models_to_test.items():
    run_name = f"{model_name}_img{1024}_seed{args.seed}"
    os.environ["WANDB_RUN_GROUP"] = model_name
    
    log_filename = f"log_{run_name}.txt"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    logger = FileLogger(log_filename)
    sys.stdout, sys.stderr = logger, logger

    # Přesměrování RF-DETR loggeru
    rf_logger = logging.getLogger("rf-detr")
    root_logger = logging.getLogger()
    for h in rf_logger.handlers[:] + root_logger.handlers[:]:
        h.name == "rf-detr" and rf_logger.removeHandler(h)
        root_logger.removeHandler(h)

    new_handler = logging.StreamHandler(logger)
    new_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    rf_logger.addHandler(new_handler)
    root_logger.addHandler(new_handler)
    rf_logger.propagate = False
    
    try:
        model = ModelClass(resolution=1024)
        model.train(
            dataset_dir=args.dataset,
            epochs=EPOCHS,
            batch_size=2,
            grad_accum_steps=4,
            gradient_checkpointing=True,
            lr=1e-4,
            output_dir=f"runs/{run_name}",
            wandb=True,                   
            project=args.project,      
            run=run_name                  
        )
    except Exception as e:
        print(f"KRITICKÁ CHYBA u {run_name}: {e}")
    finally:
        try: wandb.finish()
        except: pass

        if 'model' in locals(): del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        sys.stdout, sys.stderr = original_stdout, original_stderr
        logger.close()
        print(f"Běh {run_name} dokončen. Log uložen do {log_filename}")