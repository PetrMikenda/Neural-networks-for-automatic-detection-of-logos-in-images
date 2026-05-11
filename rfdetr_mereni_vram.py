import torch
import time
import gc
import numpy as np
from rfdetr import RFDETRLarge, RFDETRSmall

def benchmark_rfdetr(model_path, model_class, imgsz=640):
    device = torch.device('cuda:0')
    
    print(f"🚀 Načítám RF-DETR z: {model_path}")
    # Inicializace wrapperu
    wrapper = model_class(pretrain_weights=model_path)
    model = wrapper.model.model # Přístup k čistému modelu pro měření na tenzoru
    model.to(device)
    model.eval()
    model.half() # Převod na FP16

    # Předalokace tenzoru na GPU
    input_tensor = torch.zeros((1, 3, imgsz, imgsz), device=device, dtype=torch.float16)

    # 1. Zahřívací fáze (20 iterací)
    with torch.inference_mode():
        for _ in range(20):
            _ = model(input_tensor)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # 2. Ostré měření (100 iterací)
    latencies = []
    
    with torch.inference_mode():
        for _ in range(100):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = model(input_tensor)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    # 3. Výsledky
    avg_latency = (sum(latencies) / 100) * 1000
    fps = 1000 / avg_latency
    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 50)
    print(f"Průměrná latence:       {avg_latency:.2f} ms")
    print(f"FPS:                    {fps:.2f}")
    print(f"Špičková VRAM:          {max_vram:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    benchmark_rfdetr("path/to/checkpoint.pth", RFDETRLarge)