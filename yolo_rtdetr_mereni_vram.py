import torch
import time
import gc
from ultralytics import YOLO, RTDETR

def benchmark_ultralytics(model_path, imgsz=640):
    device = torch.device('cuda:0')
    
    # 1. Příprava modelu a GPU
    if 'rtdetr' in model_path.lower():
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)
    
    model.to(device)
    model.eval() # Nastavení do režimu inference

    # 2. Předalokace tenzoru přímo na GPU (eliminace diskových operací a dekódování)
    # Používáme float16 (half precision), což odpovídá produkčnímu nasazení na A40
    input_tensor = torch.zeros((1, 3, imgsz, imgsz), device=device, dtype=torch.float16)

    # 3. Zahřívací fáze (20 iterací)
    for _ in range(20):
        # Měříme end-to-end včetně post-processingu (NMS)
        _ = model.predict(source=input_tensor, imgsz=imgsz, half=True, verbose=False)

    # Reset statistik paměti
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # 4. Ostré měření (100 iterací)
    latencies = []
    
    for _ in range(100):
        torch.cuda.synchronize() # Synchronizace před začátkem
        start_time = time.perf_counter()
        
        # Inference + Post-processing
        _ = model.predict(source=input_tensor, imgsz=imgsz, half=True, verbose=False)
        
        torch.cuda.synchronize() # Synchronizace po dokončení asynchronních operací
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    # 5. Výpočet výsledků
    avg_latency = (sum(latencies) / 100) * 1000
    fps = 1000 / avg_latency
    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    params = sum(p.numel() for p in model.model.parameters())

    print("-" * 50)
    print(f"Průměrná latence:       {avg_latency:.2f} ms")
    print(f"FPS:                    {fps:.2f}")
    print(f"Špičková VRAM:          {max_vram:.2f} MB")
    print(f"Počet parametrů:        {params:,}")
    print("-" * 50)

if __name__ == "__main__":
    benchmark_ultralytics("yolo11m.pt")