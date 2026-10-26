import time
import torch
import numpy as np

def set_global_seeds(seed=42):
    """
    Sets global seeds across Python, NumPy, and PyTorch for exact reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred):
    """
    Computes true regression metrics: MAE, RMSE, and MSE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse)
    }

def calculate_track_dwell_time(track_df, win_end_frame, fps=25.0):
    """
    Computes real remaining dwell-time target (in minutes) for tracks active at win_end_frame.
    Target = (last_frame_of_track - win_end_frame) / fps / 60.0
    """
    track_max_frames = track_df.groupby("id")["frame"].max().to_dict()
    active_ids = track_df[track_df["frame"] == win_end_frame]["id"].unique()
    
    if len(active_ids) == 0:
        return 0.0
        
    remaining_frames = [track_max_frames[tid] - win_end_frame for tid in active_ids if tid in track_max_frames]
    if len(remaining_frames) == 0:
        return 0.0
        
    avg_remaining_sec = np.mean(remaining_frames) / fps
    return max(0.0, float(avg_remaining_sec / 60.0))

def benchmark_inference_latency(model, sample_input, device, num_warmup=100, num_runs=1000):
    """
    Measures median batch-size-1 inference latency across 1000 runs after warmup.
    """
    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_input)
            
    # Measure executions
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            t1 = time.perf_counter()
            timings.append(t1 - t0)
            
    return float(np.median(timings))