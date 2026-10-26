import os
import torch
import pandas as pd
import numpy as np

from src.dataset import get_scene_splits
from src.features import extract_frame_features
from src.models import Proposed1DTCN, LSTMBaseline, CNNBaseline, MLPBaseline
from src.utils import set_global_seeds, compute_metrics, benchmark_inference_latency

def load_checkpoint(model, filepath, device):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Loaded weights from {filepath}")
    else:
        print(f"⚠️ Checkpoint {filepath} not found. Running with initial weights.")
    model.to(device)
    model.eval()
    return model

def run_evaluation():
    set_global_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Evaluation running on: {device}")

    # Load test scene MOT17-13 split exclusively
    _, _, (X_test, y_test) = get_scene_splits()
    
    if len(X_test) == 0:
        print("❌ Error: No test data found. Please ensure data/MOT17/train/ contains MOT17-13-DPM.")
        return

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    models_to_evaluate = {
        "1D-TCN (Proposed)": (Proposed1DTCN(), "models/tcn_model.pth", "Temporal CNN"),
        "LSTM-only": (LSTMBaseline(), "models/lstm_model.pth", "Recurrent Network"),
        "CNN-only": (CNNBaseline(), "models/cnn_model.pth", "1D Convolutional"),
        "Density-Feature MLP": (MLPBaseline(input_dim=50), "models/mlp_model.pth", "Density MLP")
    }

    results = []

    for name, (model_inst, ckpt_path, arch_type) in models_to_evaluate.items():
        model = load_checkpoint(model_inst, ckpt_path, device)
        
        # Predictions
        with torch.no_grad():
            preds = model(X_test_tensor.to(device)).cpu().numpy().flatten()
            
        metrics = compute_metrics(y_test, preds)
        latency = benchmark_inference_latency(model, X_test_tensor[0:1], device)

        results.append({
            "Model Name": name,
            "Model Type": arch_type,
            "MAE (min)": round(metrics["MAE"], 4),
            "RMSE (min)": round(metrics["RMSE"], 4),
            "MSE": round(metrics["MSE"], 4),
            "Latency (s)": round(latency, 6)
        })

    # Save benchmark table
    os.makedirs("paper_outputs", exist_ok=True)
    df_results = pd.DataFrame(results)
    output_path = "paper_outputs/table3_prediction.csv"
    df_results.to_csv(output_path, index=False)
    
    print("\n--- FINAL BENCHMARK EVALUATION RESULTS ---")
    print(df_results.to_string(index=False))
    print(f"\n✅ Results successfully saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()