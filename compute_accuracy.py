import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from src.dataset import get_scene_splits
from src.models import Proposed1DTCN, LSTMBaseline, CNNBaseline, MLPBaseline

def compute_all_accuracies():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_train, y_train), _, (X_test, y_test) = get_scene_splits()
    
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 5))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)
    X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # Tolerances in minutes (5 sec = 5/60 min, 10 sec = 10/60 min)
    tol_5s = 5.0 / 60.0
    tol_10s = 10.0 / 60.0

    models = {
        "1D-TCN (Proposed)": (Proposed1DTCN(), "models/tcn_model.pth"),
        "LSTM-only": (LSTMBaseline(), "models/lstm_model.pth"),
        "CNN-only": (CNNBaseline(), "models/cnn_model.pth"),
        "Density-Feature MLP": (MLPBaseline(input_dim=50), "models/mlp_model.pth"),
    }

    results = []

    # 1. Evaluate PyTorch Models
    for name, (model, path) in models.items():
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                preds = model(X_te_t).cpu().numpy().flatten()
            preds = np.clip(preds, 0.0, None)
            
            # Compute Overall Prediction Accuracy (1 - Normalized MAE)
            range_y = np.max(y_test) - np.min(y_test) if (np.max(y_test) - np.min(y_test)) > 0 else np.mean(y_test)
            mae = np.mean(np.abs(y_test - preds))
            overall_acc = max(0.0, (1.0 - (mae / range_y)) * 100.0)
            
            acc_5s = np.mean(np.abs(y_test - preds) <= tol_5s) * 100.0
            acc_10s = np.mean(np.abs(y_test - preds) <= tol_10s) * 100.0
            
            results.append({
                "Model Name": name,
                "Overall Acc (%)": round(overall_acc, 2),
                "Accuracy @ ±5s (%)": round(acc_5s, 2),
                "Accuracy @ ±10s (%)": round(acc_10s, 2)
            })

    # 2. Linear Regression Baseline
    X_tr_flat = scaler.transform(X_train.reshape(-1, 5)).reshape(len(X_train), -1)
    X_te_flat = X_test_scaled.reshape(len(X_test), -1)
    lr = LinearRegression().fit(X_tr_flat, y_train)
    preds_lr = np.clip(lr.predict(X_te_flat), 0.0, None)
    
    range_y = np.max(y_test) - np.min(y_test) if (np.max(y_test) - np.min(y_test)) > 0 else np.mean(y_test)
    mae_lr = np.mean(np.abs(y_test - preds_lr))
    overall_acc_lr = max(0.0, (1.0 - (mae_lr / range_y)) * 100.0)
    
    acc_5s_lr = np.mean(np.abs(y_test - preds_lr) <= tol_5s) * 100.0
    acc_10s_lr = np.mean(np.abs(y_test - preds_lr) <= tol_10s) * 100.0
    
    results.append({
        "Model Name": "Linear Regression",
        "Overall Acc (%)": round(overall_acc_lr, 2),
        "Accuracy @ ±5s (%)": round(acc_5s_lr, 2),
        "Accuracy @ ±10s (%)": round(acc_10s_lr, 2)
    })

    # 3. Naive Baseline
    preds_naive = np.full_like(y_test, fill_value=np.mean(y_train))
    mae_naive = np.mean(np.abs(y_test - preds_naive))
    overall_acc_n = max(0.0, (1.0 - (mae_naive / range_y)) * 100.0)
    
    acc_5s_n = np.mean(np.abs(y_test - preds_naive) <= tol_5s) * 100.0
    acc_10s_n = np.mean(np.abs(y_test - preds_naive) <= tol_10s) * 100.0
    
    results.append({
        "Model Name": "Naive Baseline",
        "Overall Acc (%)": round(overall_acc_n, 2),
        "Accuracy @ ±5s (%)": round(acc_5s_n, 2),
        "Accuracy @ ±10s (%)": round(acc_10s_n, 2)
    })

    df_acc = pd.DataFrame(results)
    df_acc.to_csv("paper_outputs/table4_accuracy_percentages.csv", index=False)
    
    print("\n--- OVERALL ACCURACY PERCENTAGE BENCHMARK ---")
    print(df_acc.to_string(index=False))
    print("\nSaved to paper_outputs/table4_accuracy_percentages.csv")

if __name__ == "__main__":
    compute_all_accuracies()