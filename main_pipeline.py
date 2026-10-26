import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from src.dataset import get_scene_splits
from src.models import Proposed1DTCN, LSTMBaseline, CNNBaseline, MLPBaseline
from src.utils import set_global_seeds, benchmark_inference_latency

def fit_scaler(X_train):
    scaler = StandardScaler()
    N, T, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)
    return scaler

def transform_data(X, scaler):
    if len(X) == 0:
        return X
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(N, T, F)

def train_pytorch_model(model, loader, device, epochs=50):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(y_true, y_pred):
    # Enforce non-negative wait time boundary
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    return mae, rmse, mse

def main():
    set_global_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Compute Hardware Device: {device}")
    
    os.makedirs("paper_outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("⏳ Loading MOT17 dataset scene splits...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_scene_splits()
    print(f"Data shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    scaler = fit_scaler(X_train)
    X_train_scaled = transform_data(X_train, scaler)
    X_test_scaled  = transform_data(X_test, scaler)
    
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
    
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    results = []

    # 1. Proposed 1D-TCN
    print("🚀 [1/6] Training Proposed 1D-TCN...")
    tcn_model = Proposed1DTCN()
    tcn_model = train_pytorch_model(tcn_model, loader, device)
    torch.save(tcn_model.state_dict(), "models/tcn_model.pth")
    
    tcn_model.eval()
    with torch.no_grad():
        preds_tcn = tcn_model(X_te_t.to(device)).cpu().numpy().flatten()
    mae, rmse, mse = evaluate_model(y_test, preds_tcn)
    lat_tcn = benchmark_inference_latency(tcn_model, X_te_t[0:1], device)
    
    results.append({
        "Model Name": "1D-TCN (Proposed)", "Model Type": "Temporal CNN",
        "MAE (min)": round(mae, 4), "RMSE (min)": round(rmse, 4),
        "MSE": round(mse, 4), "Latency (s)": round(lat_tcn, 6)
    })

    # 2. LSTM Baseline
    print("🚀 [2/6] Training LSTM Baseline...")
    lstm_model = LSTMBaseline()
    lstm_model = train_pytorch_model(lstm_model, loader, device)
    torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
    
    lstm_model.eval()
    with torch.no_grad():
        preds_lstm = lstm_model(X_te_t.to(device)).cpu().numpy().flatten()
    mae, rmse, mse = evaluate_model(y_test, preds_lstm)
    lat_lstm = benchmark_inference_latency(lstm_model, X_te_t[0:1], device)
    
    results.append({
        "Model Name": "LSTM-only", "Model Type": "Recurrent Network",
        "MAE (min)": round(mae, 4), "RMSE (min)": round(rmse, 4),
        "MSE": round(mse, 4), "Latency (s)": round(lat_lstm, 6)
    })

    # 3. 1D-CNN Baseline
    print("🚀 [3/6] Training 1D-CNN Baseline...")
    cnn_model = CNNBaseline()
    cnn_model = train_pytorch_model(cnn_model, loader, device)
    torch.save(cnn_model.state_dict(), "models/cnn_model.pth")
    
    cnn_model.eval()
    with torch.no_grad():
        preds_cnn = cnn_model(X_te_t.to(device)).cpu().numpy().flatten()
    mae, rmse, mse = evaluate_model(y_test, preds_cnn)
    lat_cnn = benchmark_inference_latency(cnn_model, X_te_t[0:1], device)
    
    results.append({
        "Model Name": "CNN-only", "Model Type": "1D Convolutional",
        "MAE (min)": round(mae, 4), "RMSE (min)": round(rmse, 4),
        "MSE": round(mse, 4), "Latency (s)": round(lat_cnn, 6)
    })

    # 4. Density MLP Baseline
    print("🚀 [4/6] Training Density MLP Baseline...")
    mlp_model = MLPBaseline(input_dim=50)
    mlp_model = train_pytorch_model(mlp_model, loader, device)
    torch.save(mlp_model.state_dict(), "models/mlp_model.pth")
    
    mlp_model.eval()
    with torch.no_grad():
        preds_mlp = mlp_model(X_te_t.to(device)).cpu().numpy().flatten()
    mae, rmse, mse = evaluate_model(y_test, preds_mlp)
    lat_mlp = benchmark_inference_latency(mlp_model, X_te_t[0:1], device)
    
    results.append({
        "Model Name": "Density-Feature MLP", "Model Type": "Density MLP",
        "MAE (min)": round(mae, 4), "RMSE (min)": round(rmse, 4),
        "MSE": round(mse, 4), "Latency (s)": round(lat_mlp, 6)
    })

    # 5. Linear Regression Baseline
    print("🚀 [5/6] Fitting Linear Regression Baseline...")
    X_train_flat = X_train_scaled.reshape(len(X_train), -1)
    X_test_flat  = X_test_scaled.reshape(len(X_test), -1)
    
    t0 = time.perf_counter()
    lr_model = LinearRegression().fit(X_train_flat, y_train)
    t1 = time.perf_counter()
    
    preds_lr = lr_model.predict(X_test_flat)
    mae, rmse, mse = evaluate_model(y_test, preds_lr)
    lat_lr = (t1 - t0) / len(X_test_flat) if len(X_test_flat) > 0 else 0.000001
    
    results.append({
        "Model Name": "Linear Regression", "Model Type": "ML Baseline",
        "MAE (min)": round(mae, 4), "RMSE (min)": round(rmse, 4),
        "MSE": round(mse, 4), "Latency (s)": round(lat_lr, 6)
    })

    # 6. Naive Baseline
    print("🚀 [6/6] Computing Naive Mean Baseline...")
    mean_val = float(np.mean(y_train))
    preds_naive = np.full_like(y_test, fill_value=mean_val)
    mae, rmse, mse = evaluate_model(y_test, preds_naive)
    
    results.append({
        "Model Name": "Naive Baseline", "Model Type": "Heuristic Control",
        "MAE (min)": round(mae, 4), "RMSE (min)": round(rmse, 4),
        "MSE": round(mse, 4), "Latency (s)": 0.000000
    })

    df_res = pd.DataFrame(results)
    df_res.to_csv("paper_outputs/table3_prediction.csv", index=False)
    
    print("\n--- COMPLETE BENCHMARK RESULTS (TABLE III) ---")
    print(df_res.to_string(index=False))
    print("\n✅ Results saved to paper_outputs/table3_prediction.csv")

if __name__ == "__main__":
    main()