import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        # x shape: (batch, channels, length)
        out = self.conv(x)
        # Trim off future padding to enforce strict causality
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return self.final_relu(out + res)

class Proposed1DTCN(nn.Module):
    """
    Proposed 1D Temporal Convolutional Network with Causal Residual Blocks
    """
    def __init__(self, in_features=5, hidden_channels=32, num_levels=3):
        super(Proposed1DTCN, self).__init__()
        layers = []
        num_channels = [hidden_channels] * num_levels
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_features if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TCNResidualBlock(in_ch, out_ch, kernel_size=3, dilation=dilation_size))
            
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # Permute input from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = self.global_pool(out).squeeze(-1)
        return self.fc(out)

class LSTMBaseline(nn.Module):
    def __init__(self, in_features=5, hidden_size=64, num_layers=2):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNNBaseline(nn.Module):
    def __init__(self, in_features=5, hidden_channels=32):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.global_pool(out).squeeze(-1)
        return self.fc(out)

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=64):
        super(MLPBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.net(x_flat)