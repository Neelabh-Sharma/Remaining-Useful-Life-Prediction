import torch
import torch.nn as nn

class GRURegressor(nn.Module):
    """
    GRU Regressor: 2 layers, hidden 64, dropout 0.2, linear output.
    Input shape: (batch_size, seq_len=30, num_features)
    Output shape: (batch_size,)
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # Take the last time step output: out[:, -1, :]
        last_out = out[:, -1, :]
        pred = self.fc(last_out).squeeze(-1)
        return pred

if __name__ == "__main__":
    model = GRURegressor(input_size=18)
    dummy_x = torch.randn(8, 30, 18)
    out = model(dummy_x)
    print("Model test output shape:", out.shape)
    assert out.shape == (8,)
    print("model.py verified successfully!")

