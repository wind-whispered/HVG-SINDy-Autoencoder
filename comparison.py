#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparison.py
Comparative analysis of four dynamical modeling approaches
in the autoencoder latent space:
    1. ARIMA  (univariate linear)
    2. VAR    (multivariate linear)
    3. LSTM   (multivariate nonlinear black-box)
    4. SINDy  (multivariate nonlinear interpretable)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from hvg_sindy import preprocess_data

# ============================================================================
# Device
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# LSTM Model
# ============================================================================
class LatentLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3):
        super(LatentLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # take the last time step
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(latent_train, seq_len=10, hidden_dim=64,
               num_layers=2, epochs=500, lr=1e-3):
    """
    Training LSTM models
    latent_train: (T, 3) numpy array
    """
    # Building a sliding window dataset
    X, y = [], []
    for i in range(len(latent_train) - seq_len):
        X.append(latent_train[i:i + seq_len])
        y.append(latent_train[i + seq_len])
    X = np.array(X)  # (N, seq_len, 3)
    y = np.array(y)  # (N, 3)

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LatentLSTM(input_dim=3, hidden_dim=hidden_dim,
                       num_layers=num_layers, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model, seq_len


def predict_lstm(model, latent_series, seq_len, test_len=50):
    """
    LSTM one-step prediction
    """
    model.eval()
    n = len(latent_series)
    test_start = n - test_len

    preds = []
    actuals = []

    with torch.no_grad():
        for i in range(test_start, n - 1):
            x = latent_series[i - seq_len + 1:i + 1]  # (seq_len, 3)
            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
            pred = model(x_tensor).cpu().numpy()[0]
            preds.append(pred)
            actuals.append(latent_series[i + 1])

    return np.array(preds), np.array(actuals)


# ============================================================================
# ARIMA Prediction
# ============================================================================
def predict_arima(latent_series, test_len=50, order=(2, 0, 2)):
    """
    Fit ARIMA to the three dimensions of the latent space separately, and make a one-step prediction.
    """
    n = len(latent_series)
    test_start = n - test_len

    preds = np.zeros((test_len - 1, 3))
    actuals = latent_series[test_start + 1:]

    for dim in range(3):
        ts = latent_series[:, dim]
        for i, t in enumerate(range(test_start, n - 1)):
            train_ts = ts[:t + 1]
            try:
                model = ARIMA(train_ts, order=order)
                fit = model.fit()
                forecast = fit.forecast(steps=1)[0]
            except Exception:
                forecast = train_ts[-1]  # fallback: naive
            preds[i, dim] = forecast

    return preds, actuals


# ============================================================================
# VAR Prediction
# ============================================================================
def predict_var(latent_series, test_len=50, maxlags=5):
    """
    Three-dimensional joint fitting of VAR to latent space for one-step prediction
    """
    n = len(latent_series)
    test_start = n - test_len

    preds = np.zeros((test_len - 1, 3))
    actuals = latent_series[test_start + 1:]

    for i, t in enumerate(range(test_start, n - 1)):
        train_data = latent_series[:t + 1]
        try:
            model = VAR(train_data)
            fit = model.fit(maxlags=maxlags, ic='aic')
            lag_order = fit.k_ar
            forecast_input = train_data[-lag_order:]
            forecast = fit.forecast(forecast_input, steps=1)[0]
        except Exception:
            forecast = train_data[-1]  # fallback: naive
        preds[i] = forecast

    return preds, actuals


# ============================================================================
# SINDy Prediction (from existing models)
# ============================================================================
def predict_sindy(sindy_model, latent_series, test_len=50):
    """
    Predict using the existing SINDy model in one step.
    """
    n = len(latent_series)
    test_start = n - test_len

    X_test = latent_series[test_start:-1]
    actuals = latent_series[test_start + 1:]
    preds = sindy_model.predict(X_test)

    return preds, actuals


# ============================================================================
# NRMSE
# ============================================================================
def compute_nrmse(preds, actuals):
    """
    Calculate the average NRMSE for each dimensionE
    """
    nrmse_list = []
    for dim in range(preds.shape[1]):
        y_true = actuals[:, dim]
        y_pred = preds[:, dim]
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        y_range = np.max(y_true) - np.min(y_true)
        if y_range > 1e-10:
            nrmse_list.append(rmse / y_range * 100)
    return np.mean(nrmse_list)


# ============================================================================
# Main comparison function
# ============================================================================
def run_comparison(processed_data, sindy_models, province_latents,
                   autoencoder, all_labels, output_dir,
                   test_len=50):
    """
    A comparative analysis of the four methods was performed, using only the Mobile channel.
    """
    provinces = ['Beijing', 'Guangdong', 'Henan', 'Shanghai']
    features_per_province = len(next(iter(province_latents.values())))

    comparison_results = {}

    for province in provinces:
        print(f"\nRunning comparison for {province} Mobile...")

        # Obtain the latent spatial sequence of the province
        latent = province_latents[province]  # (T, 3)
        n = len(latent)

        # ---- ARIMA ----
        print(f"  Fitting ARIMA...")
        arima_preds, actuals = predict_arima(latent, test_len=test_len)

        # ---- VAR ----
        print(f"  Fitting VAR...")
        var_preds, _ = predict_var(latent, test_len=test_len)

        # ---- LSTM ----
        print(f"  Training LSTM...")
        seq_len = 10
        train_latent = latent[:n - test_len]
        lstm_model, seq_len = train_lstm(train_latent, seq_len=seq_len)
        lstm_preds, _ = predict_lstm(lstm_model, latent,
                                     seq_len=seq_len, test_len=test_len)

        # ---- SINDy ----
        print(f"  Predicting SINDy...")
        sindy_preds, _ = predict_sindy(sindy_models[province],
                                       latent, test_len=test_len)

        # ---- Calculate NRMSE ----
        nrmse = {
            'ARIMA': compute_nrmse(arima_preds, actuals),
            'VAR': compute_nrmse(var_preds, actuals),
            'LSTM': compute_nrmse(lstm_preds, actuals),
            'SINDy': compute_nrmse(sindy_preds, actuals),
        }
        print(f"  NRMSE: {nrmse}")

        # ---- Calculate residuals ----
        residuals = {
            'ARIMA': actuals - arima_preds,
            'VAR': actuals - var_preds,
            'LSTM': actuals - lstm_preds,
            'SINDy': actuals - sindy_preds,
        }

        comparison_results[province] = {
            'actuals': actuals,
            'arima': arima_preds,
            'var': var_preds,
            'lstm': lstm_preds,
            'sindy': sindy_preds,
            'nrmse': nrmse,
            'residuals': residuals,
        }

    return comparison_results


# ============================================================================
# Plotting
# ============================================================================
def plot_comparison(comparison_results, output_dir, test_len=50):
    """
    2×4 figure:
    Row 1: one-step-ahead prediction curves (4 provinces)
    Row 2: residual boxplots (4 provinces)
    """
    provinces = ['Beijing', 'Guangdong', 'Henan', 'Shanghai']

    method_colors = {
        'Actual': 'black',
        'ARIMA': '#d62728',
        'VAR': '#ff7f0e',
        'LSTM': '#2ca02c',
        'SINDy': '#1F52B4',
    }
    method_styles = {
        'Actual': '-',
        'ARIMA': '--',
        'VAR': '--',
        'LSTM': '--',
        'SINDy': '--',
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    t = np.arange(test_len - 1)

    for col, province in enumerate(provinces):
        res = comparison_results[province]
        actuals = res['actuals']
        arima_p = res['arima']
        var_p = res['var']
        lstm_p = res['lstm']
        sindy_p = res['sindy']
        nrmse = res['nrmse']
        residuals = res['residuals']

        # ---- Row 1: prediction curves (z1 only for clarity) ----
        ax_pred = axes[0, col]

        ax_pred.plot(t, actuals[:, 0], color=method_colors['Actual'],
                     linewidth=1.2, label='Actual', zorder=5)
        ax_pred.plot(t, arima_p[:, 0], color=method_colors['ARIMA'],
                     linewidth=1.0, linestyle='--', alpha=0.85,
                     label=f'ARIMA ({nrmse["ARIMA"]:.1f}%)')
        ax_pred.plot(t, var_p[:, 0], color=method_colors['VAR'],
                     linewidth=1.0, linestyle='--', alpha=0.85,
                     label=f'VAR ({nrmse["VAR"]:.1f}%)')
        ax_pred.plot(t, lstm_p[:, 0], color=method_colors['LSTM'],
                     linewidth=1.0, linestyle='--', alpha=0.85,
                     label=f'LSTM ({nrmse["LSTM"]:.1f}%)')
        ax_pred.plot(t, sindy_p[:, 0], color=method_colors['SINDy'],
                     linewidth=1.0, linestyle='--', alpha=0.85,
                     label=f'SINDy ({nrmse["SINDy"]:.1f}%)')

        ax_pred.set_title(f'({chr(97 + col)}) {province}',
                          fontsize=13, fontweight='bold')
        ax_pred.set_xlabel('Test Time Step', fontsize=11)
        ax_pred.set_ylabel('Latent $z_1$', fontsize=11)
        ax_pred.legend(fontsize=8, loc='upper right')
        ax_pred.grid(True, alpha=0.3)

        # ---- Row 2: residual boxplots ----
        ax_box = axes[1, col]

        # Flattening the residuals in three dimensions
        box_data = [
            res['residuals']['ARIMA'].flatten(),
            res['residuals']['VAR'].flatten(),
            res['residuals']['LSTM'].flatten(),
            res['residuals']['SINDy'].flatten(),
        ]
        method_labels = ['ARIMA', 'VAR', 'LSTM', 'SINDy']
        colors_box = [method_colors[m] for m in method_labels]

        vp = ax_box.violinplot(box_data,
                               positions=range(1, len(method_labels) + 1),
                               showmedians=True,
                               showextrema=True)

        # Set color
        for body, color in zip(vp['bodies'], colors_box):
            body.set_facecolor(color)
            body.set_alpha(0.7)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)

        # Set the colors of the median and extreme value lines.
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(2)
        vp['cmaxes'].set_color('black')
        vp['cmaxes'].set_linewidth(1.2)
        vp['cmins'].set_color('black')
        vp['cmins'].set_linewidth(1.2)
        vp['cbars'].set_color('black')
        vp['cbars'].set_linewidth(1.0)

        ax_box.set_xticks(range(1, len(method_labels) + 1))
        ax_box.set_xticklabels(method_labels, fontsize=10)

        ax_box.axhline(y=0, color='gray', linestyle='--',
                       alpha=0.6, linewidth=1.0)
        ax_box.set_title(f'({chr(101 + col)}) {province} Residuals',
                         fontsize=13, fontweight='bold')
        ax_box.set_ylabel('Residual', fontsize=11)
        ax_box.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Comparative Analysis of Dynamical Modeling Approaches '
        '(Mobile Channel, 50-day Test Period)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure_comparison.pdf',
                bbox_inches='tight')
    plt.close()
    print(f"\nComparison figure saved to {output_dir}/figure_comparison.png/pdf")
