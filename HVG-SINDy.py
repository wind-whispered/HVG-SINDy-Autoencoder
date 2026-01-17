#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVG-SINDy Autoencoder for Taijiquan Public Attention Dynamics and System Evolution Modeling

date: 2026\1\1

This code implements the following functions:
1.Construct a Horizontal Visibility Graph (HVG) and use the HVG degree distribution exponent λ as a criterion for chaos
2.Construct HVG feature vectors from a time series
3.Use an autoencoder to learn low-dimensional evolutionary states
4.Identify the dynamics in the latent space using discrete SINDy
5.Estimate the Lyapunov exponent and plot phase portraits and Poincaré sections
6.Perform analysis in combination with system parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import networkx as nx
from scipy import stats
from scipy.signal import find_peaks
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# set random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Part 1: Data Loading and Preprocessing
# ============================================================================

def load_data(file_paths):
    """
    Load Baidu Index data for four provinces

    Parameters:
    -----------
    file_paths : dict
        Mapping of province names to file paths

    Returns:
    --------
    data_dict : dict
        A dictionary containing data from each province
    """
    data_dict = {}
    for province, path in file_paths.items():
        df = pd.read_excel(path)
        df.columns = ['date', 'PC', 'Mobile']
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.set_index('date')
        data_dict[province] = df
    return data_dict


def preprocess_data(data_dict, smooth_window=7):
    """
    Data preprocessing: smoothing and normalization

    Parameters:
    -----------
    data_dict : dict
        Dictionary of raw data
    smooth_window : int
        Moving average window size

    Returns:
    --------
    processed_dict : dict
        Dictionary of processed data
    """
    processed_dict = {}
    for province, df in data_dict.items():
        # Moving-average smoothing
        df_smooth = df.rolling(window=smooth_window, center=True).mean().dropna()
        # Standardization
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_smooth),
            index=df_smooth.index,
            columns=df_smooth.columns
        )
        processed_dict[province] = {
            'raw': df,
            'smooth': df_smooth,
            'scaled': df_scaled,
            'scaler': scaler
        }
    return processed_dict


# ============================================================================
# Part 2: Horizontal Visibility Graph (HVG)
# ============================================================================

def construct_hvg(time_series):
    """
    Construct a Horizontal Visibility Graph (HVG)

    Two nodes i and j are connected in the HVG if and only if:
    for all i < k < j, satisfy x_k < min(x_i, x_j)

    Parameters:
    -----------
    time_series : array-like
        time series data

    Returns:
    --------
    G : networkx.Graph
        HVG
    """
    n = len(time_series)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        # turn right
        for j in range(i + 1, n):
            # Check visibility conditions
            visible = True
            min_val = min(time_series[i], time_series[j])
            for k in range(i + 1, j):
                if time_series[k] >= min_val:
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)

    return G


def construct_hvg_fast(time_series):
    """
    Uses a divide-and-conquer strategy
    to reduce time complexity

    Parameters:
    -----------
    time_series : array-like
        time series data

    Returns:
    --------
    G : networkx.Graph
        HVG
    """
    n = len(time_series)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []

    def divide_and_conquer(left, right):
        if left >= right:
            return

        # Find the index of the maximum value in the range
        max_idx = left + np.argmax(time_series[left:right + 1])

        # turn left to connect
        for i in range(max_idx - 1, left - 1, -1):
            visible = True
            min_val = min(time_series[i], time_series[max_idx])
            for k in range(i + 1, max_idx):
                if time_series[k] >= min_val:
                    visible = False
                    break
            if visible:
                edges.append((i, max_idx))

        # turn right to connect
        for j in range(max_idx + 1, right + 1):
            visible = True
            min_val = min(time_series[max_idx], time_series[j])
            for k in range(max_idx + 1, j):
                if time_series[k] >= min_val:
                    visible = False
                    break
            if visible:
                edges.append((max_idx, j))

        # Recursively process the left and right subintervals
        divide_and_conquer(left, max_idx - 1)
        divide_and_conquer(max_idx + 1, right)

    divide_and_conquer(0, n - 1)
    G.add_edges_from(edges)

    return G


def calculate_degree_distribution(G):
    """
    Compute the degree distribution of the graph

    Parameters:
    -----------
    G : networkx.Graph

    Returns:
    --------
    degrees : array
        节点度数组
    degree_dist : dict
        度分布字典
    """
    degrees = np.array([d for n, d in G.degree()])
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    # 归一化
    total = sum(degree_counts.values())
    degree_dist = {k: v / total for k, v in degree_counts.items()}

    return degrees, degree_dist


def fit_exponential_distribution(degree_dist):
    """
    拟合度分布的指数衰减: P(k) ~ exp(-λk)
    λ是混沌判据:
    - λ ≈ ln(3/2) ≈ 0.405 表示随机噪声
    - λ < ln(3/2) 可能表示混沌
    - λ > ln(3/2) 可能表示周期性

    Parameters:
    -----------
    degree_dist : dict
        度分布

    Returns:
    --------
    lambda_exp : float
        指数衰减参数
    r_squared : float
        拟合优度
    """
    k_values = np.array(sorted(degree_dist.keys()))
    p_values = np.array([degree_dist[k] for k in k_values])

    # 过滤零值用于对数拟合
    mask = p_values > 0
    k_fit = k_values[mask]
    log_p_fit = np.log(p_values[mask])

    if len(k_fit) < 2:
        return np.nan, 0

    # 线性拟合 log(P(k)) = -λk + c
    slope, intercept, r_value, p_value, std_err = stats.linregress(k_fit, log_p_fit)
    lambda_exp = -slope
    r_squared = r_value ** 2

    return lambda_exp, r_squared


def plot_hvg(G, time_series, title, save_path):
    """
    绘制HVG网络图

    Parameters:
    -----------
    G : networkx.Graph
        水平可见图
    time_series : array-like
        原始时间序列
    title : str
        图标题
    save_path : str
        保存路径(不含扩展名)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上图: 时间序列
    ax1 = axes[0]
    n = len(time_series)
    ax1.plot(range(n), time_series, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'Time Series: {title}', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 下图: HVG网络 (采样显示)
    ax2 = axes[1]
    sample_size = min(200, n)
    sample_indices = np.linspace(0, n - 1, sample_size, dtype=int)
    G_sub = G.subgraph(sample_indices)

    # 使用时间序列值作为y坐标
    pos = {i: (i, time_series[i]) for i in sample_indices}

    nx.draw_networkx_edges(G_sub, pos, ax=ax2, alpha=0.3, edge_color='gray', width=0.5)
    nx.draw_networkx_nodes(G_sub, pos, ax=ax2, node_size=20,
                           node_color='blue', alpha=0.6)

    ax2.set_xlabel('Time Index', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'Horizontal Visibility Graph (Sampled): {title}', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


# def plot_degree_distribution(degree_dist, lambda_exp, r_squared, title, save_path):
#     """
#     绘制度分布及指数拟合
#
#     Parameters:
#     -----------
#     degree_dist : dict
#         度分布
#     lambda_exp : float
#         指数衰减参数
#     r_squared : float
#         拟合优度
#     title : str
#         图标题
#     save_path : str
#         保存路径
#     """
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
#     k_values = np.array(sorted(degree_dist.keys()))
#     p_values = np.array([degree_dist[k] for k in k_values])
#
#     # 左图: 线性坐标
#     ax1 = axes[0]
#     ax1.bar(k_values, p_values, color='steelblue', alpha=0.7, edgecolor='black')
#
#     # 拟合曲线
#     k_fit = np.linspace(k_values.min(), k_values.max(), 100)
#     p_fit = np.exp(-lambda_exp * k_fit) * np.exp(-lambda_exp * k_values.min()) * degree_dist[k_values.min()]
#     ax1.plot(k_fit, p_fit, 'r-', linewidth=2, label=f'Exp fit: λ={lambda_exp:.4f}')
#
#     ax1.set_xlabel('Degree k', fontsize=14)
#     ax1.set_ylabel('P(k)', fontsize=14)
#     ax1.set_title(f'Degree Distribution: {title}', fontsize=16)
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#
#     # 右图: 半对数坐标
#     ax2 = axes[1]
#     mask = p_values > 0
#     ax2.semilogy(k_values[mask], p_values[mask], 'o', markersize=8,
#                  color='steelblue', alpha=0.7, label='Data')
#
#     # 拟合线
#     k0 = k_values.min()
#     p0 = degree_dist[k0]
#     log_p_fit = np.log(p0) - lambda_exp * (k_fit - k0)
#     # log_p_fit = -lambda_exp * k_fit+  np.log(p_values[mask][0]) + lambda_exp * k_values[mask][0]
#     ax2.semilogy(k_fit, np.exp(log_p_fit), 'r-', linewidth=2,
#                  label=f'Linear fit: λ={lambda_exp:.4f}, R²={r_squared:.4f}')
#
#     # 添加随机噪声参考线 λ = ln(3/2)
#     # lambda_random = np.log(3 / 2)
#     # ax2.axhline(y=np.exp(-lambda_random * np.mean(k_values)), color='green',
#     #             linestyle='--', alpha=0.7, label=f'Random: λ=ln(3/2)≈{lambda_random:.3f}')
#     lambda_random = np.log(3 / 2)
#     p_random = p0 * np.exp(-lambda_random * (k_fit - k0))
#     ax2.semilogy(k_fit, p_random, '--', color='green', alpha=0.7,
#                  label=f'Random: λ=ln(3/2)≈{lambda_random:.3f}')
#     ax2.set_xlabel('Degree k', fontsize=14)
#     ax2.set_ylabel('P(k) (log scale)', fontsize=14)
#     ax2.set_title(f'Semi-log Degree Distribution: {title}', fontsize=18)
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     # 添加混沌判据注释
#     chaos_criterion = "Chaotic" if lambda_exp < lambda_random else "Periodic/Random"
#     fig.text(0.5, 0.02, f'Chaos Criterion: λ={lambda_exp:.4f} vs ln(3/2)={lambda_random:.4f} → {chaos_criterion}',
#              ha='center', fontsize=12, style='italic')
#
#     plt.tight_layout(rect=[0, 0.05, 1, 1])
#     plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
#     plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
#     plt.close()


# ============================================================================
# Part 3: HVG Feature Vector Construction
# ============================================================================
def plot_degree_distribution(degree_dist, lambda_exp, r_squared, title, save_path):
    """
    绘制度分布及指数拟合
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if not degree_dist:
        raise ValueError("degree_dist is empty")

    k_values = np.array(sorted(degree_dist.keys()))
    p_values = np.array([degree_dist[k] for k in k_values])

    # 右图需要正值
    mask = p_values > 0
    if not np.any(mask):
        raise ValueError("All p_values are non-positive; cannot use semilogy.")

    # 拟合区间
    k_fit = np.linspace(k_values.min(), k_values.max(), 200)

    # ====== 用log域线性回归同时拟合截距A和lambda ======
    x = k_values[mask].astype(float)
    y = np.log(p_values[mask].astype(float))

    # y = a + b*x  =>  lambda = -b, A = exp(a)
    b, a = np.polyfit(x, y, 1)
    lambda_fit = -b
    A_fit = np.exp(a)
    p_fit = A_fit * np.exp(-lambda_fit * k_fit)

    # 左图: 线性坐标
    ax1 = axes[0]
    ax1.bar(k_values, p_values, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.plot(k_fit, p_fit, 'r-', linewidth=2, label=f'Exp fit: λ={lambda_fit:.4f}')
    ax1.set_xlabel('Degree k', fontsize=14)
    ax1.set_ylabel('P(k)', fontsize=14)
    ax1.set_title(f'Degree Distribution: {title}', fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 半对数坐标
    ax2 = axes[1]
    ax2.semilogy(k_values[mask], p_values[mask], 'o', markersize=8,
                 color='steelblue', alpha=0.7, label='Data')
    ax2.semilogy(k_fit, p_fit, 'r-', linewidth=2,
                 label=f'Linear fit: λ={lambda_fit:.4f}, R²={r_squared:.4f}')

    # ====== 随机参考线也画成指数曲线，并在中位k处对齐 ======
    lambda_random = np.log(3 / 2)

    k_ref = float(np.median(x))
    # 在log域插值，数值更稳
    log_p_ref = float(np.interp(k_ref, x, y))
    p_ref = np.exp(log_p_ref)

    A_random = p_ref * np.exp(lambda_random * k_ref)
    p_random = A_random * np.exp(-lambda_random * k_fit)

    ax2.semilogy(k_fit, p_random, color='green', linestyle='--', alpha=0.7,
                 label=f'Random: λ=ln(3/2)≈{lambda_random:.3f}')

    ax2.set_xlabel('Degree k', fontsize=14)
    ax2.set_ylabel('P(k) (log scale)', fontsize=14)
    ax2.set_title(f'Semi-log Degree Distribution: {title}', fontsize=18)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 混沌判据（这里用拟合出的lambda_fit更一致）
    chaos_criterion = "Chaotic" if lambda_fit < lambda_random else "Periodic/Random"
    fig.text(0.5, 0.02,
             f'Chaos Criterion: λ={lambda_fit:.4f} vs ln(3/2)={lambda_random:.4f} → {chaos_criterion}',
             ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


def extract_hvg_features(G, time_series, window_size=50, stride=25):
    """
    从HVG中提取特征向量

    使用滑动窗口方法提取局部HVG特征:
    - 局部度分布统计
    - 聚类系数
    - 平均路径长度
    - 网络密度

    Parameters:
    -----------
    G : networkx.Graph
        完整的HVG
    time_series : array-like
        时间序列
    window_size : int
        窗口大小
    stride : int
        滑动步长

    Returns:
    --------
    features : np.ndarray
        特征矩阵 (n_windows, n_features)
    feature_names : list
        特征名称列表
    """
    n = len(time_series)
    features_list = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size

        # 提取子图
        nodes = list(range(start, end))
        G_sub = G.subgraph(nodes)

        # 计算局部特征
        degrees = np.array([d for n, d in G_sub.degree()])

        # 度统计
        degree_mean = np.mean(degrees)
        degree_std = np.std(degrees)
        degree_max = np.max(degrees)
        degree_min = np.min(degrees)
        degree_skew = stats.skew(degrees) if len(degrees) > 2 else 0
        degree_kurtosis = stats.kurtosis(degrees) if len(degrees) > 3 else 0

        # 聚类系数
        clustering = nx.average_clustering(G_sub) if G_sub.number_of_edges() > 0 else 0

        # 网络密度
        density = nx.density(G_sub)

        # 局部λ估计
        _, degree_dist = calculate_degree_distribution(G_sub)
        lambda_local, _ = fit_exponential_distribution(degree_dist)
        if np.isnan(lambda_local):
            lambda_local = 0

        # 时间序列统计
        ts_window = time_series[start:end]
        ts_mean = np.mean(ts_window)
        ts_std = np.std(ts_window)
        ts_trend = np.polyfit(range(window_size), ts_window, 1)[0]

        # 组合特征
        features = [
            degree_mean, degree_std, degree_max, degree_min,
            degree_skew, degree_kurtosis,
            clustering, density, lambda_local,
            ts_mean, ts_std, ts_trend
        ]
        features_list.append(features)

    feature_names = [
        'degree_mean', 'degree_std', 'degree_max', 'degree_min',
        'degree_skew', 'degree_kurtosis',
        'clustering', 'density', 'lambda_local',
        'ts_mean', 'ts_std', 'ts_trend'
    ]

    return np.array(features_list), feature_names


def extract_hvg_node_features(G, time_series, k_neighbors=5):
    """
    为每个节点提取局部HVG特征

    Parameters:
    -----------
    G : networkx.Graph
        HVG图
    time_series : array-like
        时间序列
    k_neighbors : int
        邻域大小

    Returns:
    --------
    node_features : np.ndarray
        节点特征矩阵
    """
    n = len(time_series)
    node_features = []

    for i in range(n):
        # 节点度
        degree = G.degree(i)

        # 邻居节点的度
        neighbors = list(G.neighbors(i))
        if len(neighbors) > 0:
            neighbor_degrees = [G.degree(n) for n in neighbors]
            neighbor_degree_mean = np.mean(neighbor_degrees)
            neighbor_degree_std = np.std(neighbor_degrees)
        else:
            neighbor_degree_mean = 0
            neighbor_degree_std = 0

        # 局部聚类系数
        local_clustering = nx.clustering(G, i)

        # 时间序列值
        ts_value = time_series[i]

        # 局部差分
        left_diff = time_series[i] - time_series[i - 1] if i > 0 else 0
        right_diff = time_series[i + 1] - time_series[i] if i < n - 1 else 0

        features = [
            degree, neighbor_degree_mean, neighbor_degree_std,
            local_clustering, ts_value, left_diff, right_diff
        ]
        node_features.append(features)

    return np.array(node_features)


# ============================================================================
# Part 4: Autoencoder Models
# ============================================================================

class HVGAutoencoder(nn.Module):
    """
    用于学习HVG特征低维表示的Autoencoder

    架构:
    - Encoder: 多层MLP，逐渐降维到潜空间
    - Decoder: 对称的MLP，从潜空间重构
    """

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], latent_dim=3):
        """
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        hidden_dims : list
            隐藏层维度列表
        latent_dim : int
            潜空间维度
        """
        super(HVGAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 构建Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 构建Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """编码到潜空间"""
        return self.encoder(x)

    def decode(self, z):
        """从潜空间解码"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class VariationalHVGAutoencoder(nn.Module):
    """
    变分Autoencoder (VAE) 用于学习更平滑的潜空间
    """

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], latent_dim=3):
        super(VariationalHVGAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 均值和方差层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE损失函数: 重构损失 + KL散度
    """
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_autoencoder(model, train_loader, val_loader=None, epochs=2000,
                      lr=1e-4, weight_decay=1e-5, is_vae=False):
    """
    训练Autoencoder模型

    Parameters:
    -----------
    model : nn.Module
        Autoencoder模型
    train_loader : DataLoader
        训练数据加载器
    val_loader : DataLoader
        验证数据加载器
    epochs : int
        训练轮数
    lr : float
        学习率
    weight_decay : float
        权重衰减
    is_vae : bool
        是否为VAE模型

    Returns:
    --------
    model : nn.Module
        训练后的模型
    history : dict
        训练历史
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            if is_vae:
                x_recon, z, mu, logvar = model(batch_x)
                loss = vae_loss(x_recon, batch_x, mu, logvar)
            else:
                x_recon, z = model(batch_x)
                loss = nn.functional.mse_loss(x_recon, batch_x, reduction='sum')

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, in val_loader:
                    batch_x = batch_x.to(device)
                    if is_vae:
                        x_recon, z, mu, logvar = model(batch_x)
                        loss = vae_loss(x_recon, batch_x, mu, logvar)
                    else:
                        x_recon, z = model(batch_x)
                        loss = nn.functional.mse_loss(x_recon, batch_x, reduction='sum')
                    val_loss += loss.item()

            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
        else:
            scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            val_str = f", Val Loss: {history['val_loss'][-1]:.6f}" if val_loader else ""
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}{val_str}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def plot_training_history(history, save_path):
    """
    绘制训练历史
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)

    if history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Autoencoder Training History', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


def plot_latent_space(latent_vectors, labels=None, title='Latent Space', save_path='latent_space'):
    """
    绘制潜空间分布

    Parameters:
    -----------
    latent_vectors : np.ndarray
        潜空间向量 (n_samples, latent_dim)
    labels : array-like
        样本标签 (用于着色)
    title : str
        图标题
    save_path : str
        保存路径
    """
    latent_dim = latent_vectors.shape[1]

    if latent_dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1],
                             c=labels if labels is not None else range(len(latent_vectors)),
                             cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        plt.colorbar(scatter, label='Time Index' if labels is None else 'Label')

    elif latent_dim >= 3:
        fig = plt.figure(figsize=(14, 5))

        # 3D图
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2],
                              c=labels if labels is not None else range(len(latent_vectors)),
                              cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel('Latent Dim 1')
        ax1.set_ylabel('Latent Dim 2')
        ax1.set_zlabel('Latent Dim 3')
        ax1.set_title('3D Latent Space')

        # 2D投影 (前两个维度)
        ax2 = fig.add_subplot(122)
        scatter2 = ax2.scatter(latent_vectors[:, 0], latent_vectors[:, 1],
                               c=labels if labels is not None else range(len(latent_vectors)),
                               cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('Latent Dim 1', fontsize=12)
        ax2.set_ylabel('Latent Dim 2', fontsize=12)
        ax2.set_title('2D Projection')
        plt.colorbar(scatter2, label='Time Index' if labels is None else 'Label')

    else:  # 1D
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(latent_vectors[:, 0], 'b-', linewidth=0.5)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Latent Value', fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# art 5: SINDy (Sparse Identification of Nonlinear Dynamics)
# ============================================================================

class SINDyLibrary:
    """
    SINDy候选函数库

    生成多项式和三角函数候选项
    """

    def __init__(self, poly_order=2, include_sine=True, include_interaction=True):
        """
        Parameters:
        -----------
        poly_order : int
            多项式最高阶数
        include_sine : bool
            是否包含三角函数
        include_interaction : bool
            是否包含交互项
        """
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.include_interaction = include_interaction
        self.feature_names = []

    def fit_transform(self, X):
        """
        构建候选函数库

        Parameters:
        -----------
        X : np.ndarray
            状态变量矩阵 (n_samples, n_features)

        Returns:
        --------
        Theta : np.ndarray
            候选函数库矩阵
        """
        n_samples, n_features = X.shape
        self.feature_names = []

        # 常数项
        Theta = [np.ones(n_samples)]
        self.feature_names.append('1')

        # 多项式项
        for order in range(1, self.poly_order + 1):
            for i in range(n_features):
                Theta.append(X[:, i] ** order)
                self.feature_names.append(f'x{i + 1}^{order}' if order > 1 else f'x{i + 1}')

        # 交互项
        if self.include_interaction:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    Theta.append(X[:, i] * X[:, j])
                    self.feature_names.append(f'x{i + 1}*x{j + 1}')

        # 三角函数
        if self.include_sine:
            for i in range(n_features):
                Theta.append(np.sin(X[:, i]))
                self.feature_names.append(f'sin(x{i + 1})')
                Theta.append(np.cos(X[:, i]))
                self.feature_names.append(f'cos(x{i + 1})')

        return np.column_stack(Theta)


class DiscreteSINDy:
    """
    离散时间SINDy算法

    识别离散动力学: x(t+1) = f(x(t))
    """

    def __init__(self, library, threshold=0.1, max_iter=20, alpha=0.1):
        """
        Parameters:
        -----------
        library : SINDyLibrary
            候选函数库
        threshold : float
            稀疏化阈值
        max_iter : int
            最大迭代次数
        alpha : float
            岭回归正则化参数
        """
        self.library = library
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.coefficients = None

    def fit(self, X):
        """
        拟合离散SINDy模型

        Parameters:
        -----------
        X : np.ndarray
            状态序列 (n_samples, n_features)

        Returns:
        --------
        self
        """
        # 构建输入输出对
        X_in = X[:-1]  # x(t)
        X_out = X[1:]  # x(t+1)

        # 构建候选函数库
        Theta = self.library.fit_transform(X_in)

        n_features = X.shape[1]
        n_library = Theta.shape[1]

        # 对每个输出维度进行稀疏回归
        self.coefficients = np.zeros((n_features, n_library))

        for i in range(n_features):
            y = X_out[:, i]
            xi = self._sparse_regression(Theta, y)
            self.coefficients[i] = xi

        return self

    def _sparse_regression(self, Theta, y):
        """
        序贯阈值最小二乘法 (STLS)
        """
        n_library = Theta.shape[1]
        xi = np.zeros(n_library)

        # 初始最小二乘解
        xi = np.linalg.lstsq(
            Theta.T @ Theta + self.alpha * np.eye(n_library),
            Theta.T @ y,
            rcond=None
        )[0]

        # 迭代稀疏化
        for _ in range(self.max_iter):
            # 阈值化
            small_inds = np.abs(xi) < self.threshold
            xi[small_inds] = 0

            # 在非零位置重新拟合
            big_inds = ~small_inds
            if np.sum(big_inds) == 0:
                break

            Theta_reduced = Theta[:, big_inds]
            xi[big_inds] = np.linalg.lstsq(
                Theta_reduced.T @ Theta_reduced + self.alpha * np.eye(np.sum(big_inds)),
                Theta_reduced.T @ y,
                rcond=None
            )[0]

        return xi

    def predict(self, X):
        """
        预测下一时刻状态
        """
        Theta = self.library.fit_transform(X)
        return Theta @ self.coefficients.T

    def simulate(self, x0, n_steps):
        """
        模拟动力学演化
        """
        trajectory = [x0]
        x = x0.copy()

        for _ in range(n_steps - 1):
            Theta = self.library.fit_transform(x.reshape(1, -1))
            x_next = (Theta @ self.coefficients.T).flatten()
            trajectory.append(x_next)
            x = x_next

        return np.array(trajectory)

    def get_equations(self):
        """
        获取识别的方程
        """
        equations = []
        feature_names = self.library.feature_names
        n_features = self.coefficients.shape[0]

        for i in range(n_features):
            terms = []
            for j, name in enumerate(feature_names):
                coef = self.coefficients[i, j]
                if np.abs(coef) > 1e-10:
                    if coef > 0 and len(terms) > 0:
                        terms.append(f'+ {coef:.4f}*{name}')
                    else:
                        terms.append(f'{coef:.4f}*{name}')

            eq = f'x{i + 1}(t+1) = ' + ' '.join(terms) if terms else f'x{i + 1}(t+1) = 0'
            equations.append(eq)

        return equations


def plot_sindy_results(sindy_model, X, save_path):
    """
    绘制SINDy识别结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_features = X.shape[1]
    n_steps = X.shape[0]

    # 系数矩阵热力图
    ax1 = axes[0, 0]
    coef_matrix = sindy_model.coefficients
    im = ax1.imshow(np.abs(coef_matrix), aspect='auto', cmap='hot')
    ax1.set_xlabel('Library Functions', fontsize=18)
    ax1.set_ylabel('State Variables', fontsize=18)
    ax1.set_title('SINDy Coefficient Matrix (|ξ|)', fontsize=18)
    plt.colorbar(im, ax=ax1)

    # 预测对比
    ax2 = axes[0, 1]
    X_pred = sindy_model.predict(X[:-1])
    for i in range(min(n_features, 3)):
        ax2.scatter(X[1:, i], X_pred[:, i], alpha=0.5, s=10, label=f'Dim {i + 1}')

    # 添加对角线
    all_vals = np.concatenate([X[1:].flatten(), X_pred.flatten()])
    min_val, max_val = all_vals.min(), all_vals.max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('True x(t+1)', fontsize=12)
    ax2.set_ylabel('Predicted x(t+1)', fontsize=12)
    ax2.set_title('Prediction vs Ground Truth', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 时间序列对比
    ax3 = axes[1, 0]
    t = np.arange(n_steps)
    for i in range(min(n_features, 3)):
        ax3.plot(t, X[:, i], '-', linewidth=1, label=f'True Dim {i + 1}')
        # 模拟
        x0 = X[0]
        X_sim = sindy_model.simulate(x0, n_steps)
        ax3.plot(t, X_sim[:, i], '--', linewidth=1, label=f'Simulated Dim {i + 1}')

    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Time Series: True vs Simulated', fontsize=14)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 残差分布
    ax4 = axes[1, 1]
    residuals = X[1:] - X_pred
    for i in range(min(n_features, 3)):
        ax4.hist(residuals[:, i], bins=50, alpha=0.5, label=f'Dim {i + 1}')

    ax4.set_xlabel('Residual', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Prediction Residuals Distribution', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# Part 6: Dynamical Analysis - Lyapunov exponent, phase portrait, Poincaré section
# ============================================================================

def estimate_lyapunov_exponent(time_series, embedding_dim=3, time_delay=1,
                               min_separation=10, max_iterations=100):
    """
    使用Wolf算法估计最大Lyapunov指数

    Parameters:
    -----------
    time_series : array-like
        时间序列数据
    embedding_dim : int
        嵌入维度
    time_delay : int
        时间延迟
    min_separation : int
        最小分离时间
    max_iterations : int
        最大迭代次数

    Returns:
    --------
    lyap_exp : float
        最大Lyapunov指数估计值
    lyap_trajectory : array
        Lyapunov指数随时间的变化
    """
    # 构建相空间嵌入
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    # 计算距离矩阵
    distances = squareform(pdist(embedded))

    # Wolf算法
    sum_lyap = 0
    count = 0
    lyap_trajectory = []

    current_idx = 0

    while current_idx < n_embedded - min_separation and count < max_iterations:
        # 找最近邻 (排除时间上接近的点)
        dists = distances[current_idx].copy()

        # 排除自己和时间上接近的点
        for i in range(max(0, current_idx - min_separation),
                       min(n_embedded, current_idx + min_separation + 1)):
            dists[i] = np.inf

        if np.all(np.isinf(dists)):
            break

        nearest_idx = np.argmin(dists)
        initial_dist = dists[nearest_idx]

        if initial_dist < 1e-10:
            current_idx += 1
            continue

        # 演化
        evolution_time = min_separation
        if current_idx + evolution_time >= n_embedded or nearest_idx + evolution_time >= n_embedded:
            break

        final_dist = np.linalg.norm(
            embedded[current_idx + evolution_time] - embedded[nearest_idx + evolution_time]
        )

        if final_dist > 1e-10:
            sum_lyap += np.log(final_dist / initial_dist)
            count += 1
            lyap_trajectory.append(sum_lyap / count / evolution_time)

        current_idx += evolution_time

    if count > 0:
        lyap_exp = sum_lyap / count / min_separation
    else:
        lyap_exp = 0

    return lyap_exp, np.array(lyap_trajectory)


def estimate_lyapunov_rosenstein(time_series, embedding_dim=3, time_delay=1,
                                 mean_period=10):
    """
    使用Rosenstein方法估计最大Lyapunov指数

    Parameters:
    -----------
    time_series : array-like
        时间序列
    embedding_dim : int
        嵌入维度
    time_delay : int
        时间延迟
    mean_period : int
        平均周期

    Returns:
    --------
    lyap_exp : float
        Lyapunov指数
    divergence : array
        对数散度随时间的变化
    """
    # 相空间嵌入
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    # 计算距离矩阵
    distances = squareform(pdist(embedded))

    # 对每个点找最近邻
    divergence_list = []
    max_j = min(n_embedded // 4, 200)

    for i in range(n_embedded - max_j):
        # 排除时间上接近的点找最近邻
        dists = distances[i].copy()
        for k in range(max(0, i - mean_period), min(n_embedded, i + mean_period + 1)):
            dists[k] = np.inf

        if np.all(np.isinf(dists)):
            continue

        nearest = np.argmin(dists)

        # 计算散度随时间的变化
        div_i = []
        for j in range(max_j):
            if i + j < n_embedded and nearest + j < n_embedded:
                d = np.linalg.norm(embedded[i + j] - embedded[nearest + j])
                if d > 1e-10:
                    div_i.append(np.log(d))
                else:
                    div_i.append(np.nan)

        if len(div_i) == max_j:
            divergence_list.append(div_i)

    if len(divergence_list) == 0:
        return 0, np.array([])

    # 平均散度
    divergence = np.nanmean(divergence_list, axis=0)

    # 线性拟合估计Lyapunov指数
    valid_idx = ~np.isnan(divergence)
    if np.sum(valid_idx) < 10:
        return 0, divergence

    t = np.arange(len(divergence))
    # 取线性增长区域
    fit_range = min(50, len(divergence) // 2)
    slope, _, _, _, _ = stats.linregress(t[:fit_range], divergence[:fit_range])

    return slope, divergence


def plot_phase_portrait(time_series, embedding_dim=3, time_delay=1,
                        title='Phase Portrait', save_path='phase_portrait'):
    """
    绘制相图

    Parameters:
    -----------
    time_series : array-like
        时间序列
    embedding_dim : int
        嵌入维度
    time_delay : int
        时间延迟
    title : str
        图标题
    save_path : str
        保存路径
    """
    # 相空间嵌入
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    if embedding_dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                             c=np.arange(n_embedded), cmap='viridis',
                             s=5, alpha=0.6)
        ax.plot(embedded[:, 0], embedded[:, 1], 'b-', alpha=0.1, linewidth=0.5)

        ax.set_xlabel(f'x(t)', fontsize=14)
        ax.set_ylabel(f'x(t+{time_delay})', fontsize=14)
        ax.set_title(f'{title} (2D)', fontsize=16)
        plt.colorbar(scatter, label='Time Index')

    else:  # 3D
        fig = plt.figure(figsize=(14, 6))

        # 3D相图
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                              c=np.arange(n_embedded), cmap='viridis', s=3, alpha=0.6)
        ax1.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                 'b-', alpha=0.1, linewidth=0.3)

        ax1.set_xlabel(f'x(t)', fontsize=10)
        ax1.set_ylabel(f'x(t+{time_delay})', fontsize=10)
        ax1.set_zlabel(f'x(t+{2 * time_delay})', fontsize=10)
        ax1.set_title(f'{title} (3D)', fontsize=14)

        # 2D投影
        ax2 = fig.add_subplot(122)
        scatter2 = ax2.scatter(embedded[:, 0], embedded[:, 1],
                               c=np.arange(n_embedded), cmap='viridis', s=5, alpha=0.6)
        ax2.plot(embedded[:, 0], embedded[:, 1], 'b-', alpha=0.1, linewidth=0.5)

        ax2.set_xlabel(f'x(t)', fontsize=12)
        ax2.set_ylabel(f'x(t+{time_delay})', fontsize=12)
        ax2.set_title(f'{title} (2D Projection)', fontsize=14)
        plt.colorbar(scatter2, label='Time Index')

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


def plot_poincare_section(time_series, embedding_dim=3, time_delay=1,
                          section_dim=0, section_value=None,
                          title='Poincaré Section', save_path='poincare_section'):
    """
    绘制Poincaré截面

    Parameters:
    -----------
    time_series : array-like
        时间序列
    embedding_dim : int
        嵌入维度
    time_delay : int
        时间延迟
    section_dim : int
        截面所在维度
    section_value : float
        截面值 (默认为中值)
    title : str
        图标题
    save_path : str
        保存路径
    """
    # 相空间嵌入
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    # 确定截面值
    if section_value is None:
        section_value = np.median(embedded[:, section_dim])

    # 找穿越截面的点 (正向穿越)
    crossings = []
    for i in range(n_embedded - 1):
        if (embedded[i, section_dim] < section_value and
                embedded[i + 1, section_dim] >= section_value):
            # 线性插值
            t_cross = (section_value - embedded[i, section_dim]) / \
                      (embedded[i + 1, section_dim] - embedded[i, section_dim])

            cross_point = embedded[i] + t_cross * (embedded[i + 1] - embedded[i])
            crossings.append(cross_point)

    crossings = np.array(crossings)

    if len(crossings) < 10:
        print(f"Warning: Only {len(crossings)} crossings found. Try different section parameters.")
        return

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 获取非截面维度
    other_dims = [i for i in range(embedding_dim) if i != section_dim]

    # Poincaré截面
    ax1 = axes[0]
    if len(other_dims) >= 2:
        ax1.scatter(crossings[:, other_dims[0]], crossings[:, other_dims[1]],
                    s=20, alpha=0.6, c='blue')
        ax1.set_xlabel(f'Dim {other_dims[0] + 1}', fontsize=12)
        ax1.set_ylabel(f'Dim {other_dims[1] + 1}', fontsize=12)
    else:
        ax1.scatter(range(len(crossings)), crossings[:, other_dims[0]],
                    s=20, alpha=0.6, c='blue')
        ax1.set_xlabel('Crossing Index', fontsize=12)
        ax1.set_ylabel(f'Dim {other_dims[0] + 1}', fontsize=12)

    ax1.set_title(f'{title}\n(Section at Dim {section_dim + 1} = {section_value:.2f})', fontsize=14)
    ax1.grid(True, alpha=0.3)  # 回归映射 (连续穿越点之间的关系)
    ax2 = axes[1]
    if len(crossings) > 1 and len(other_dims) >= 1:
        ax2.scatter(crossings[:-1, other_dims[0]], crossings[1:, other_dims[0]],
                    s=20, alpha=0.6, c='red')
        ax2.plot([crossings[:, other_dims[0]].min(), crossings[:, other_dims[0]].max()],
                 [crossings[:, other_dims[0]].min(), crossings[:, other_dims[0]].max()],
                 'k--', alpha=0.5)
        ax2.set_xlabel(f'x_n (Dim {other_dims[0] + 1})', fontsize=12)
        ax2.set_ylabel(f'x_{{n+1}} (Dim {other_dims[0] + 1})', fontsize=12)
        ax2.set_title('Return Map', fontsize=14)
        ax2.grid(True, alpha=0.3)

    # 回归映射 (连续穿越点之间的关系)
    ax2 = axes[1]
    if len(crossings) > 1 and len(other_dims) >= 1:
        ax2.scatter(crossings[:-1, other_dims[0]], crossings[1:, other_dims[0]],
                    s=20, alpha=0.6, c='red')
        ax2.plot([crossings[:, other_dims[0]].min(), crossings[:, other_dims[0]].max()],
                 [crossings[:, other_dims[0]].min(), crossings[:, other_dims[0]].max()],
                 'k--', alpha=0.5)
        ax2.set_xlabel(f'x_n (Dim {other_dims[0] + 1})', fontsize=12)
        ax2.set_ylabel(f'x_{{n+1}} (Dim {other_dims[0] + 1})', fontsize=12)
        ax2.set_title('Return Map', fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


def plot_lyapunov_analysis(time_series, embedding_dim=3, time_delay=1,
                           title='Lyapunov Analysis', save_path='lyapunov'):
    """
    Lyapunov指数分析的综合绘图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 估计Lyapunov指数 (Rosenstein方法)
    lyap_exp, divergence = estimate_lyapunov_rosenstein(
        time_series, embedding_dim, time_delay
    )

    # Wolf方法
    lyap_wolf, lyap_traj = estimate_lyapunov_exponent(
        time_series, embedding_dim, time_delay
    )

    # 散度曲线
    ax1 = axes[0, 0]
    if len(divergence) > 0:
        ax1.plot(divergence, 'b-', linewidth=1)
        # 拟合线
        fit_range = min(50, len(divergence) // 2)
        t = np.arange(fit_range)
        ax1.plot(t, lyap_exp * t + divergence[0], 'r--', linewidth=2,
                 label=f'λ (Rosenstein) = {lyap_exp:.4f}')
        ax1.set_xlabel('Time Steps', fontsize=12)
        ax1.set_ylabel('⟨ln(divergence)⟩', fontsize=12)
        ax1.set_title('Average Divergence', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Wolf方法轨迹
    ax2 = axes[0, 1]
    if len(lyap_traj) > 0:
        ax2.plot(lyap_traj, 'g-', linewidth=1)
        ax2.axhline(y=lyap_wolf, color='r', linestyle='--',
                    label=f'λ (Wolf) = {lyap_wolf:.4f}')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Lyapunov Exponent', fontsize=12)
        ax2.set_title('Wolf Method Convergence', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 相空间轨迹
    ax3 = axes[1, 0]
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay
    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    ax3.scatter(embedded[:, 0], embedded[:, 1], c=np.arange(n_embedded),
                cmap='viridis', s=3, alpha=0.5)
    ax3.set_xlabel(f'x(t)', fontsize=12)
    ax3.set_ylabel(f'x(t+{time_delay})', fontsize=12)
    ax3.set_title('Phase Space (2D Projection)', fontsize=14)

    # 混沌判断总结
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    Lyapunov Exponent Analysis Summary
    {'=' * 40}

    Rosenstein Method: λ = {lyap_exp:.4f}
    Wolf Method: λ = {lyap_wolf:.4f}

    Interpretation:
    • λ > 0: Chaotic behavior (sensitive to initial conditions)
    • λ ≈ 0: Quasi-periodic or at edge of chaos
    • λ < 0: Stable fixed point or periodic orbit

    Result: {'Chaotic' if lyap_exp > 0 or lyap_wolf > 0 else 'Non-chaotic'}

    Embedding Parameters:
    • Dimension: {embedding_dim}
    • Time Delay: {time_delay}
    """

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()

    return lyap_exp, lyap_wolf


# ============================================================================
# Part 7: Comprehensive Analysis and Parameter Study
# ============================================================================

def comprehensive_analysis(data_dict, output_dir='./results'):
    """
    综合分析流程

    Parameters:
    -----------
    data_dict : dict
        数据字典
    output_dir : str
        输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 存储分析结果
    results = {
        'hvg_lambda': {},
        'lyapunov': {},
        'sindy_equations': {},
        'hvg_degree_dist': {},  # 存储度分布数据用于Figure 3
        'lyapunov_divergence': {}  # 存储Lyapunov散度曲线数据用于Figure 3
    }

    # 预处理数据
    processed_data = preprocess_data(data_dict)

    all_features = []
    all_labels = []
    province_list = list(processed_data.keys())

    print("=" * 60)
    print("Part 1 & 2: HVG Analysis and Feature Extraction")
    print("=" * 60)

    for idx, (province, data) in enumerate(processed_data.items()):
        print(f"\nAnalyzing {province}...")

        for component in ['PC', 'Mobile']:
            ts = data['smooth'][component].values
            ts_name = f'{province}_{component}'

            # 构建HVG
            print(f"  Building HVG for {component}...")
            G = construct_hvg_fast(ts)

            # 度分布分析
            degrees, degree_dist = calculate_degree_distribution(G)
            lambda_exp, r_squared = fit_exponential_distribution(degree_dist)
            results['hvg_lambda'][ts_name] = {
                'lambda': lambda_exp,
                'r_squared': r_squared
            }
            # 存储度分布数据用于Figure 3
            results['hvg_degree_dist'][ts_name] = degree_dist

            # 绘制HVG
            plot_hvg(G, ts, ts_name, f'{output_dir}/hvg_{ts_name}')

            # 绘制度分布
            plot_degree_distribution(degree_dist, lambda_exp, r_squared,
                                     ts_name, f'{output_dir}/degree_dist_{ts_name}')

            # 提取HVG特征
            features, feature_names = extract_hvg_features(G, ts)
            all_features.append(features)
            all_labels.extend([idx] * len(features))

            print(f"    λ = {lambda_exp:.4f}, R² = {r_squared:.4f}")

    # 合并所有特征
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    # 标准化
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    print("\n" + "=" * 60)
    print("Part 3: Autoencoder Training")
    print("=" * 60)

    # 准备数据
    X_tensor = torch.FloatTensor(all_features_scaled)
    dataset = TensorDataset(X_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 训练标准Autoencoder
    print("\nTraining Standard Autoencoder...")
    input_dim = all_features_scaled.shape[1]
    autoencoder = HVGAutoencoder(input_dim, hidden_dims=[64, 32, 16], latent_dim=3)
    autoencoder, history = train_autoencoder(
        autoencoder, train_loader, val_loader,
        epochs=2000, lr=1e-4
    )

    plot_training_history(history, f'{output_dir}/ae_training_history')

    # 训练VAE
    print("\nTraining Variational Autoencoder...")
    vae = VariationalHVGAutoencoder(input_dim, hidden_dims=[64, 32, 16], latent_dim=3)
    vae, vae_history = train_autoencoder(
        vae, train_loader, val_loader,
        epochs=2000, lr=1e-4, is_vae=True
    )

    plot_training_history(vae_history, f'{output_dir}/vae_training_history')

    # 提取潜空间表示
    autoencoder.eval()
    vae.eval()

    with torch.no_grad():
        X_tensor_device = X_tensor.to(device)
        _, latent_ae = autoencoder(X_tensor_device)
        latent_ae = latent_ae.cpu().numpy()

        mu, _ = vae.encode(X_tensor_device)
        latent_vae = mu.cpu().numpy()

    # 绘制潜空间
    plot_latent_space(latent_ae, all_labels, 'AE Latent Space',
                      f'{output_dir}/latent_space_ae')
    plot_latent_space(latent_vae, all_labels, 'VAE Latent Space',
                      f'{output_dir}/latent_space_vae')

    print("\n" + "=" * 60)
    print("Part 4: SINDy Dynamics Identification")
    print("=" * 60)

    # 对每个省份的潜空间轨迹应用SINDy
    features_per_province = len(all_features) // len(province_list)

    # 存储SINDy模型和省份潜空间数据用于Figure 4
    sindy_models = {}
    province_latents = {}

    for idx, province in enumerate(province_list):
        print(f"\nIdentifying dynamics for {province}...")

        start_idx = idx * features_per_province
        end_idx = (idx + 1) * features_per_province
        latent_province = latent_ae[start_idx:end_idx]

        # 存储省份潜空间数据
        province_latents[province] = latent_province

        # SINDy
        library = SINDyLibrary(poly_order=2, include_sine=True, include_interaction=True)
        sindy = DiscreteSINDy(library, threshold=0.05, max_iter=20)
        sindy.fit(latent_province)

        # 存储SINDy模型
        sindy_models[province] = sindy

        equations = sindy.get_equations()
        results['sindy_equations'][province] = equations

        print(f"  Identified equations:")
        for eq in equations:
            print(f"    {eq}")

        # 绘制SINDy结果
        plot_sindy_results(sindy, latent_province, f'{output_dir}/sindy_{province}')

    print("\n" + "=" * 60)
    print("Part 5: Dynamical Analysis (Lyapunov, Phase, Poincaré)")
    print("=" * 60)

    for province, data in processed_data.items():
        print(f"\nDynamical analysis for {province}...")

        for component in ['PC', 'Mobile']:
            ts = data['smooth'][component].values
            ts_name = f'{province}_{component}'

            # Lyapunov分析
            lyap_r, lyap_w = plot_lyapunov_analysis(
                ts, embedding_dim=3, time_delay=7,
                title=f'Lyapunov Analysis: {ts_name}',
                save_path=f'{output_dir}/lyapunov_{ts_name}'
            )
            results['lyapunov'][ts_name] = {'rosenstein': lyap_r, 'wolf': lyap_w}

            # 额外计算并存储Rosenstein散度曲线用于Figure 3
            _, divergence = estimate_lyapunov_rosenstein(
                ts, embedding_dim=3, time_delay=7
            )
            results['lyapunov_divergence'][ts_name] = divergence

            # 相图
            plot_phase_portrait(
                ts, embedding_dim=3, time_delay=7,
                title=f'Phase Portrait: {ts_name}',
                save_path=f'{output_dir}/phase_{ts_name}'
            )

            # Poincaré截面
            plot_poincare_section(
                ts, embedding_dim=3, time_delay=7,
                title=f'Poincaré Section: {ts_name}',
                save_path=f'{output_dir}/poincare_{ts_name}'
            )

            print(f"  {component}: λ_R={lyap_r:.4f}, λ_W={lyap_w:.4f}")

    print("\n" + "=" * 60)
    print("Part 6: Summary and Parameter Analysis")
    print("=" * 60)

    # 生成综合分析图
    plot_summary_analysis(results, processed_data, output_dir)

    # 生成Figure 3: Integrated chaos characterization
    plot_figure3_chaos_characterization(results, processed_data, output_dir)

    # 生成Figure 4: Latent space learning and SINDy identification
    plot_figure4_latent_sindy(results, processed_data, output_dir,
                              latent_ae, latent_vae, all_labels,
                              history, vae_history,
                              sindy_models, province_latents)

    # 生成Figure 5: Phase space geometry and attractor structure
    plot_figure5_phase_space(results, processed_data, output_dir)

    # ========== 生成表格 ==========
    print("\n" + "=" * 60)
    print("Part 7: Generating Tables for the Paper")
    print("=" * 60)

    # 计算每个省份的特征数量
    features_per_province = len(all_features) // len(province_list)

    # 生成所有5个表格
    tables = generate_all_tables(
        data_dict=data_dict,
        results=results,
        output_dir=output_dir,
        autoencoder=autoencoder,
        vae=vae,
        X_tensor=X_tensor,
        all_labels=all_labels,
        features_per_province=features_per_province,
        sindy_models=sindy_models,
        province_latents=province_latents,
        device=device
    )

    # 保存结果到文件
    save_results(results, output_dir)

    return results


def plot_summary_analysis(results, processed_data, output_dir):
    """
    绘制综合分析结果
    """
    provinces = list(processed_data.keys())
    components = ['PC', 'Mobile']

    # 图1: λ值比较
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # HVG λ比较
    ax1 = axes[0, 0]
    lambda_values = []
    labels = []
    for p in provinces:
        for c in components:
            key = f'{p}_{c}'
            lambda_values.append(results['hvg_lambda'][key]['lambda'])
            labels.append(f'{p}\n{c}')

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax1.bar(range(len(labels)), lambda_values, color=colors)
    ax1.axhline(y=np.log(3 / 2), color='red', linestyle='--',
                label=f'Random threshold (ln(3/2)≈{np.log(3 / 2):.3f})')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('λ (HVG degree exponent)', fontsize=16)
    ax1.set_title('HVG Degree Distribution Exponent', fontsize=18)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Lyapunov指数比较
    ax2 = axes[0, 1]
    lyap_r_values = []
    lyap_w_values = []
    for p in provinces:
        for c in components:
            key = f'{p}_{c}'
            lyap_r_values.append(results['lyapunov'][key]['rosenstein'])
            lyap_w_values.append(results['lyapunov'][key]['wolf'])

    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width / 2, lyap_r_values, width, label='Rosenstein', color='steelblue')
    ax2.bar(x + width / 2, lyap_w_values, width, label='Wolf', color='darkorange')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Lyapunov Exponent', fontsize=16)
    ax2.set_title('Lyapunov Exponent Comparison', fontsize=18)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 时间序列趋势
    ax3 = axes[1, 0]
    for p in provinces:
        data = processed_data[p]
        ts_total = data['smooth']['PC'].values + data['smooth']['Mobile'].values
        # 月度平均
        monthly = pd.Series(ts_total, index=data['smooth'].index).resample('M').mean()
        ax3.plot(monthly.index, monthly.values, '-', linewidth=1.5, label=p)

    ax3.set_xlabel('Date', fontsize=17)
    ax3.set_ylabel('Baidu Index (PC + Mobile)', fontsize=18)
    ax3.set_title('Monthly Trend Comparison', fontsize=18)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 混沌判据总结
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_lines = ["Chaos Analysis Summary", "=" * 40, ""]
    lambda_random = np.log(3 / 2)

    for p in provinces:
        summary_lines.append(f"{p}:")
        for c in components:
            key = f'{p}_{c}'
            lam = results['hvg_lambda'][key]['lambda']
            lyap = results['lyapunov'][key]['rosenstein']

            hvg_chaos = "Chaotic" if lam < lambda_random else "Non-chaotic"
            lyap_chaos = "Chaotic" if lyap > 0 else "Non-chaotic"

            summary_lines.append(f"  {c}: HVG→{hvg_chaos}, Lyap→{lyap_chaos}")
        summary_lines.append("")

    ax4.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax4.transAxes,
             fontsize=14, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/summary_analysis.pdf', bbox_inches='tight')
    plt.close()

    # 图2: 相关性热力图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 构建相关性矩阵
    all_series = []
    series_names = []
    for p in provinces:
        data = processed_data[p]
        for c in components:
            all_series.append(data['smooth'][c].values)
            series_names.append(f'{p[:2]}_{c}')

    # 计算相关系数
    n_series = len(all_series)
    corr_matrix = np.zeros((n_series, n_series))
    for i in range(n_series):
        for j in range(n_series):
            corr_matrix[i, j] = np.corrcoef(all_series[i], all_series[j])[0, 1]

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_series))
    ax.set_yticks(range(n_series))
    ax.set_xticklabels(series_names, rotation=45, ha='right')
    ax.set_yticklabels(series_names)

    # 添加数值标签
    for i in range(n_series):
        for j in range(n_series):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black', fontsize=8)

    plt.colorbar(im, label='Correlation')
    ax.set_title('Cross-Province Correlation Matrix', fontsize=18)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/correlation_matrix.pdf', bbox_inches='tight')
    plt.close()


def plot_figure4_latent_sindy(results, processed_data, output_dir,
                              latent_ae, latent_vae, all_labels,
                              ae_history, vae_history,
                              sindy_models, province_latents):
    """
    Figure 4. Latent space learning and SINDy identification.
    (a) 3D latent space visualization colored by province
    (b) Temporal latent trajectories for each province
    (c) AE/VAE training loss curves
    (d) SINDy prediction vs. actual latent values for Beijing (200-day test period)
    (e) Coefficient magnitude heatmap across provinces and term categories
    """
    from mpl_toolkits.mplot3d import Axes3D

    provinces = list(processed_data.keys())
    province_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    province_markers = ['o', 's', '^', 'D']

    # 创建Figure 4: 2行3列布局
    fig = plt.figure(figsize=(18, 12))

    # ========== (a) 3D latent space visualization colored by province ==========
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    for idx, province in enumerate(provinces):
        mask = all_labels == idx
        ax1.scatter(latent_ae[mask, 0], latent_ae[mask, 1], latent_ae[mask, 2],
                    c=province_colors[idx], marker=province_markers[idx],
                    s=15, alpha=0.6, label=province)

    ax1.set_xlabel('Latent Dim 1', fontsize=11)
    ax1.set_ylabel('Latent Dim 2', fontsize=11)
    ax1.set_zlabel('Latent Dim 3', fontsize=11)
    ax1.set_title('(a) 3D Latent Space by Province', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=14)
    ax1.view_init(elev=20, azim=45)

    # ========== (b) Temporal latent trajectories for each province ==========
    ax2 = fig.add_subplot(2, 3, 2)

    features_per_province = len(latent_ae) // len(provinces)
    for idx, province in enumerate(provinces):
        start_idx = idx * features_per_province
        end_idx = (idx + 1) * features_per_province
        latent_province = latent_ae[start_idx:end_idx]

        # 绘制第一个潜变量维度的时间轨迹
        t = np.arange(len(latent_province))
        ax2.plot(t, latent_province[:, 0], '-', linewidth=0.5,
                 color=province_colors[idx], label=f'{province} (z₁)', alpha=0.6)

    ax2.set_xlabel('Time Step', fontsize=16)
    ax2.set_ylabel('Latent Value (z1)', fontsize=16)
    ax2.set_title('(b) Temporal Latent Trajectories', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ========== (c) AE/VAE training loss curves ==========
    ax3 = fig.add_subplot(2, 3, 3)

    epochs_ae = range(1, len(ae_history['train_loss']) + 1)
    epochs_vae = range(1, len(vae_history['train_loss']) + 1)

    ax3.plot(epochs_ae, ae_history['train_loss'], 'b-', linewidth=1.5,
             label='AE Train', alpha=0.8)
    if ae_history['val_loss']:
        ax3.plot(epochs_ae, ae_history['val_loss'], 'b--', linewidth=1.5,
                 label='AE Val', alpha=0.8)

    ax3.plot(epochs_vae, vae_history['train_loss'], 'r-', linewidth=1.5,
             label='VAE Train', alpha=0.8)
    if vae_history['val_loss']:
        ax3.plot(epochs_vae, vae_history['val_loss'], 'r--', linewidth=1.5,
                 label='VAE Val', alpha=0.8)

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('(c) AE/VAE Training Loss Curves', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # ========== (d) SINDy prediction vs. actual latent values for Beijing ==========
    ax4 = fig.add_subplot(2, 3, 4)

    # 获取Beijing的潜空间数据和SINDy模型
    beijing_idx = provinces.index('Beijing') if 'Beijing' in provinces else 0
    beijing_province = provinces[beijing_idx]

    if beijing_province in sindy_models and beijing_province in province_latents:
        sindy_beijing = sindy_models[beijing_province]
        latent_beijing = province_latents[beijing_province]

        # 使用后200个点作为测试期
        test_len = min(200, len(latent_beijing) - 1)
        test_start = len(latent_beijing) - test_len - 1

        X_test = latent_beijing[test_start:-1]
        X_actual = latent_beijing[test_start + 1:]
        X_pred = sindy_beijing.predict(X_test)

        t_test = np.arange(test_len)

        # 绘制三个潜变量维度
        colors_dim = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(min(3, X_actual.shape[1])):
            ax4.plot(t_test, X_actual[:, i], '-', linewidth=0.5,
                     color=colors_dim[i], label=f'Actual z{i + 1}', alpha=0.6)
            ax4.plot(t_test, X_pred[:, i], '--', linewidth=1.5,
                     color=colors_dim[i], label=f'SINDy z{i + 1}', alpha=0.6)

        ax4.set_xlabel('Test Time Step', fontsize=14)
        ax4.set_ylabel('Latent Value', fontsize=16)
        ax4.set_title(f'(d) SINDy Prediction vs Actual ({beijing_province}, {test_len}-day)',
                      fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=12, ncol=2)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Beijing SINDy data not available',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=16)
        ax4.set_title('(d) SINDy Prediction vs Actual', fontsize=16, fontweight='bold')

    # ========== (e) Coefficient magnitude heatmap across provinces and term categories ==========
    ax5 = fig.add_subplot(2, 3, 5)

    # 收集所有省份的系数
    if sindy_models:
        # 获取特征名称（从第一个模型）
        first_model = list(sindy_models.values())[0]
        feature_names = first_model.library.feature_names

        # 将特征分类
        term_categories = {
            'Constant': [],
            'Linear': [],
            'Quadratic': [],
            'Interaction': [],
            'Trigonometric': []
        }

        for i, name in enumerate(feature_names):
            if name == '1':
                term_categories['Constant'].append(i)
            elif '^2' in name:
                term_categories['Quadratic'].append(i)
            elif '*' in name:
                term_categories['Interaction'].append(i)
            elif 'sin' in name or 'cos' in name:
                term_categories['Trigonometric'].append(i)
            else:
                term_categories['Linear'].append(i)

        # 计算每个省份每个类别的平均系数大小
        category_names = list(term_categories.keys())
        coef_matrix = np.zeros((len(provinces), len(category_names)))

        for p_idx, province in enumerate(provinces):
            if province in sindy_models:
                model = sindy_models[province]
                coefs = np.abs(model.coefficients)

                for c_idx, cat in enumerate(category_names):
                    indices = term_categories[cat]
                    if len(indices) > 0:
                        coef_matrix[p_idx, c_idx] = np.mean(coefs[:, indices])

        # 绘制热力图
        im = ax5.imshow(coef_matrix, aspect='auto', cmap='YlOrRd')
        ax5.set_xticks(range(len(category_names)))
        ax5.set_xticklabels(category_names, rotation=45, ha='right', fontsize=10)
        ax5.set_yticks(range(len(provinces)))
        ax5.set_yticklabels(provinces, fontsize=11)
        ax5.set_xlabel('Term Category', fontsize=12)
        ax5.set_ylabel('Province', fontsize=12)
        ax5.set_title('(e) SINDy Coefficient Magnitude Heatmap', fontsize=14, fontweight='bold')

        # 添加数值标注
        for i in range(len(provinces)):
            for j in range(len(category_names)):
                text = ax5.text(j, i, f'{coef_matrix[i, j]:.3f}',
                                ha='center', va='center', fontsize=9,
                                color='white' if coef_matrix[i, j] > coef_matrix.max() / 2 else 'black')

        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Mean |Coefficient|', fontsize=10)
    else:
        ax5.text(0.5, 0.5, 'SINDy models not available',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('(e) SINDy Coefficient Magnitude Heatmap', fontsize=14, fontweight='bold')

    # # 隐藏第6个子图位置（2行3列的右下角）
    # ax6 = fig.add_subplot(2, 3, 6)
    # ax6.axis('off')

    # 添加图表说明文字
    # summary_text = """
    # Figure 4 Summary:
    # ─────────────────────────────
    # (a) 3D latent space shows province-specific
    #     clustering patterns learned by AE
    #
    # (b) Temporal trajectories reveal dynamic
    #     evolution in latent dimensions
    #
    # (c) Training curves show convergence of
    #     both AE and VAE models
    #
    # (d) SINDy captures latent dynamics with
    #     sparse polynomial representation
    #
    # (e) Coefficient heatmap indicates relative
    #     importance of different term types
    # """
    # ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
    #          verticalalignment='center', fontfamily='monospace',
    #          bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure4_latent_sindy.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure4_latent_sindy.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure 4 saved to {output_dir}/figure4_latent_sindy.png/pdf")


def estimate_correlation_dimension(time_series, embedding_dim=3, time_delay=1, r_values=None):
    """
    估计关联维数 (Correlation Dimension)

    使用Grassberger-Procaccia算法

    Parameters:
    -----------
    time_series : array-like
        时间序列数据
    embedding_dim : int
        嵌入维度
    time_delay : int
        时间延迟
    r_values : array-like
        用于计算C(r)的距离值范围

    Returns:
    --------
    d2 : float
        关联维数
    r_vals : array
        距离值
    c_vals : array
        关联积分值
    """
    # 相空间嵌入
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    # 计算所有点对之间的距离
    distances = pdist(embedded)

    if len(distances) == 0:
        return 0, np.array([]), np.array([])

    # 设置r值范围
    if r_values is None:
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
        if r_min <= 0:
            r_min = 1e-10
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), 30)

    # 计算关联积分 C(r)
    n_pairs = len(distances)
    c_values = []

    for r in r_values:
        count = np.sum(distances < r)
        c_r = count / n_pairs
        if c_r > 0:
            c_values.append(c_r)
        else:
            c_values.append(1e-10)

    c_values = np.array(c_values)

    # 在log-log域拟合斜率得到关联维数
    log_r = np.log(r_values)
    log_c = np.log(c_values)

    # 选择线性区域进行拟合
    valid_mask = np.isfinite(log_c) & (c_values > 1e-8) & (c_values < 0.9)
    if np.sum(valid_mask) < 5:
        return 0, r_values, c_values

    slope, _, r_value, _, _ = stats.linregress(log_r[valid_mask], log_c[valid_mask])

    return slope, r_values, c_values


def calculate_convex_hull_volume(time_series, embedding_dim=3, time_delay=1):
    """
    计算相空间吸引子的凸包体积

    Parameters:
    -----------
    time_series : array-like
        时间序列数据
    embedding_dim : int
        嵌入维度
    time_delay : int
        时间延迟

    Returns:
    --------
    volume : float
        凸包体积
    """
    from scipy.spatial import ConvexHull

    # 相空间嵌入
    n = len(time_series)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay:i * time_delay + n_embedded]

    try:
        # 标准化以便比较
        scaler = StandardScaler()
        embedded_scaled = scaler.fit_transform(embedded)

        hull = ConvexHull(embedded_scaled)
        return hull.volume
    except Exception as e:
        # 如果无法计算凸包，返回协方差矩阵行列式作为体积度量
        cov = np.cov(embedded.T)
        return np.sqrt(np.abs(np.linalg.det(cov)))


def plot_figure5_phase_space(results, processed_data, output_dir):
    """
    Figure 5. Phase space geometry and attractor structure.
    (a) 3D phase portraits for eight time series (rows: provinces; columns: PC/Mobile)
    (b) Poincaré sections at z₁ = mean(z₁) (Beijing Mobile)
    (c) Return maps (z_n vs. z_{n+1}) with local regression fits (Beijing Mobile)
    (d) Attractor metric comparison: correlation dimension vs. Lyapunov exponent
        with convex hull volume indicated by marker size
    """
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.ndimage import gaussian_filter1d

    provinces = list(processed_data.keys())
    components = ['PC', 'Mobile']

    # 颜色和标记设置
    province_colors = {'Beijing': '#1f77b4', 'Guangdong': '#ff7f0e',
                       'Henan': '#2ca02c', 'Shanghai': '#d62728'}
    province_markers = {'Beijing': 'o', 'Guangdong': 's', 'Henan': '^', 'Shanghai': 'D'}
    channel_colors = {'PC': '#1f77b4', 'Mobile': '#ff7f0e'}

    # 嵌入参数
    embedding_dim = 3
    time_delay = 7

    # 创建Figure 5: 复杂布局
    fig = plt.figure(figsize=(20, 16))

    # ========== (a) 3D phase portraits for 8 time series (4x2 grid) ==========
    # 使用GridSpec实现更灵活的布局
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.2)

    # 上半部分: 3D相图 (2行4列)
    for p_idx, province in enumerate(provinces):
        for c_idx, component in enumerate(components):
            row = c_idx  # 0 for PC, 1 for Mobile
            col = p_idx

            ax = fig.add_subplot(gs[row, col], projection='3d')

            ts = processed_data[province]['smooth'][component].values
            n = len(ts)
            n_embedded = n - (embedding_dim - 1) * time_delay

            embedded = np.zeros((n_embedded, embedding_dim))
            for i in range(embedding_dim):
                embedded[:, i] = ts[i * time_delay:i * time_delay + n_embedded]

            # 绘制3D轨迹
            color = province_colors.get(province, '#333333')
            ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                    '-', linewidth=0.3, alpha=0.5, color=color)
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                                 c=np.arange(n_embedded), cmap='viridis', s=2, alpha=0.6)

            ax.set_xlabel('z1', fontsize=14)
            ax.set_ylabel('z2', fontsize=14)
            ax.set_zlabel('z3', fontsize=14)
            ax.set_title(f'{province}\n{component}', fontsize=16, fontweight='bold')
            ax.view_init(elev=20, azim=45)
            ax.tick_params(labelsize=7)

    # 添加行标签
    fig.text(0.03, 0.85, 'PC', fontsize=18, fontweight='bold', rotation=90, va='center')
    fig.text(0.03, 0.65, 'Mobile', fontsize=18, fontweight='bold', rotation=90, va='center')

    # 添加子图标题
    fig.text(0.5, 0.93, '(a) 3D Phase Portraits', fontsize=18, fontweight='bold',
             ha='center', va='top')

    # ========== (b) Poincaré sections for Beijing Mobile ==========
    ax_poincare = fig.add_subplot(gs[2, 0:2])

    # 获取Beijing Mobile数据
    ts_beijing = processed_data['Beijing']['smooth']['Mobile'].values
    n = len(ts_beijing)
    n_embedded = n - (embedding_dim - 1) * time_delay

    embedded_beijing = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        embedded_beijing[:, i] = ts_beijing[i * time_delay:i * time_delay + n_embedded]

    # 计算截面值 (z₁的均值)
    section_value = np.mean(embedded_beijing[:, 0])

    # 找穿越截面的点 (正向穿越)
    crossings = []
    crossing_indices = []
    for i in range(n_embedded - 1):
        if embedded_beijing[i, 0] < section_value <= embedded_beijing[i + 1, 0]:
            # 线性插值
            t_cross = (section_value - embedded_beijing[i, 0]) / \
                      (embedded_beijing[i + 1, 0] - embedded_beijing[i, 0])
            cross_point = embedded_beijing[i] + t_cross * (embedded_beijing[i + 1] - embedded_beijing[i])
            crossings.append(cross_point)
            crossing_indices.append(i)

    crossings = np.array(crossings)

    if len(crossings) > 5:
        # 绘制Poincaré截面 (z₂ vs z₃)
        scatter_poincare = ax_poincare.scatter(crossings[:, 1], crossings[:, 2],
                                               c=np.arange(len(crossings)), cmap='plasma',
                                               s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax_poincare.set_xlabel('z2', fontsize=14)
        ax_poincare.set_ylabel('z3', fontsize=14)
        ax_poincare.set_title(f'(b) Poincaré Section at z1 = {section_value:.2f} (Beijing Mobile)',
                              fontsize=12, fontweight='bold')
        ax_poincare.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter_poincare, ax=ax_poincare)
        cbar.set_label('Crossing Index', fontsize=18)
    else:
        ax_poincare.text(0.5, 0.5, 'Insufficient crossings', ha='center', va='center',
                         transform=ax_poincare.transAxes, fontsize=12)
        ax_poincare.set_title('(b) Poincaré Section (Beijing Mobile)', fontsize=18, fontweight='bold')

    # ========== (c) Return maps with local regression fits ==========
    ax_return = fig.add_subplot(gs[2, 2:4])

    # 获取非截面维度 - 与原代码plot_poincare_section完全一致
    section_dim = 0
    other_dims = [i for i in range(embedding_dim) if i != section_dim]

    # 回归映射 (连续穿越点之间的关系) - 完全照搬原代码
    if len(crossings) > 1 and len(other_dims) >= 1:
        ax_return.scatter(crossings[:-1, other_dims[0]], crossings[1:, other_dims[0]],
                          s=20, alpha=0.6, c='red')
        ax_return.plot([crossings[:, other_dims[0]].min(), crossings[:, other_dims[0]].max()],
                       [crossings[:, other_dims[0]].min(), crossings[:, other_dims[0]].max()],
                       'k--', alpha=0.5)
        ax_return.set_xlabel(f'x_n (Dim {other_dims[0] + 1})', fontsize=12)
        ax_return.set_ylabel(f'x_{n + 1} (Dim {other_dims[0] + 1})', fontsize=12)
        ax_return.set_title('(c) Return Map (Beijing Mobile)', fontsize=14, fontweight='bold')
        ax_return.grid(True, alpha=0.3)
    else:
        ax_return.text(0.5, 0.5, 'Insufficient data for return map',
                       ha='center', va='center', transform=ax_return.transAxes, fontsize=12)
        ax_return.set_title('(c) Return Map (Beijing Mobile)', fontsize=12, fontweight='bold')

    # ========== (d) Attractor metric comparison ==========
    ax_metrics = fig.add_subplot(gs[3, :])

    # 计算所有时间序列的指标
    corr_dims = []
    lyap_exps = []
    hull_volumes = []
    labels = []
    marker_list = []
    color_list = []

    for province in provinces:
        for component in components:
            ts_name = f'{province}_{component}'
            ts = processed_data[province]['smooth'][component].values

            # 关联维数
            d2, _, _ = estimate_correlation_dimension(ts, embedding_dim, time_delay)
            corr_dims.append(d2)

            # Lyapunov指数 (从已有结果获取)
            lyap_exp = results['lyapunov'][ts_name]['rosenstein']
            lyap_exps.append(lyap_exp)

            # 凸包体积
            volume = calculate_convex_hull_volume(ts, embedding_dim, time_delay)
            hull_volumes.append(volume)

            labels.append(f'{province}\n{component}')
            marker_list.append(province_markers.get(province, 'o'))
            color_list.append(channel_colors.get(component, '#333333'))

    corr_dims = np.array(corr_dims)
    lyap_exps = np.array(lyap_exps)
    hull_volumes = np.array(hull_volumes)

    # 归一化体积用于marker大小
    if hull_volumes.max() > hull_volumes.min():
        sizes = 100 + 400 * (hull_volumes - hull_volumes.min()) / (hull_volumes.max() - hull_volumes.min())
    else:
        sizes = np.ones_like(hull_volumes) * 200

    # 绘制散点图
    for i, (cd, le, s, marker, color, label) in enumerate(zip(corr_dims, lyap_exps, sizes,
                                                              marker_list, color_list, labels)):
        ax_metrics.scatter(cd, le, s=s, marker=marker, c=color, alpha=0.7,
                           edgecolors='black', linewidths=1.5, label=label)

    # 添加参考线
    ax_metrics.axhline(y=0, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                       label='λ_Lyap = 0 (chaos threshold)')

    # 添加文字标注
    for i, (cd, le, label) in enumerate(zip(corr_dims, lyap_exps, labels)):
        ax_metrics.annotate(label.replace('\n', ' '), (cd, le),
                            xytext=(5, 6), textcoords='offset points',
                            fontsize=12, alpha=0.8)

    ax_metrics.set_xlabel('Correlation Dimension (D2)', fontsize=14)
    ax_metrics.set_ylabel('Lyapunov Exponent (λ_max)', fontsize=14)
    ax_metrics.set_title(
        '(d) Attractor Metrics: Correlation Dimension vs. Lyapunov Exponent\n(Marker size indicates Convex Hull Volume)',
        fontsize=12, fontweight='bold')
    ax_metrics.grid(True, alpha=0.3)

    # 创建图例
    legend_elements = []
    # 省份标记
    for province in provinces:
        legend_elements.append(plt.Line2D([0], [0], marker=province_markers[province],
                                          color='gray', markerfacecolor='gray',
                                          markersize=10, linestyle='None', label=province))
    # 通道颜色
    for component in components:
        legend_elements.append(plt.Line2D([0], [0], marker='o',
                                          color=channel_colors[component],
                                          markerfacecolor=channel_colors[component],
                                          markersize=10, linestyle='None', label=component))
    # 参考线
    legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--',
                                      label='Chaos threshold'))

    ax_metrics.legend(handles=legend_elements, loc='lower right', fontsize=14, ncol=2)

    # # 添加体积大小说明
    # ax_metrics.text(0.02, 0.98, 'Marker size indicates\nConvex Hull Volume',
    #                 transform=ax_metrics.transAxes, fontsize=10, va='top',
    #                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure5_phase_space.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure5_phase_space.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure 5 saved to {output_dir}/figure5_phase_space.png/pdf")


def plot_figure3_chaos_characterization(results, processed_data, output_dir):
    """
    Figure 3. Integrated chaos characterization.
    (a) HVG degree distributions P(k) with exponential fits for four provinces (Mobile channel).
    (b) Lyapunov exponent convergence curves showing average logarithmic divergence vs. iteration.
    (c) Scatter plot of HVG λ vs. Lyapunov λ_max with provincial labels (shapes: province; colors: channel).
    (d) Paired comparison of chaos metrics between PC and Mobile channels.
    """
    provinces = list(processed_data.keys())
    components = ['PC', 'Mobile']

    # 颜色和标记设置
    province_markers = {'Beijing': 'o', 'Guangdong': 's', 'Henan': '^', 'Shanghai': 'D'}
    channel_colors = {'PC': '#1f77b4', 'Mobile': '#ff7f0e'}
    province_colors = plt.cm.Set2(np.linspace(0, 1, len(provinces)))

    fig = plt.figure(figsize=(16, 14))

    # ========== (a) HVG degree distributions P(k) for Mobile channel ==========
    ax1 = fig.add_subplot(2, 2, 1)

    lambda_random = np.log(3 / 2)

    for idx, province in enumerate(provinces):
        ts_name = f'{province}_Mobile'
        degree_dist = results['hvg_degree_dist'][ts_name]
        lambda_exp = results['hvg_lambda'][ts_name]['lambda']

        k_values = np.array(sorted(degree_dist.keys()))
        p_values = np.array([degree_dist[k] for k in k_values])

        # 过滤零值
        mask = p_values > 0
        k_plot = k_values[mask]
        p_plot = p_values[mask]

        # 绘制数据点
        ax1.semilogy(k_plot, p_plot, 'o', markersize=8, alpha=0.7,
                     color=province_colors[idx], label=f'{province}')

        # 拟合曲线
        k_fit = np.linspace(k_values.min(), k_values.max(), 100)
        # 使用对数域拟合
        x = k_plot.astype(float)
        y = np.log(p_plot.astype(float))
        b, a = np.polyfit(x, y, 1)
        p_fit = np.exp(a) * np.exp(b * k_fit)
        ax1.semilogy(k_fit, p_fit, '-', linewidth=1.5, color=province_colors[idx], alpha=0.8)

    # 添加随机参考线
    ax1.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5)
    ax1.text(ax1.get_xlim()[1] * 0.7, 0.12, f'λ_random = ln(3/2) ≈ {lambda_random:.3f}',
             fontsize=10, color='gray')

    ax1.set_xlabel('Degree k', fontsize=18)
    ax1.set_ylabel('P(k)', fontsize=18)
    ax1.set_title('(a) HVG Degree Distributions (Mobile)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(bottom=1e-4)

    # ========== (b) Lyapunov exponent convergence curves ==========
    ax2 = fig.add_subplot(2, 2, 2)

    for idx, province in enumerate(provinces):
        ts_name = f'{province}_Mobile'
        divergence = results['lyapunov_divergence'][ts_name]

        if len(divergence) > 0:
            # 绘制散度曲线
            t = np.arange(len(divergence))
            ax2.plot(t, divergence, '-', linewidth=1.5, color=province_colors[idx],
                     label=f'{province}', alpha=0.8)

    ax2.set_xlabel('Iteration (time steps)', fontsize=18)
    ax2.set_ylabel('⟨ln(divergence)⟩', fontsize=18)
    ax2.set_title('(b) Lyapunov Exponent Convergence (Mobile)', fontsize=16, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=16)
    ax2.grid(True, alpha=0.3)

    # ========== (c) Scatter plot of HVG λ vs. Lyapunov λ_max ==========
    ax3 = fig.add_subplot(2, 2, 3)

    for province in provinces:
        for component in components:
            ts_name = f'{province}_{component}'
            hvg_lambda = results['hvg_lambda'][ts_name]['lambda']
            lyap_max = results['lyapunov'][ts_name]['rosenstein']

            ax3.scatter(hvg_lambda, lyap_max,
                        marker=province_markers[province],
                        c=channel_colors[component],
                        s=120, alpha=0.8, edgecolors='black', linewidths=0.5)

    # 添加参考线
    ax3.axvline(x=lambda_random, color='red', linestyle='--', alpha=0.7,
                label=f'λ_HVG = ln(3/2)')
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7,
                label='λ_Lyap = 0')

    # 添加图例
    legend_elements = []
    for province in provinces:
        legend_elements.append(plt.Line2D([0], [0], marker=province_markers[province],
                                          color='gray', markerfacecolor='gray',
                                          markersize=10, linestyle='None', label=province))
    for component in components:
        legend_elements.append(plt.Line2D([0], [0], marker='o',
                                          color=channel_colors[component],
                                          markerfacecolor=channel_colors[component],
                                          markersize=10, linestyle='None', label=component))
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='λ_HVG threshold'))
    legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', label='λ_Lyap = 0'))

    ax3.legend(handles=legend_elements, loc='lower right', fontsize=16, ncol=2)
    ax3.set_xlabel('HVG λ (degree exponent)', fontsize=18)
    ax3.set_ylabel('Lyapunov λ_max (Rosenstein)', fontsize=18)
    ax3.set_title('(c) HVG λ vs. Lyapunov λ_max', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 添加象限标注
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    ax3.text(xlim[0] + 0.02, ylim[1] - 0.01, 'Chaotic\n(both metrics)', fontsize=14,
             ha='left', va='top', color='darkred', alpha=0.7)
    ax3.text(xlim[1] - 0.02, ylim[0] + 0.01, 'Non-chaotic\n(both metrics)', fontsize=14,
             ha='right', va='bottom', color='darkblue', alpha=0.7)

    # ========== (d) Paired comparison between PC and Mobile ==========
    ax4 = fig.add_subplot(2, 2, 4)

    x_pos = np.arange(len(provinces))
    width = 0.35

    hvg_pc = [results['hvg_lambda'][f'{p}_PC']['lambda'] for p in provinces]
    hvg_mobile = [results['hvg_lambda'][f'{p}_Mobile']['lambda'] for p in provinces]
    lyap_pc = [results['lyapunov'][f'{p}_PC']['rosenstein'] for p in provinces]
    lyap_mobile = [results['lyapunov'][f'{p}_Mobile']['rosenstein'] for p in provinces]

    # 创建双Y轴
    ax4_twin = ax4.twinx()

    # HVG λ bars
    bars1 = ax4.bar(x_pos - width / 2, hvg_pc, width * 0.8, label='HVG λ (PC)',
                    color='#1f77b4', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x_pos + width / 2, hvg_mobile, width * 0.8, label='HVG λ (Mobile)',
                    color='#ff7f0e', alpha=0.7, edgecolor='black')

    # 添加HVG阈值线
    ax4.axhline(y=lambda_random, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.text(len(provinces) - 0.5, lambda_random + 0.01, 'ln(3/2)', fontsize=12, color='red')

    # Lyapunov点
    ax4_twin.plot(x_pos - width / 2, lyap_pc, 'D', markersize=10,
                  color='#1f77b4', markeredgecolor='black', label='Lyap λ (PC)')
    ax4_twin.plot(x_pos + width / 2, lyap_mobile, 's', markersize=10,
                  color='#ff7f0e', markeredgecolor='black', label='Lyap λ (Mobile)')

    # 连接PC和Mobile的Lyapunov点
    for i in range(len(provinces)):
        ax4_twin.plot([x_pos[i] - width / 2, x_pos[i] + width / 2],
                      [lyap_pc[i], lyap_mobile[i]],
                      'k-', alpha=0.3, linewidth=1)

    ax4_twin.axhline(y=0, color='green', linestyle='--', alpha=0.7, linewidth=1.5)

    ax4.set_xlabel('Province', fontsize=18)
    ax4.set_ylabel('HVG λ (bars)', fontsize=18, color='black')
    ax4_twin.set_ylabel('Lyapunov λ_max (markers)', fontsize=18, color='gray')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(provinces, fontsize=11)
    ax4.set_title('(d) PC vs. Mobile Channel Comparison', fontsize=18, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)

    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure3_chaos_characterization.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure3_chaos_characterization.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure 3 saved to {output_dir}/figure3_chaos_characterization.png/pdf")


# ============================================================================
# Table Generation Function - Generate Tables for the Paper
# ============================================================================

def generate_table1_descriptive_stats(data_dict, output_dir):
    """
    Table 1: Descriptive statistics of daily Taijiquan search index (2014-2024)
    Columns: Province, Channel, N, Mean, SD, Min, Median, Max, Skewness, Kurtosis, CV(%)
    """
    rows = []
    provinces_order = ['Beijing', 'Guangdong', 'Henan', 'Shanghai']

    for province in provinces_order:
        if province not in data_dict:
            continue
        df = data_dict[province]
        for channel in ['PC', 'Mobile']:
            ts = df[channel].values
            n = len(ts)
            mean_val = np.mean(ts)
            std_val = np.std(ts, ddof=1)
            min_val = np.min(ts)
            median_val = np.median(ts)
            max_val = np.max(ts)
            skewness = stats.skew(ts)
            kurtosis = stats.kurtosis(ts)  # excess kurtosis
            cv = (std_val / mean_val) * 100 if mean_val != 0 else 0

            rows.append({
                'Province': province,
                'Channel': channel,
                'N': n,
                'Mean': round(mean_val, 1),
                'SD': round(std_val, 1),
                'Min': int(min_val),
                'Median': int(median_val),
                'Max': int(max_val),
                'Skewness': round(skewness, 2),
                'Kurtosis': round(kurtosis, 2),
                'CV(%)': round(cv, 1)
            })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(f'{output_dir}/Table1_Descriptive_Statistics.csv', index=False)

    print("\n" + "=" * 70)
    print("Table 1: Descriptive Statistics of Daily Taijiquan Search Index")
    print("=" * 70)
    print(df_table.to_string(index=False))

    return df_table


def generate_table2_annual_trend(data_dict, output_dir):
    """
    Table 2: Annual mean total search index and temporal change
    Columns: Year, Beijing, Guangdong, Henan, Shanghai, Δ(%)
    """
    provinces_order = ['Beijing', 'Guangdong', 'Henan', 'Shanghai']
    years = range(2014, 2025)

    # 计算每年每省份的总指数（PC + Mobile）平均值
    annual_data = {province: {} for province in provinces_order}

    for province in provinces_order:
        if province not in data_dict:
            continue
        df = data_dict[province]
        df_with_date = df.copy()
        df_with_date['Total'] = df_with_date['PC'] + df_with_date['Mobile']
        df_with_date['Year'] = df_with_date.index.year

        for year in years:
            year_data = df_with_date[df_with_date['Year'] == year]['Total']
            if len(year_data) > 0:
                annual_data[province][year] = round(year_data.mean())

    # 构建表格
    rows = []
    for year in years:
        row = {'Year': year}
        for province in provinces_order:
            if year in annual_data[province]:
                row[province] = annual_data[province][year]
            else:
                row[province] = None
        rows.append(row)

    # 计算变化率（从峰值年到2024年）
    delta_row = {'Year': 'Δ(%)'}
    for province in provinces_order:
        values = [annual_data[province].get(y, 0) for y in years if annual_data[province].get(y, 0) > 0]
        if len(values) > 0 and annual_data[province].get(2024, 0) > 0:
            peak_val = max(values)
            val_2024 = annual_data[province].get(2024, 0)
            delta = ((val_2024 - peak_val) / peak_val) * 100
            delta_row[province] = f"{delta:.1f}"
        else:
            delta_row[province] = None
    rows.append(delta_row)

    df_table = pd.DataFrame(rows)
    df_table.to_csv(f'{output_dir}/Table2_Annual_Trend.csv', index=False)

    print("\n" + "=" * 70)
    print("Table 2: Annual Mean Total Search Index and Temporal Change")
    print("=" * 70)
    print(df_table.to_string(index=False))

    return df_table


def generate_table3_chaos_analysis(results, output_dir):
    """
    Table 3: Chaos characterization - HVG and Lyapunov exponent analysis
    Columns: Province, Channel, HVG λ, R², λ_max(Rosenstein), λ_max(Wolf), Chaos Strength
    """
    rows = []
    provinces_order = ['Beijing', 'Guangdong', 'Henan', 'Shanghai']
    lambda_random = np.log(3 / 2)  # ≈ 0.4055

    for province in provinces_order:
        for channel in ['PC', 'Mobile']:
            ts_name = f'{province}_{channel}'

            if ts_name not in results['hvg_lambda'] or ts_name not in results['lyapunov']:
                continue

            hvg_lambda = results['hvg_lambda'][ts_name]['lambda']
            r_squared = results['hvg_lambda'][ts_name]['r_squared']
            lyap_rosenstein = results['lyapunov'][ts_name]['rosenstein']
            lyap_wolf = results['lyapunov'][ts_name]['wolf']

            # 判断混沌强度
            # 基于HVG λ和Lyapunov指数综合判断
            hvg_chaotic = hvg_lambda < lambda_random
            lyap_chaotic = lyap_rosenstein > 0

            if hvg_chaotic and lyap_chaotic:
                if lyap_rosenstein > 0.025 or hvg_lambda < 0.28:
                    chaos_strength = "Strong"
                elif lyap_rosenstein > 0.02 or hvg_lambda < 0.31:
                    chaos_strength = "Moderate"
                else:
                    chaos_strength = "Weak"
            else:
                chaos_strength = "Weak"

            rows.append({
                'Province': province,
                'Channel': channel,
                'HVG λ': round(hvg_lambda, 3),
                'R²': round(r_squared, 3),
                'λ_max (Rosenstein)': round(lyap_rosenstein, 3),
                'λ_max (Wolf)': round(lyap_wolf, 3),
                'Chaos Strength': chaos_strength
            })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(f'{output_dir}/Table3_Chaos_Analysis.csv', index=False)

    print("\n" + "=" * 70)
    print("Table 3: Chaos Characterization - HVG and Lyapunov Exponent Analysis")
    print("=" * 70)
    print(f"Note: Reference HVG threshold ln(3/2) ≈ {lambda_random:.4f}")
    print(df_table.to_string(index=False))

    return df_table


def generate_table4_autoencoder_performance(ae_results, output_dir):
    """
    Table 4: Autoencoder reconstruction performance
    Columns: Province, AE Error(%), VAE Error(%), Correlation(AE), Correlation(VAE)
    """
    df_table = pd.DataFrame(ae_results)
    df_table.to_csv(f'{output_dir}/Table4_Autoencoder_Performance.csv', index=False)

    print("\n" + "=" * 70)
    print("Table 4: Autoencoder Reconstruction Performance")
    print("=" * 70)
    print("Note: Error = normalized MSE (%); Correlation = Pearson r")
    print(df_table.to_string(index=False))

    return df_table


def generate_table5_sindy_coefficients(sindy_results, output_dir):
    """
    Table 5: SINDy coefficient structure comparison across provinces
    Columns: Province, Linear(mean), Quadratic(mean), Trigonometric(mean), Active Terms, NRMSE(%)
    """
    df_table = pd.DataFrame(sindy_results)
    df_table.to_csv(f'{output_dir}/Table5_SINDy_Coefficients.csv', index=False)

    print("\n" + "=" * 70)
    print("Table 5: SINDy Coefficient Structure Comparison Across Provinces")
    print("=" * 70)
    print("Note: Mean values = average absolute coefficient magnitudes")
    print(df_table.to_string(index=False))

    return df_table


def compute_autoencoder_metrics(autoencoder, vae, X_tensor, all_labels, provinces, features_per_province, device):
    """
    计算自编码器的重建性能指标
    """
    autoencoder.eval()
    vae.eval()

    ae_results = []

    with torch.no_grad():
        X_device = X_tensor.to(device)

        # AE重建
        x_recon_ae, _ = autoencoder(X_device)
        x_recon_ae = x_recon_ae.cpu().numpy()
        X_np = X_tensor.numpy()

        # VAE重建
        x_recon_vae, _, _, _ = vae(X_device)
        x_recon_vae = x_recon_vae.cpu().numpy()

        for idx, province in enumerate(provinces):
            start_idx = idx * features_per_province
            end_idx = (idx + 1) * features_per_province

            X_province = X_np[start_idx:end_idx]
            recon_ae_province = x_recon_ae[start_idx:end_idx]
            recon_vae_province = x_recon_vae[start_idx:end_idx]

            # AE指标
            mse_ae = np.mean((X_province - recon_ae_province) ** 2)
            var_original = np.var(X_province)
            nmse_ae = (mse_ae / var_original) * 100 if var_original > 0 else 0

            # 计算相关系数（展平后计算）
            corr_ae = np.corrcoef(X_province.flatten(), recon_ae_province.flatten())[0, 1]

            # VAE指标
            mse_vae = np.mean((X_province - recon_vae_province) ** 2)
            nmse_vae = (mse_vae / var_original) * 100 if var_original > 0 else 0
            corr_vae = np.corrcoef(X_province.flatten(), recon_vae_province.flatten())[0, 1]

            ae_results.append({
                'Province': province,
                'AE Error (%)': round(nmse_ae, 1),
                'VAE Error (%)': round(nmse_vae, 1),
                'Correlation (AE)': round(corr_ae, 3),
                'Correlation (VAE)': round(corr_vae, 3)
            })

    return ae_results


def compute_sindy_metrics(sindy_models, province_latents, provinces):
    """
    计算SINDy系数的统计指标
    """
    sindy_results = []

    for province in provinces:
        if province not in sindy_models or province not in province_latents:
            continue

        model = sindy_models[province]
        latent = province_latents[province]

        # 获取系数矩阵
        coefs = model.coefficients  # shape: (n_dims, n_library)
        feature_names = model.library.feature_names

        # 分类系数
        linear_indices = []
        quadratic_indices = []
        trigonometric_indices = []

        for i, name in enumerate(feature_names):
            if name == '1':
                continue  # 常数项
            elif '^2' in name:
                quadratic_indices.append(i)
            elif '*' in name:
                quadratic_indices.append(i)  # 交互项也算quadratic
            elif 'sin' in name or 'cos' in name:
                trigonometric_indices.append(i)
            else:
                linear_indices.append(i)

        # 计算平均绝对系数
        linear_mean = np.mean(np.abs(coefs[:, linear_indices])) if linear_indices else 0
        quadratic_mean = np.mean(np.abs(coefs[:, quadratic_indices])) if quadratic_indices else 0
        trig_mean = np.mean(np.abs(coefs[:, trigonometric_indices])) if trigonometric_indices else 0

        # 计算活跃项数量
        active_terms = np.sum(np.abs(coefs) > 1e-10)
        total_terms = coefs.size

        # 计算NRMSE - 离散时间点一步预测误差
        # x(t) -> x(t+1) 预测
        X_in = latent[:-1]  # x(t), shape: (T-1, n_dims)
        X_out_actual = latent[1:]  # x(t+1) 真实值
        X_out_pred = model.predict(X_in)  # x(t+1) 预测值

        # 逐维度计算NRMSE，然后取平均
        n_dims = X_out_actual.shape[1]
        nrmse_per_dim = []

        for d in range(n_dims):
            y_true = X_out_actual[:, d]
            y_pred = X_out_pred[:, d]

            # 每个时间点的误差
            errors = y_true - y_pred

            # RMSE
            rmse = np.sqrt(np.mean(errors ** 2))

            # 归一化：使用实际值的范围 (max - min)
            y_range = np.max(y_true) - np.min(y_true)
            if y_range > 1e-10:
                nrmse_d = (rmse / y_range) * 100
            else:
                nrmse_d = 0

            nrmse_per_dim.append(nrmse_d)

        # 取所有维度的平均NRMSE
        nrmse = np.mean(nrmse_per_dim)

        sindy_results.append({
            'Province': province,
            'Linear (mean)': round(linear_mean, 3),
            'Quadratic (mean)': round(quadratic_mean, 3),
            'Trigonometric (mean)': round(trig_mean, 3),
            'Active Terms': f"{active_terms}/{total_terms}",
            'NRMSE (%)': round(nrmse, 1)
        })

    return sindy_results


def generate_all_tables(data_dict, results, output_dir,
                        autoencoder=None, vae=None, X_tensor=None,
                        all_labels=None, features_per_province=None,
                        sindy_models=None, province_latents=None, device=None):
    """
    生成所需的表格
    """
    print("\n" + "=" * 70)
    print("GENERATING ALL TABLES FOR THE PAPER")
    print("=" * 70)

    provinces = list(data_dict.keys())

    # Table 1: Descriptive Statistics
    table1 = generate_table1_descriptive_stats(data_dict, output_dir)

    # Table 2: Annual Trend
    table2 = generate_table2_annual_trend(data_dict, output_dir)

    # Table 3: Chaos Analysis (HVG + Lyapunov)
    table3 = generate_table3_chaos_analysis(results, output_dir)

    # Table 4: Autoencoder Performance
    if autoencoder is not None and vae is not None and X_tensor is not None:
        ae_results = compute_autoencoder_metrics(
            autoencoder, vae, X_tensor, all_labels,
            provinces, features_per_province, device
        )
        table4 = generate_table4_autoencoder_performance(ae_results, output_dir)
    else:
        print("\nWarning: Cannot generate Table 4 - Autoencoder models not provided")
        table4 = None

    # Table 5: SINDy Coefficients
    if sindy_models is not None and province_latents is not None:
        sindy_results = compute_sindy_metrics(sindy_models, province_latents, provinces)
        table5 = generate_table5_sindy_coefficients(sindy_results, output_dir)
    else:
        print("\nWarning: Cannot generate Table 5 - SINDy models not provided")
        table5 = None

    print("\n" + "=" * 70)
    print("ALL TABLES GENERATED SUCCESSFULLY!")
    print(f"Tables saved to: {output_dir}/")
    print("  - Table1_Descriptive_Statistics.csv")
    print("  - Table2_Annual_Trend.csv")
    print("  - Table3_Chaos_Analysis.csv")
    print("  - Table4_Autoencoder_Performance.csv")
    print("  - Table5_SINDy_Coefficients.csv")
    print("=" * 70)

    return {
        'table1': table1,
        'table2': table2,
        'table3': table3,
        'table4': table4,
        'table5': table5
    }


def save_results(results, output_dir):
    """
    保存分析结果到文件
    """
    import json

    # 转换numpy类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_json = convert_numpy(results)

    with open(f'{output_dir}/analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # 生成文本报告
    report_lines = [
        "=" * 70,
        "HVG-SINDy Autoencoder Analysis Report",
        "=" * 70,
        "",
        "1. HVG Degree Distribution Analysis",
        "-" * 40,
    ]

    for key, val in results['hvg_lambda'].items():
        report_lines.append(f"  {key}: λ = {val['lambda']:.4f}, R² = {val['r_squared']:.4f}")

    report_lines.extend([
        "",
        "2. Lyapunov Exponent Analysis",
        "-" * 40,
    ])

    for key, val in results['lyapunov'].items():
        report_lines.append(f"  {key}: λ_R = {val['rosenstein']:.4f}, λ_W = {val['wolf']:.4f}")

    report_lines.extend([
        "",
        "3. SINDy Identified Equations",
        "-" * 40,
    ])

    for province, equations in results['sindy_equations'].items():
        report_lines.append(f"\n  {province}:")
        for eq in equations:
            report_lines.append(f"    {eq}")

    report_lines.extend([
        "",
        "=" * 70,
        "Analysis Complete",
        "=" * 70,
    ])

    with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


# ============================================================================
# Main function
# ============================================================================

def main():
    """
    Main function
    """
    # Set file path
    file_paths = {
        'Beijing': './data/Beijing.xlsx',
        'Guangdong': './data/Guangdong.xlsx',
        'Henan': './data/Henan.xlsx',
        'Shanghai': './data/Shanghai.xlsx'
    }

    # Check if the file exists
    import os
    for province, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Please check file paths.")

    # Loading data
    print("Loading data...")
    data_dict = load_data(file_paths)

    # Run comprehensive analysis
    results = comprehensive_analysis(data_dict, output_dir='results')

    print("\n" + "=" * 60)
    print("All analyses complete!")
    print("Results saved to ./results/")
    print("=" * 60)


if __name__ == '__main__':

    main()
