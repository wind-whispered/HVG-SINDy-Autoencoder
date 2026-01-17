#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

HVG-SINDy Autoencoder for Taijiquan Public Attention Dynamics and System Evolution Modeling

data: 2026.1.1

该代码实现以下功能:
1. 绘制HVG，并利用HVG度分布指数λ作为混沌判据
2. 利用时间序列构造HVG特征向量
3. 用Autoencoder学习低维演化状态
4. 在潜空间上用离散SINDy识别动力学
5. 估计Lyapunov指数、绘制相图、Poincaré截面
6. 结合参数进行分析
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

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# 第一部分: 数据加载与预处理
# ============================================================================

def load_data(file_paths):
    """
    加载四个省份的百度指数数据

    Parameters:
    -----------
    file_paths : dict
        省份名称到文件路径的映射

    Returns:
    --------
    data_dict : dict
        包含各省份数据的字典
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
    数据预处理: 平滑、归一化

    Parameters:
    -----------
    data_dict : dict
        原始数据字典
    smooth_window : int
        移动平均窗口大小

    Returns:
    --------
    processed_dict : dict
        处理后的数据字典
    """
    processed_dict = {}
    for province, df in data_dict.items():
        # 移动平均平滑
        df_smooth = df.rolling(window=smooth_window, center=True).mean().dropna()
        # 标准化
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
# 第二部分: 水平可见图 (Horizontal Visibility Graph, HVG)
# ============================================================================

def construct_hvg(time_series):
    """
    构建水平可见图 (HVG)

    两个节点i和j在HVG中相连，当且仅当:
    对于所有 i < k < j, 满足 x_k < min(x_i, x_j)

    Parameters:
    -----------
    time_series : array-like
        时间序列数据

    Returns:
    --------
    G : networkx.Graph
        水平可见图
    """
    n = len(time_series)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        # 向右扫描
        for j in range(i + 1, n):
            # 检查可见性条件
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
    快速构建HVG的优化算法
    使用分治策略降低时间复杂度

    Parameters:
    -----------
    time_series : array-like
        时间序列数据

    Returns:
    --------
    G : networkx.Graph
        水平可见图
    """
    n = len(time_series)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []

    def divide_and_conquer(left, right):
        if left >= right:
            return

        # 找到区间内的最大值索引
        max_idx = left + np.argmax(time_series[left:right + 1])

        # 向左连接
        for i in range(max_idx - 1, left - 1, -1):
            visible = True
            min_val = min(time_series[i], time_series[max_idx])
            for k in range(i + 1, max_idx):
                if time_series[k] >= min_val:
                    visible = False
                    break
            if visible:
                edges.append((i, max_idx))

        # 向右连接
        for j in range(max_idx + 1, right + 1):
            visible = True
            min_val = min(time_series[max_idx], time_series[j])
            for k in range(max_idx + 1, j):
                if time_series[k] >= min_val:
                    visible = False
                    break
            if visible:
                edges.append((max_idx, j))

        # 递归处理左右子区间
        divide_and_conquer(left, max_idx - 1)
        divide_and_conquer(max_idx + 1, right)

    divide_and_conquer(0, n - 1)
    G.add_edges_from(edges)

    return G


def calculate_degree_distribution(G):
    """
    计算图的度分布

    Parameters:
    -----------
    G : networkx.Graph
        图

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
# 第三部分: HVG特征向量构造
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

    # ====== 关键改动1：用log域线性回归同时拟合截距A和lambda ======
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

    # ====== 关键改动2：随机参考线也画成指数曲线，并在中位k处对齐 ======
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
# 第四部分: Autoencoder 模型
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
# 第五部分: SINDy (Sparse Identification of Nonlinear Dynamics)
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
# 第六部分: 动力学分析 - Lyapunov指数、相图、Poincaré截面
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
    ax1.grid(True, alpha=0.3)

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
# 第七部分: 综合分析与参数研究
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
        'sindy_equations': {}
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

    for idx, province in enumerate(province_list):
        print(f"\nIdentifying dynamics for {province}...")

        start_idx = idx * features_per_province
        end_idx = (idx + 1) * features_per_province
        latent_province = latent_ae[start_idx:end_idx]

        # SINDy
        library = SINDyLibrary(poly_order=2, include_sine=True, include_interaction=True)
        sindy = DiscreteSINDy(library, threshold=0.05, max_iter=20)
        sindy.fit(latent_province)

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
        "太极拳公众关注度动态与系统演化建模",
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
# 主程序
# ============================================================================

def main():
    """
    主函数
    """
    # 设置文件路径
    file_paths = {
        'Beijing': 'Beijing.xlsx',
        'Guangdong': 'Guangdong.xlsx',
        'Henan': 'Henan.xlsx',
        'Shanghai': 'Shanghai.xlsx'
    }

    # 检查文件是否存在
    import os
    for province, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Please check file paths.")

    # 加载数据
    print("Loading data...")
    data_dict = load_data(file_paths)

    # 运行综合分析
    results = comprehensive_analysis(data_dict, output_dir='./results')

    print("\n" + "=" * 60)
    print("All analyses complete!")
    print("Results saved to ./results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
