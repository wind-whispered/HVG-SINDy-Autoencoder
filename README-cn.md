# HVG–自编码器–SINDy：文化关注度的混沌动力学

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![数据: 2014–2024](https://img.shields.io/badge/数据-2014--2024-teal.svg)]()
[![状态: 预印本](https://img.shields.io/badge/状态-预印本-orange.svg)]()

本仓库为论文 **《基于水平可见图、自编码器与稀疏辨识的集成框架揭示太极拳公众关注度的混沌动力学》** 的官方实现代码。

利用2014–2024年中国四省（北京、上海、广东、河南）的百度指数日度数据，本框架刻画了低维混沌吸引子，发现了稀疏支配方程，并与ARIMA、VAR和LSTM基线方法进行了对比评估。

---

## 研究概览

本框架采用序列化分析流程：

```
时间序列 → HVG模块 → 自编码器 → SINDy (STLS) → Lyapunov指数 / 相空间
          (12维特征)  (3维隐变量)  (稀疏方程)      (验证)
```

**核心组件：**
- **HVG模块** — 构建水平可见图，通过滑动窗口提取12维拓扑特征向量；以λ与ln(3/2) ≈ 0.405的比较作为混沌判据
- **自编码器** — 全连接编码器结构 [12→64→32→16→3]，配对对称解码器；提供VAE变体以供对比
- **SINDy (STLS)** — 从包含48项候选函数（常数项、线性项、二次项、三角函数项）的函数库中发现稀疏离散时间支配方程 $z(t+1) = f(z(t))$
- **基线方法** — ARIMA、VAR、LSTM在相同隐空间中进行公平对比
- **混沌诊断** — Rosenstein法与Wolf法Lyapunov指数、相图、庞加莱截面、关联维数

支配方程的形式如下：

$$
\begin{bmatrix} z_1(t+1) \\ z_2(t+1) \\ z_3(t+1) \end{bmatrix} ^\top= C + A \underbrace{\mathbf{z}}_{\text{线性项}} + \underbrace{\begin{bmatrix}
    \mathbf{z}^\top Q_1 \mathbf{z} \\
    \mathbf{z}^\top Q_2 \mathbf{z} \\
    \mathbf{z}^\top Q_3 \mathbf{z}
\end{bmatrix}}_{\text{二次项}} + D \underbrace{\boldsymbol{\phi}}_{\text{三角函数项}}
$$

其中

$$
\mathbf{z} = \begin{bmatrix} z_1 & z_2 & z_3 \end{bmatrix}^\top , \boldsymbol{\phi} = \begin{bmatrix} \sin z_1 & \cos z_1 & \sin z_2 & \cos z_2 & \sin z_3 & \cos z_3 \end{bmatrix}^\top
$$

---

## 主要结果

| 指标 | 数值 |
|------|------|
| HVG λ 范围 | 0.257 – 0.437（参考值：ln(3/2) ≈ 0.405） |
| 自编码器重构相关系数 | 0.896 – 0.918（VAE：0.821–0.823） |
| SINDy NRMSE（样本内） | 11.7 – 12.8% |
| SINDy 激活项数 | 30–38 / 48个候选项 |
| Rosenstein λ_max | 0.025 – 0.036 bits/天 |
| 可预测时间域 | 30 – 70 天 |

### 各省份混沌分类结果

| 省份 | 渠道 | HVG λ | λ_max（Rosenstein） | λ_max（Wolf） | 动力学状态 |
|------|------|-------|---------------------|---------------|------------|
| 北京 | PC端 | 0.291 | 0.027 | 0.171 | 强混沌 |
| 北京 | 移动端 | 0.316 | 0.033 | 0.179 | 强混沌 |
| 广东 | PC端 | 0.376 | 0.026 | 0.020 | 强混沌 |
| 广东 | 移动端 | 0.376 | 0.034 | 0.174 | 强混沌 |
| 上海 | 移动端 | 0.257 | 0.032 | 0.178 | 强混沌 |
| 河南 | PC端 | 0.415 | 0.026 | 0.016 | 准周期 |
| 河南 | 移动端 | 0.434 | 0.036 | 0.217 | 准周期 |
| 上海 | PC端 | 0.437 | 0.025 | 0.195 | 准周期 |

### 与基线方法对比（50天测试集，移动端渠道，NRMSE %）

| 方法 | 北京 | 广东 | 河南 | 上海 |
|------|------|------|------|------|
| ARIMA | 21.1 | 24.2 | 21.0 | 18.2 |
| VAR | 21.2 | 24.7 | 21.5 | 18.7 |
| LSTM | 31.1 | 28.9 | 24.2 | 25.9 |
| **SINDy（本文）** | **20.8** | **22.4** | **19.9** | **17.6** |

---

## 项目结构

```
HVG-SINDy-Autoencoder/
├── data/                  # 百度指数 Excel 数据文件
│   ├── Beijing.xlsx
│   ├── Guangdong.xlsx
│   ├── Henan.xlsx
│   └── Shanghai.xlsx
├── results/               # 输出图表、数据表、JSON报告
├── hvg_sindy.py           # 核心模块：HVG构建、AE/VAE、SINDy、Lyapunov指数
├── comparison.py          # 基线方法：ARIMA、VAR、LSTM
├── data_sequence.py       # 时间序列模式可视化
├── main.py                # 主程序入口
└── requirements.txt
```

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/wind-whispered/HVG-SINDy-Autoencoder.git
cd HVG-SINDy-Autoencoder

# 安装依赖
pip install -r requirements.txt
pip install openpyxl
```

**运行环境要求：** Python >= 3.8，PyTorch >= 2.0，CUDA 可选。

核心依赖包：

```
torch>=2.0
numpy
pandas
scipy
networkx          # HVG 构建
statsmodels       # ARIMA / VAR 基线
scikit-learn      # 数据预处理与评估指标
matplotlib
seaborn
openpyxl          # Excel 文件读取
```

---

## 数据准备

将各省份 Excel 文件放置于 `./data/` 目录下，每个文件应包含以下三列：

| 列名 | 格式 | 说明 |
|------|------|------|
| `date` | YYYYMMDD | 日期 |
| `PC` | 整数 | PC端搜索指数 |
| `Mobile` | 整数 | 移动端搜索指数 |

文件须命名为 `Beijing.xlsx`、`Guangdong.xlsx`、`Henan.xlsx`、`Shanghai.xlsx`。数据集覆盖各省份每渠道4,018条日度观测值（2014年1月1日至2024年12月31日），共计32,144条总观测记录。

---

## 使用方法

运行完整分析流程（HVG分析 → 自编码器训练 → SINDy辨识 → 对比评估 → 生成所有图表与数据表）：

```bash
python main.py
```

所有结果将保存至 `./results/` 目录：
- `figure_*.png / .pdf` — 论文所有图表
- `Table*.csv` — 描述性统计、混沌指标、自编码器性能、SINDy系数
- `analysis_results.json` — 完整数值结果

各模块脚本亦可单独运行：

```bash
# 时间序列模式可视化
python data_sequence.py

# 仅运行对比分析（需先通过 main.py 保存 SINDy 模型）
python comparison.py
```

---

## SINDy 系数结构

各省份代表性支配方程参数统计（3维隐变量空间）：

| 省份 | 线性项均值 | 二次项均值 | 三角函数项均值 | 激活项数 | NRMSE (%) |
|------|-----------|-----------|---------------|---------|-----------|
| 北京 | 0.337 | 0.150 | 0.500 | 37/48 | 12.7 |
| 广东 | 0.398 | 0.111 | 0.354 | 38/48 | 11.9 |
| 河南 | 0.222 | 0.115 | 0.287 | 38/48 | 12.8 |
| 上海 | 0.240 | 0.092 | 0.349 | 30/48 | 11.7 |

---

## 引用

```bibtex
@article{kang2026taichi,
  title   = {Chaotic Dynamics of Tai Chi Public Attention Revealed by an Integrated
             Framework of Horizontal Visibility Graphs, Autoencoders, and Sparse
             Identification},
  author  = {Kang, Yafeng and Li, Pengchao and Tang, Lu and Zhang, Chao},
  year    = {2026},
  note    = {Preprint}
}
```

---

## 参考文献

- Brunton et al. (2016) — SINDy 原始框架（[PySINDy](https://github.com/dynamicslab/pysindy)）

---

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。数据来源于百度指数，用户须自行确保符合百度的服务条款。

---
