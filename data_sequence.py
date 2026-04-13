"""
Figure 2: Temporal Patterns of Taijiquan Search Index
(a) Raw time series for four provinces (PC: blue; Mobile: red)
(b) Annual mean trends by province
(c) Mobile/PC ratio evolution (2014-2024)
(d) Cross-province correlation heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import seaborn as sns

# Set up matplotlib for Language characters and publication quality
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# Load data
data_path = './data/'
files = {
    'Beijing': 'Beijing.xlsx',
    'Guangdong': 'Guangdong.xlsx',
    'Henan': 'Henan.xlsx',
    'Shanghai': 'Shanghai.xlsx'
}

# Chinese province names for labels
province_cn = {
    'Beijing': 'Beijing ',
    'Guangdong': 'Guangdong',
    'Henan': 'Henan',
    'Shanghai': 'Shanghai'
}

all_data = {}
for province, filename in files.items():
    df = pd.read_excel(data_path + filename)
    df.columns = ['date', 'PC', 'Mobile']
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date')
    all_data[province] = df

# Create figure with 2x2 subplots
fig = plt.figure(figsize=(12, 10))

# Define colors
colors = {
    'Beijing': '#E64B35',  # Red
    'Guangdong': '#4DBBD5',  # Cyan
    'Henan': '#00A087',  # Green
    'Shanghai': '#3C5488'  # Blue
}

pc_color = '#3C5488'  # Blue for PC
mobile_color = '#E64B35'  # Red for Mobile

# Key events
covid_date = pd.Timestamp('2020-01-23')
unesco_date = pd.Timestamp('2020-12-17')

# ============================================
# Panel (a): Raw time series - 2x2 grid
# ============================================
provinces_order = ['Beijing', 'Guangdong', 'Henan', 'Shanghai']

for idx, province in enumerate(provinces_order):
    ax = fig.add_subplot(3, 2, idx + 1 if idx < 2 else idx + 1)

    df = all_data[province]

    # Plot PC and Mobile
    ax.plot(df.index, df['PC'], color=pc_color, linewidth=0.5, alpha=0.8, label='PC')
    ax.plot(df.index, df['Mobile'], color=mobile_color, linewidth=0.5, alpha=0.8, label='Mobile')

    # Add event annotations (only on first subplot to avoid clutter)
    if idx == 0:
        ax.axvline(x=covid_date, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=unesco_date, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.annotate('COVID-19', xy=(covid_date, ax.get_ylim()[1] * 0.9),
                    fontsize=7, ha='right', color='gray')
        ax.annotate('UNESCO', xy=(unesco_date, ax.get_ylim()[1] * 0.9),
                    fontsize=7, ha='left', color='gray')
    else:
        ax.axvline(x=covid_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=unesco_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_title(f'{province_cn[province]}', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Search Index')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(df.index[0], df.index[-1])

    if idx == 0:
        ax.legend(loc='upper right', framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')

# Adjust layout for top 4 panels
plt.tight_layout()

# ============================================
# Panel (b): Annual mean trends
# ============================================
ax_b = fig.add_subplot(3, 2, 5)

# Calculate annual means (Total = PC + Mobile)
annual_data = {}
years = range(2014, 2025)

for province in provinces_order:
    df = all_data[province]
    df['Total'] = df['PC'] + df['Mobile']
    annual_means = df.groupby(df.index.year)['Total'].mean()
    annual_data[province] = annual_means

# Plot annual trends
for province in provinces_order:
    ax_b.plot(list(years), [annual_data[province].get(y, np.nan) for y in years],
              color=colors[province], marker='o', markersize=5, linewidth=2,
              label=province)

ax_b.set_xlabel('Year')
ax_b.set_ylabel('Annual Mean (Total)')
ax_b.set_title('(b) Annual Mean Trends', fontweight='bold')
ax_b.legend(loc='upper right', ncol=2, framealpha=0.9)
ax_b.set_xticks(list(years))
ax_b.set_xticklabels([str(y) for y in years], rotation=45)
ax_b.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax_b.set_facecolor('#fafafa')

# ============================================
# Panel (c): Mobile/PC ratio evolution
# ============================================
ax_c = fig.add_subplot(3, 2, 6)

# Calculate annual Mobile/PC ratio
ratio_data = {}
for province in provinces_order:
    df = all_data[province]
    annual_pc = df.groupby(df.index.year)['PC'].mean()
    annual_mobile = df.groupby(df.index.year)['Mobile'].mean()
    ratio_data[province] = annual_mobile / annual_pc

# Bar plot with grouped bars
x = np.arange(len(years))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, province in enumerate(provinces_order):
    ratios = [ratio_data[province].get(y, np.nan) for y in years]
    ax_c.bar(x + offsets[i] * width, ratios, width,
             color=colors[province], label=province, alpha=0.85)

ax_c.set_xlabel('Year')
ax_c.set_ylabel('Mobile/PC Ratio')
ax_c.set_title('(c) Mobile/PC Ratio Evolution', fontweight='bold')
ax_c.set_xticks(x)
ax_c.set_xticklabels([str(y) for y in years], rotation=45)
ax_c.legend(loc='upper left', ncol=2, framealpha=0.9)
ax_c.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
ax_c.set_facecolor('#fafafa')

# Adjust layout
plt.tight_layout()

# ============================================
# Save first version (a, b, c panels)
# ============================================
# We need to reorganize to fit (d) - let's create a new figure with proper layout

plt.close()

# Create new figure with GridSpec for better control
fig = plt.figure(figsize=(14, 12))
from matplotlib.gridspec import GridSpec

gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

# Panel (a): 2x2 time series in top 2 rows, left side
ax_a1 = fig.add_subplot(gs[0, 0:2])  # Beijing
ax_a2 = fig.add_subplot(gs[0, 2:4])  # Guangdong
ax_a3 = fig.add_subplot(gs[1, 0:2])  # Henan
ax_a4 = fig.add_subplot(gs[1, 2:4])  # Shanghai

axes_ts = [ax_a1, ax_a2, ax_a3, ax_a4]

for idx, (ax, province) in enumerate(zip(axes_ts, provinces_order)):
    df = all_data[province]

    # Plot PC and Mobile
    ax.plot(df.index, df['PC'], color=pc_color, linewidth=0.6, alpha=0.85, label='PC')
    ax.plot(df.index, df['Mobile'], color=mobile_color, linewidth=0.6, alpha=0.85, label='Mobile')

    # Add event lines
    ymin, ymax = df['PC'].min(), df['Mobile'].max() * 1.1
    ax.set_ylim(0, ymax)

    ax.axvline(x=covid_date, color='#666666', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axvline(x=unesco_date, color='#666666', linestyle='--', linewidth=1.2, alpha=0.8)

    # Annotations
    if idx == 0:
        ax.annotate('COVID-19\n2020.01', xy=(covid_date, ymax * 0.85),
                    fontsize=7, ha='right', va='top', color='#444444')
        ax.annotate('UNESCO\n2020.12', xy=(unesco_date, ymax * 0.85),
                    fontsize=7, ha='left', va='top', color='#444444')

    ax.set_title(f'{province_cn[province]}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Search Index', fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(df.index[0], df.index[-1])

    if idx == 0:
        ax.legend(loc='upper right', framealpha=0.95, fontsize=8)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')

    # Add panel label
    panel_labels = ['(a-i)', '(a-ii)', '(a-iii)', '(a-iv)']
    ax.text(0.02, 0.95, panel_labels[idx], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

# Panel (b): Annual mean trends
ax_b = fig.add_subplot(gs[2, 0:2])

for province in provinces_order:
    ax_b.plot(list(years), [annual_data[province].get(y, np.nan) for y in years],
              color=colors[province], marker='o', markersize=6, linewidth=2.5,
              label=province, alpha=0.9)

ax_b.set_xlabel('Year', fontsize=10)
ax_b.set_ylabel('Annual Mean (PC + Mobile)', fontsize=10)
ax_b.set_title('', fontsize=11)
ax_b.legend(loc='upper right', ncol=2, framealpha=0.95, fontsize=8)
ax_b.set_xticks(list(years))
ax_b.set_xticklabels([str(y) for y in years], rotation=45)
ax_b.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax_b.set_facecolor('#fafafa')
ax_b.text(0.02, 0.95, '(b)', transform=ax_b.transAxes,
          fontsize=12, fontweight='bold', va='top')

# Panel (c): Mobile/PC ratio - line plot for cleaner look
ax_c = fig.add_subplot(gs[2, 2])

for province in provinces_order:
    ratios = [ratio_data[province].get(y, np.nan) for y in years]
    ax_c.plot(list(years), ratios, color=colors[province], marker='s',
              markersize=5, linewidth=2, label=province, alpha=0.9)

ax_c.set_xlabel('Year', fontsize=10)
ax_c.set_ylabel('Mobile/PC Ratio', fontsize=10)
ax_c.set_title('', fontsize=11)
ax_c.legend(loc='upper right',ncol=2, fontsize=7, framealpha=0.95)
ax_c.set_xticks(list(years)[::2])
ax_c.set_xticklabels([str(y) for y in list(years)[::2]], rotation=45)
ax_c.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax_c.set_facecolor('#fafafa')
ax_c.text(0.02, 0.95, '(c)', transform=ax_c.transAxes,
          fontsize=12, fontweight='bold', va='top')

# Panel (d): Correlation heatmap
ax_d = fig.add_subplot(gs[2, 3])

# Calculate correlation matrix for Mobile channel
corr_matrix = pd.DataFrame(index=provinces_order, columns=provinces_order, dtype=float)
for p1 in provinces_order:
    for p2 in provinces_order:
        corr_matrix.loc[p1, p2] = all_data[p1]['Mobile'].corr(all_data[p2]['Mobile'])

# Plot heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.3f', cmap='RdYlBu_r',
            vmin=0.85, vmax=1.0, ax=ax_d, square=True,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            annot_kws={'fontsize': 9},
            linewidths=0.5, linecolor='white')

ax_d.set_title('', fontsize=11)
ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax_d.set_yticklabels(ax_d.get_yticklabels(), rotation=0, fontsize=8)
ax_d.text(-0.15, 1.02, '(d)', transform=ax_d.transAxes,
          fontsize=12, fontweight='bold', va='bottom')

# Add main title
fig.suptitle('Figure 2. Temporal Patterns of Taijiquan Search Index\n' +
             '(a) Raw time series; (b) Annual trends; (c) Mobile/PC ratio; (d) Cross-province correlation',
             fontsize=13, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = 'results/Figure2_Temporal_Patterns.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"Figure saved to: {output_path}")
print(f"PDF version saved to: {output_path.replace('.png', '.pdf')}")

# Also display summary statistics
print("\n" + "=" * 60)
print("Summary Statistics for Figure 2")
print("=" * 60)

print("\n(b) Annual Mean Trends (PC + Mobile):")
for province in provinces_order:
    start = annual_data[province].get(2014, np.nan)
    peak = annual_data[province].max()
    peak_year = annual_data[province].idxmax()
    end = annual_data[province].get(2024, np.nan)
    change = (end - peak) / peak * 100
    print(f"  {province}: Peak {peak:.0f} ({peak_year}) → {end:.0f} (2024), Δ = {change:.1f}%")

print("\n(c) Mobile/PC Ratio Evolution:")
for province in provinces_order:
    r2014 = ratio_data[province].get(2014, np.nan)
    r2024 = ratio_data[province].get(2024, np.nan)
    print(f"  {province}: {r2014:.2f} (2014) → {r2024:.2f} (2024)")

print("\n(d) Correlation Matrix (Mobile channel):")
print(corr_matrix.round(3).to_string())

plt.show()