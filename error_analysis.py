import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.cm import get_cmap
from scipy.interpolate import griddata
import pandas as pd

# 您的风速数据 (作为全局变量，方便所有函数访问)
DATA_POINTS = [
    {'Y': -2, 'Z': 4.2, 'speed': 0.67}, {'Y': -2, 'Z': 3.6, 'speed': 1.30}, {'Y': -2, 'Z': 3.0, 'speed': 2.56}, {'Y': -2, 'Z': 2.4, 'speed': 3.14}, {'Y': -2, 'Z': 1.8, 'speed': 3.23}, {'Y': -2, 'Z': 1.2, 'speed': 2.83}, {'Y': -2, 'Z': 0.6, 'speed': 1.68}, {'Y': -2, 'Z': 0.0, 'speed': 0.50},
    {'Y': -1.5, 'Z': 4.2, 'speed': 1.24}, {'Y': -1.5, 'Z': 3.6, 'speed': 3.13}, {'Y': -1.5, 'Z': 3.0, 'speed': 3.62}, {'Y': -1.5, 'Z': 2.4, 'speed': 4.07}, {'Y': -1.5, 'Z': 1.8, 'speed': 4.04}, {'Y': -1.5, 'Z': 1.2, 'speed': 3.64}, {'Y': -1.5, 'Z': 0.6, 'speed': 2.50}, {'Y': -1.5, 'Z': 0.0, 'speed': 1.60},
    {'Y': 2, 'Z': 4.2, 'speed': 1.58}, {'Y': 2, 'Z': 3.6, 'speed': 2.54}, {'Y': 2, 'Z': 3.0, 'speed': 2.52}, {'Y': 2, 'Z': 2.4, 'speed': 2.26}, {'Y': 2, 'Z': 1.8, 'speed': 1.79}, {'Y': 2, 'Z': 1.2, 'speed': 1.29}, {'Y': 2, 'Z': 0.6, 'speed': 1.15}, {'Y': 2, 'Z': 0.0, 'speed': 1.76},
    {'Y': 1.5, 'Z': 4.2, 'speed': 1.18}, {'Y': 1.5, 'Z': 3.6, 'speed': 3.08}, {'Y': 1.5, 'Z': 3.0, 'speed': 3.35}, {'Y': 1.5, 'Z': 2.4, 'speed': 3.22}, {'Y': 1.5, 'Z': 1.8, 'speed': 3.73}, {'Y': 1.5, 'Z': 1.2, 'speed': 3.34}, {'Y': 1.5, 'Z': 0.6, 'speed': 2.59}, {'Y': 1.5, 'Z': 0.0, 'speed': 2.84},
    {'Y': -1, 'Z': 4.2, 'speed': 2.04}, {'Y': -1, 'Z': 3.6, 'speed': 3.87}, {'Y': -1, 'Z': 3.0, 'speed': 4.19}, {'Y': -1, 'Z': 2.4, 'speed': 4.19}, {'Y': -1, 'Z': 1.8, 'speed': 4.06}, {'Y': -1, 'Z': 1.2, 'speed': 4.08}, {'Y': -1, 'Z': 0.6, 'speed': 3.77}, {'Y': -1, 'Z': 0.0, 'speed': 3.02},
    {'Y': 1, 'Z': 4.2, 'speed': 1.24}, {'Y': 1, 'Z': 3.6, 'speed': 3.24}, {'Y': 1, 'Z': 3.0, 'speed': 4.14}, {'Y': 1, 'Z': 2.4, 'speed': 4.19}, {'Y': 1, 'Z': 1.8, 'speed': 4.10}, {'Y': 1, 'Z': 1.2, 'speed': 3.99}, {'Y': 1, 'Z': 0.6, 'speed': 3.98}, {'Y': 1, 'Z': 0.0, 'speed': 3.89},
    {'Y': -0.5, 'Z': 4.2, 'speed': 1.78}, {'Y': -0.5, 'Z': 3.6, 'speed': 3.15}, {'Y': -0.5, 'Z': 3.0, 'speed': 4.34}, {'Y': -0.5, 'Z': 2.4, 'speed': 4.19}, {'Y': -0.5, 'Z': 1.8, 'speed': 4.09}, {'Y': -0.5, 'Z': 1.2, 'speed': 4.07}, {'Y': -0.5, 'Z': 0.6, 'speed': 4.14}, {'Y': -0.5, 'Z': 0.0, 'speed': 3.96},
    {'Y': 0.5, 'Z': 4.2, 'speed': 1.22}, {'Y': 0.5, 'Z': 3.6, 'speed': 3.53}, {'Y': 0.5, 'Z': 3.0, 'speed': 4.31}, {'Y': 0.5, 'Z': 2.4, 'speed': 4.28}, {'Y': 0.5, 'Z': 1.8, 'speed': 4.15}, {'Y': 0.5, 'Z': 1.2, 'speed': 4.09}, {'Y': 0.5, 'Z': 0.6, 'speed': 4.11}, {'Y': 0.5, 'Z': 0.0, 'speed': 4.11},
    {'Y': 0, 'Z': 4.2, 'speed': 1.19}, {'Y': 0, 'Z': 3.6, 'speed': 3.84}, {'Y': 0, 'Z': 3.0, 'speed': 4.44}, {'Y': 0, 'Z': 2.4, 'speed': 4.30}, {'Y': 0, 'Z': 1.8, 'speed': 4.17}, {'Y': 0, 'Z': 1.2, 'speed': 4.09}, {'Y': 0, 'Z': 0.6, 'speed': 4.16}, {'Y': 0, 'Z': 0.0, 'speed': 4.17},
]
TARGET_SPEED = 4.23 # m/s



# --- 字体配置函数 ---
def setup_chinese_font():
    font_name = None
    if fm.findfont('SimHei', fontext='ttf') is not None:
        font_name = 'SimHei'
    elif fm.findfont('Microsoft YaHei', fontext='ttf') is not None:
        font_name = 'Microsoft YaHei'
    elif fm.findfont('Hiragino Sans GB', fontext='ttf') is not None:
        font_name = 'Hiragino Sans GB'
    elif fm.findfont('PingFang SC', fontext='ttf') is not None:
        font_name = 'PingFang SC'
    elif fm.findfont('WenQuanYi Micro Hei', fontext='ttf') is not None:
        font_name = 'WenQuanYi Micro Hei'
    elif fm.findfont('Noto Sans CJK JP', fontext='ttf') is not None:
        font_name = 'Noto Sans CJK JP'
    elif fm.findfont('Arial Unicode MS', fontext='ttf') is not None:
        font_name = 'Arial Unicode MS'

    if font_name:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"使用字体: {font_name} 显示中文字符。")
    else:
        print("警告: 未找到合适的中文字体。中文字符可能无法正确显示。")
        print("请确保您的系统已安装中文字体，或手动设置 font_name 为已安装的字体名称。")

# --- 偏差分析函数 (保持不变) ---
def perform_deviation_analysis(data_points, target_speed, target_region_y_range, target_region_z_range, save_dir='.'):
    setup_chinese_font()
    df = pd.DataFrame(data_points)

    print(f"\n--- 对目标风速 {target_speed:.2f} m/s 进行偏差分析 ---")

    # 1. 计算每个点的偏差
    df['absolute_deviation'] = abs(df['speed'] - target_speed)
    df['percentage_deviation'] = ((df['speed'] - target_speed) / target_speed) * 100

    # 整体分析
    overall_rmse = np.sqrt(np.mean((df['speed'] - target_speed)**2))
    overall_mae = np.mean(df['absolute_deviation'])
    print(f"\n--- 全局偏差指标 ---")
    print(f"所有测量点 RMSE: {overall_rmse:.3f} m/s")
    print(f"所有测量点 MAE: {overall_mae:.3f} m/s")
    print(f"所有测量点平均风速: {df['speed'].mean():.2f} m/s")


    # 2. 核心目标区域均匀性分析
    df_target_region = df[
        (df['Y'] >= target_region_y_range[0]) & (df['Y'] <= target_region_y_range[1]) &
        (df['Z'] >= target_region_z_range[0]) & (df['Z'] <= target_region_z_range[1])
    ].copy() # 使用 .copy() 防止 SettingWithCopyWarning

    if df_target_region.empty:
        print(f"\n警告: 在核心目标区域 Y: {target_region_y_range}, Z: {target_region_z_range} 内没有找到测量点。请检查范围或数据。")
        return

    region_mean_speed = df_target_region['speed'].mean()
    region_std_dev = df_target_region['speed'].std()
    region_cv = (region_std_dev / region_mean_speed) * 100 if region_mean_speed != 0 else 0
    region_min_speed = df_target_region['speed'].min()
    region_max_speed = df_target_region['speed'].max()
    region_mean_percentage_deviation = ((region_mean_speed - target_speed) / target_speed) * 100


    print(f"\n--- 核心目标区域均匀性分析 (Y: {target_region_y_range}, Z: {target_region_z_range}) ---")
    print(f"区域内平均风速: {region_mean_speed:.3f} m/s")
    print(f"区域内平均风速相对目标偏差: {region_mean_percentage_deviation:.2f}%")
    print(f"区域内风速标准差: {region_std_dev:.3f} m/s")
    print(f"区域内风速变异系数 (CV): {region_cv:.2f}%")
    print(f"区域内最小风速: {region_min_speed:.3f} m/s")
    print(f"区域内最大风速: {region_max_speed:.3f} m/s")


    # 3. 可视化偏差
    # 3.1 偏差等值线图
    plt.figure(figsize=(10, 8))
    Y_grid, Z_grid = np.meshgrid(
        np.linspace(df['Y'].min() - 0.1, df['Y'].max() + 0.1, 200),
        np.linspace(df['Z'].min() - 0.1, df['Z'].max() + 0.1, 200)
    )
    grid_percentage_deviation = griddata((df['Y'], df['Z']), df['percentage_deviation'], (Y_grid, Z_grid), method='cubic')

    # 设定偏差颜色条范围，通常围绕0对称
    deviation_max_abs = max(abs(df['percentage_deviation'].min()), abs(df['percentage_deviation'].max()))
    dev_levels = np.linspace(-deviation_max_abs, deviation_max_abs, 20) # 20个等级
    
    # 使用RdBu颜色图，中心为白色（0偏差），两端为红色（正偏差）和蓝色（负偏差）
    dev_contourf = plt.contourf(Y_grid, Z_grid, grid_percentage_deviation, levels=dev_levels, cmap='RdBu_r', extend='both') # RdBu_r是反向的RdBu，使红色表示正偏差
    plt.colorbar(dev_contourf, label='相对偏差 (%)')
    plt.contour(Y_grid, Z_grid, grid_percentage_deviation, levels=[0], colors='black', linestyles='--', linewidths=1.5, label='0%偏差线') # 0偏差线

    # 标记目标区域
    rect = plt.Rectangle((target_region_y_range[0], target_region_z_range[0]),
                         target_region_y_range[1] - target_region_y_range[0],
                         target_region_z_range[1] - target_region_z_range[0],
                         facecolor='none', edgecolor='green', linewidth=2, linestyle=':', label='核心目标区域')
    plt.gca().add_patch(rect)

    # 绘制所有测量点，颜色表示偏差
    plt.scatter(df['Y'], df['Z'], c=df['percentage_deviation'], cmap='RdBu_r', edgecolors='black', s=100, zorder=2, label='所有测量点 (颜色表示偏差)')
    
    # 仅在核心区域的散点上方标注相对误差
    for _, row in df_target_region.iterrows():
        plt.text(row['Y'], row['Z'] + 0.1, f'{row["percentage_deviation"]:.2f}%', 
                 ha='center', va='bottom', fontsize=8, color='black', zorder=3, 
                 bbox=dict(boxstyle="round,pad=0.1", fc="yellow", ec="red", lw=0.5, alpha=0.7))


    plt.xlabel('横向位置 (Y)')
    plt.ylabel('高度 (Z)')
    plt.title(f'风速相对偏差分布图 (目标风速: {target_speed:.2f} m/s)')
    plt.grid(True, linestyle=':', alpha=0.6)
    # 更新图例以包含所有绘制元素
    handles, labels = plt.gca().get_legend_handles_labels()
    # 移除重复标签，如果scattter的label重复了
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(f'{save_dir}/deviation_contour_{target_speed:.2f}m_s.png', dpi=300, bbox_inches='tight')
    print(f"偏差等值线图已保存为 '{save_dir}/deviation_contour_{target_speed:.2f}m_s.png'")
    plt.close()

    # 3.2 偏差垂直剖面图 (这些是真正的偏差图，保持不变)
    plt.figure(figsize=(10, 7))
    unique_y_deviation_profiles = sorted(df['Y'].unique())
    colors = plt.cm.get_cmap('tab10', len(unique_y_deviation_profiles))

    for i, y_pos in enumerate(unique_y_deviation_profiles):
        profile_data = df[df['Y'] == y_pos].sort_values(by='Z')
        # 这里的X轴是 percentage_deviation
        plt.plot(profile_data['percentage_deviation'], profile_data['Z'], marker='o', linestyle='-', color=colors(i), label=f'Y = {y_pos:.1f}')
        # 标注相对误差
        for z, dev in zip(profile_data['Z'], profile_data['percentage_deviation']):
            plt.text(dev, z, 
                     f'{dev:.2f}%', fontsize=7, ha='right', va='center', color=colors(i),
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", lw=0.5, alpha=0.8))


    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='0% 偏差线') # 零偏差线
    plt.xlabel('相对偏差 (%)')
    plt.ylabel('高度 (Z)')
    plt.title(f'垂直方向相对偏差剖面图 (目标风速: {target_speed:.2f} m/s)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(unique_y_deviation_profiles)//2 + 1)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(f'{save_dir}/vertical_deviation_profiles_{target_speed:.2f}m_s.png', dpi=300, bbox_inches='tight')
    print(f"垂直偏差剖面图已保存为 '{save_dir}/vertical_deviation_profiles_{target_speed:.2f}m_s.png'")
    plt.close()

    # 3.3 偏差水平剖面图 (这些是真正的偏差图，保持不变)
    plt.figure(figsize=(10, 7))
    unique_z_deviation_profiles = sorted(df['Z'].unique())
    colors = plt.cm.get_cmap('tab10', len(unique_z_deviation_profiles))

    for i, z_pos in enumerate(unique_z_deviation_profiles):
        profile_data = df[df['Z'] == z_pos].sort_values(by='Y')
        # 这里的Y轴是 percentage_deviation
        plt.plot(profile_data['Y'], profile_data['percentage_deviation'], marker='o', linestyle='-', color=colors(i), label=f'Z = {z_pos:.1f}')
        # 标注相对误差
        for y, dev in zip(profile_data['Y'], profile_data['percentage_deviation']):
            plt.text(y, dev + 0.05, # 使用相对偏差作为Y坐标，并向上偏移
                     f'{dev:.2f}%', fontsize=7, ha='center', va='bottom', color=colors(i),
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", lw=0.5, alpha=0.8))

    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='0% 偏差线') # 零偏差线
    plt.xlabel('横向位置 (Y)')
    plt.ylabel('相对偏差 (%)')
    plt.title('水平方向相对偏差剖面图')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(unique_z_deviation_profiles)//2 + 1)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(f'{save_dir}/horizontal_deviation_profiles_{target_speed:.2f}m_s.png', dpi=300, bbox_inches='tight')
    print(f"水平偏差剖面图已保存为 '{save_dir}/horizontal_deviation_profiles_{target_speed:.2f}m_s.png'")
    plt.close()

# --- 主函数：调用所有绘图和分析函数 ---
def analyze_and_plot_wind_field(data_points, target_speed):
    """
    对风场数据进行可视化分析，现在只进行偏差分析并生成偏差相关的图表。
    """
    print("\n--- 开始生成风场可视化图表和偏差分析 ---")

    # 只执行偏差分析
    # 更新核心区域的Y和Z范围
    core_region_y = [-1.0, 1.0]
    core_region_z = [1.2, 3.0] # 对应WS4, WS7, WS2, WS3的高度

    perform_deviation_analysis(data_points, target_speed, core_region_y, core_region_z)
    
    print("\n--- 所有图表和分析生成完毕 ---")

# --- 调用主函数进行分析和绘图 ---
analyze_and_plot_wind_field(DATA_POINTS, TARGET_SPEED)