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

# --- 字体配置函数 (保持不变) ---
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

# --- 1. 等风速图函数 (保持不变) ---
def plot_wind_speed_contour(data_points, target_speed, save_path):
    setup_chinese_font()

    Y = np.array([d['Y'] for d in data_points])
    Z = np.array([d['Z'] for d in data_points])
    speeds = np.array([d['speed'] for d in data_points])

    if len(data_points) == 0:
        print("等风速图：没有可绘制的数据。")
        return

    y_min, y_max = Y.min() - 0.1, Y.max() + 0.1
    z_min, z_max = Z.min() - 0.1, Z.max() + 0.1

    grid_y, grid_z = np.meshgrid(
        np.linspace(y_min, y_max, 200),
        np.linspace(z_min, z_max, 200)
    )

    grid_speeds = griddata((Y, Z), speeds, (grid_y, grid_z), method='cubic')

    vmin_display = 0.0
    vmax_display = 5.0
    
    cmap = get_cmap('jet')
    norm = plt.Normalize(vmin=vmin_display, vmax=vmax_display)

    plt.figure(figsize=(10, 8))

    contourf_plot = plt.contourf(grid_y, grid_z, grid_speeds, levels=np.linspace(vmin_display, vmax_display, 20), cmap=cmap, extend='both')

    line_contour = plt.contour(grid_y, grid_z, grid_speeds, levels=np.linspace(vmin_display, vmax_display, 10), colors='black', linewidths=0.8, alpha=0.6)
    plt.clabel(line_contour, inline=True, fontsize=8, fmt='%1.2f')

    target_contour = plt.contour(grid_y, grid_z, grid_speeds, levels=[target_speed], colors='red', linewidths=2.5, linestyles='--', zorder=3)
    plt.clabel(target_contour, inline=True, fontsize=10, fmt='目标: %1.2f m/s', colors='red')

    plt.scatter(Y, Z, color='gray', s=50, marker='x', label='测量点', zorder=4)
    for i, txt in enumerate(speeds):
        plt.text(Y[i], Z[i], f'{txt:.2f}', ha='center', va='bottom', fontsize=8, color='black', zorder=5)

    plt.xlabel('横向位置 (Y)')
    plt.ylabel('高度 (Z)')
    plt.title(f'模拟风速 {target_speed}m/s 风场等风速分布图')

    plt.xlim(y_min, y_max)
    plt.ylim(z_min, z_max)
    
    plt.xticks(np.unique(Y))
    plt.yticks(np.unique(Z))

    plt.grid(True, linestyle=':', alpha=0.6)

    cbar = plt.colorbar(contourf_plot, pad=0.02)
    cbar.set_label('风速 (m/s)')

    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"等风速图已保存为 '{save_path}'")
    plt.close()

# --- 2. 垂直风速剖面图函数 (图例调整) ---
def plot_vertical_profiles(data_points, target_speed, save_path):
    setup_chinese_font()
    
    df = pd.DataFrame(data_points)
    
    unique_y_positions = sorted(df['Y'].unique())

    # 调整图幅大小，为下方图例预留空间
    plt.figure(figsize=(10, 7)) # 略微增加高度

    colors = plt.cm.get_cmap('tab10', len(unique_y_positions))

    for i, y_pos in enumerate(unique_y_positions):
        profile_data = df[df['Y'] == y_pos].sort_values(by='Z')
        plt.plot(profile_data['speed'], profile_data['Z'], marker='o', linestyle='-', color=colors(i), label=f'Y = {y_pos:.1f}')
        for z, speed in zip(profile_data['Z'], profile_data['speed']):
            plt.text(speed, z, f'{speed:.2f}', fontsize=7, ha='left', va='center', color=colors(i))

    plt.axvline(x=target_speed, color='red', linestyle='--', linewidth=2, label=f'目标风速 ({target_speed:.2f} m/s)')

    plt.xlabel('风速 (m/s)')
    plt.ylabel('高度 (Z)')
    plt.title('垂直风速剖面图')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # *** 核心修改：调整图例位置到下方 ***
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(unique_y_positions)//2 + 1) # 根据图例数量调整列数
    
    # 调整布局以适应下方图例。rect=[left, bottom, right, top] 调整绘图区域
    # bottom 参数从 0.0 增加，为图例留出空间
    plt.tight_layout(rect=[0, 0.15, 1, 1]) # 增大 bottom 留出更多空间

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"垂直风速剖面图已保存为 '{save_path}'")
    plt.close()

# --- 3. 水平风速剖面图函数 (图例调整) ---
def plot_horizontal_profiles(data_points, target_speed, save_path):
    setup_chinese_font()
    
    df = pd.DataFrame(data_points)

    unique_z_positions = sorted(df['Z'].unique())

    # 调整图幅大小，为下方图例预留空间
    plt.figure(figsize=(10, 7)) # 略微增加高度
    
    colors = plt.cm.get_cmap('tab10', len(unique_z_positions))

    for i, z_pos in enumerate(unique_z_positions):
        profile_data = df[df['Z'] == z_pos].sort_values(by='Y')
        plt.plot(profile_data['Y'], profile_data['speed'], marker='o', linestyle='-', color=colors(i), label=f'Z = {z_pos:.1f}')
        for y, speed in zip(profile_data['Y'], profile_data['speed']):
            plt.text(y, speed, f'{speed:.2f}', fontsize=7, ha='center', va='bottom', color=colors(i))

    plt.axhline(y=target_speed, color='red', linestyle='--', linewidth=2, label=f'目标风速 ({target_speed:.2f} m/s)')

    plt.xlabel('横向位置 (Y)')
    plt.ylabel('风速 (m/s)')
    plt.title('水平风速剖面图')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # *** 核心修改：调整图例位置到下方 ***
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(unique_z_positions)//2 + 1) # 根据图例数量调整列数
    
    # 调整布局以适应下方图例
    plt.tight_layout(rect=[0, 0.15, 1, 1]) # 增大 bottom 留出更多空间

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"水平风速剖面图已保存为 '{save_path}'")
    plt.close()

# --- 主函数：调用所有绘图函数 ---
def analyze_and_plot_wind_field(data_points, target_speed):
    """
    对风场数据进行可视化分析，生成等风速图、垂直剖面图和水平剖面图。
    """
    print("\n--- 开始生成风场可视化图表 ---")

    # 1. 绘制等风速图
    plot_wind_speed_contour(data_points, target_speed, 'wind_speed_contour_plot.png')

    # 2. 绘制垂直风速剖面图
    plot_vertical_profiles(data_points, target_speed, 'vertical_wind_profiles.png')

    # 3. 绘制水平风速剖面图
    plot_horizontal_profiles(data_points, target_speed, 'horizontal_wind_profiles.png')
    
    print("\n--- 所有图表生成完毕 ---")

# --- 调用主函数进行分析和绘图 ---
analyze_and_plot_wind_field(DATA_POINTS, TARGET_SPEED)