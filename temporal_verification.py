import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches # 用于绘制矩形
import os
import re # For regex to parse filenames
from scipy.interpolate import griddata

# --- 字体配置函数 ---
def setup_chinese_font():
    """
    配置 Matplotlib 以正确显示中文字符。
    尝试查找并使用系统上可用的中文字体。
    """
    font_name = None
    # 优先选择常用中文字体
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'Hiragino Sans GB', 'PingFang SC',
        'WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'Arial Unicode MS'
    ]
    for font_c in font_candidates:
        if fm.findfont(font_c, fontext='ttf') is not None:
            font_name = font_c
            break

    if font_name:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
        print(f"使用字体: {font_name} 显示中文字符。")
    else:
        print("警告: 未找到合适的中文字体。中文字符可能无法正确显示。")
        print("请确保您的系统已安装中文字体，或手动设置 font_name 为已安装的字体名称。")

# --- 1. 配置信息 (请根据您的实际情况修改) ---
# EXCEL_FILES_DIRECTORY 现在指向包含所有Excel文件的目录。
# 确保这个目录下有所有符合命名规则的Excel文件。
# 请根据您之前确认的工作目录和文件路径来设置
EXCEL_FILES_DIRECTORY = './ws_check_data/v4.23/wind_data_files/'
OUTPUT_DIR = 'wind_analysis_results' # 结果图表和报告的输出目录

# 定义每个风速传感器名称对应的Z坐标。
# 根据您的说明：从低到高分别是 0.0m 到 4.2m，等间距 0.6m。
# 注意：现在只有8个风速列（排除了体速1），所以Z坐标映射是 0.0 到 4.2，共8个点。
SENSOR_Z_MAPPING = {
    # '体速1 /m/s': 0.0, # <-- 已移除 '体速1 /m/s'
    '体速2 /m/s': 0.0, # 原本是 0.6，现在是最低点 0.0
    '风速8 /m/s': 0.6,
    '风速4 /m/s': 1.2,
    '风速7 /m/s': 1.8,
    '风速2 /m/s': 2.4,
    '风速3 /m/s': 3.0,
    '风速6 /m/s': 3.6,
    '风速5 /m/s': 4.2  # 最高点
}
# 为了方便绘制剖面图和箱线图，我们需要按Z坐标排序传感器
SORTED_Z_POINTS = sorted(SENSOR_Z_MAPPING.items(), key=lambda item: item[1])
# 得到一个列表，例如 [('体速2 /m/s', 0.0), ('风速8 /m/s', 0.6), ...]

# Excel文件中实际包含风速数据的列名。
WIND_SPEED_COLUMNS = [
    '体速2 /m/s', '风速8 /m/s', '风速4 /m/s',
    '风速7 /m/s', '风速2 /m/s', '风速3 /m/s', '风速6 /m/s', '风速5 /m/s'
]

# 时间戳列的名称
TIME_COLUMN_NAME = 'Sensor Sample Time' 

# 数据筛选参数
DATA_POINTS_TO_ANALYZE = 240 # 选取的数据点数量 (对应中间4分钟)

# *** 新增：核心区域定义 ***
CORE_REGION_Y = [-1.0, 1.0] # 横向 Y 范围
CORE_REGION_Z = [1.2, 3.0]  # 高度 Z 范围，对应WS4, WS7, WS2, WS3的高度

# 创建结果输出目录，如果它不存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 辅助函数：将文本内容保存到文件 ---
def save_text_to_file(content, filename, output_dir):
    """
    将给定的文本内容保存到指定文件。
    """
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"统计数据已保存到 '{filepath}'")

# --- 2. 数据读取与预处理函数 ---
def load_and_preprocess_data(excel_dir, data_points_to_analyze):
    """
    加载Excel数据并进行预处理。
    遍历指定目录下的所有Excel文件，从文件名解析Y坐标和目标风速。
    对每个文件的DataFrame，截取中间指定数量的数据点进行分析。
    返回一个列表，每个元素是一个字典，包含DataFrame和其对应的元数据。
    """
    setup_chinese_font() # 确保在数据加载前配置字体
    all_simulation_data_list = []
    
    # 获取目录中所有以 .xlsx 或 .xls 结尾的文件
    excel_files = [f for f in os.listdir(excel_dir) if f.endswith('.xlsx') or f.endswith('.xls')]
    
    if not excel_files:
        print(f"在目录 '{excel_dir}' 中未找到任何Excel文件。请检查路径或文件类型。")
        return []

    for filename in sorted(excel_files): # 按文件名排序，方便查看处理顺序
        file_path = os.path.join(excel_dir, filename)
        print(f"正在读取文件: {filename}")
        
        # 从文件名解析目标风速和Y坐标，例如 "2.58mps_y0.5.xlsx"
        match = re.match(r'(\d+\.\d+)mps_y([-\d.]+)\.xlsx', filename)
        if not match:
            print(f"警告: 文件名 '{filename}' 不符合 'X.XXmps_yY.Y.xlsx' 格式，跳过此文件。")
            continue
        
        try:
            target_speed = float(match.group(1))
            y_coordinate = float(match.group(2))
        except ValueError:
            print(f"警告: 无法从文件名 '{filename}' 解析数字（风速或Y坐标），跳过此文件。")
            continue

        try:
            # 读取第一个Sheet（默认是Sheet1，或索引为0）
            df_full = pd.read_excel(file_path, sheet_name=0) 
            
            # 去除所有列名的前后空格
            df_full.columns = df_full.columns.str.strip()

            # 检查时间戳列是否存在
            if TIME_COLUMN_NAME not in df_full.columns:
                print(f"警告: 文件 '{filename}' 未找到时间戳列 '{TIME_COLUMN_NAME}'，跳过此文件。")
                print(f"文件 '{filename}' 中的实际列名有: {df_full.columns.tolist()}")
                continue
            
            # 检查是否有预期的风速列
            actual_speed_columns_in_file = [col for col in WIND_SPEED_COLUMNS if col in df_full.columns]
            if not actual_speed_columns_in_file:
                print(f"警告: 文件 '{filename}' 未找到任何预期的风速列，跳过此文件。")
                continue

            # 选取中间的 data_points_to_analyze 条数据
            total_rows = len(df_full)
            if total_rows < data_points_to_analyze:
                print(f"警告: 文件 '{filename}' 总行数 ({total_rows}) 少于设定的分析点数 ({data_points_to_analyze})。将使用所有可用数据。")
                df_selected = df_full
            else:
                start_index = (total_rows - data_points_to_analyze) // 2
                end_index = start_index + data_points_to_analyze
                df_selected = df_full.iloc[start_index:end_index].copy()
                print(f"已从文件 '{filename}' 中选取中间 {data_points_to_analyze} 条数据 (行 {start_index} 到 {end_index-1}) 进行分析。")

            # 准备每个测量点的Y, Z坐标映射
            point_coords_for_file = {}
            for col_name in actual_speed_columns_in_file:
                if col_name in SENSOR_Z_MAPPING:
                    point_coords_for_file[col_name] = {'Y': y_coordinate, 'Z': SENSOR_Z_MAPPING[col_name]}
                else:
                    print(f"警告: 传感器 '{col_name}' 未在 SENSOR_Z_MAPPING 中定义Z坐标，将被忽略。")

            if not point_coords_for_file:
                print(f"文件 '{filename}' 中没有有效风速点位进行分析，跳过。")
                continue

            # 提取时间列和所有实际存在的风速列
            processed_df = df_selected[[TIME_COLUMN_NAME] + actual_speed_columns_in_file].copy()
            processed_df.rename(columns={TIME_COLUMN_NAME: 'Time'}, inplace=True) # 重命名时间列为'Time'

            # 存储处理后的数据和元数据
            all_simulation_data_list.append({
                'df': processed_df,
                'target_speed': target_speed,
                'y_coordinate': y_coordinate,
                'point_coords': point_coords_for_file,
                'filename': filename # 用于日志或调试
            })

        except Exception as e:
            print(f"读取文件 '{filename}' 时发生错误: {e}")
            continue
            
    return all_simulation_data_list

# --- 3. 风场均匀性分析 (空间均匀性) ---
def analyze_spatial_uniformity(sim_data_list):
    """
    分析不同目标风速下的空间均匀性。
    将相同目标风速但不同Y坐标的数据聚合起来进行空间分析。
    """
    print(f"\n--- 正在进行空间均匀性分析 ---")
    
    aggregated_data_by_target_speed = {} 
    
    for item in sim_data_list:
        target_speed = item['target_speed']
        df = item['df'] 
        point_coords = item['point_coords']

        if target_speed not in aggregated_data_by_target_speed:
            aggregated_data_by_target_speed[target_speed] = [] 
        
        mean_speeds_for_file = df.drop(columns='Time').mean()

        for col_name, mean_speed in mean_speeds_for_file.items():
            if col_name in point_coords:
                coords = point_coords[col_name]
                deviation = mean_speed - target_speed
                percentage_deviation = (deviation / target_speed) * 100 if target_speed != 0 else np.nan
                
                aggregated_data_by_target_speed[target_speed].append({
                    'Y': coords['Y'],
                    'Z': coords['Z'],
                    'PointName': col_name,
                    'MeanSpeed': mean_speed,
                    'Deviation': deviation,
                    'PercentageDeviation': percentage_deviation
                })

    for target_speed, points_data in aggregated_data_by_target_speed.items():
        spatial_df = pd.DataFrame(points_data)
        
        if spatial_df.empty:
            print(f"目标风速 {target_speed:.2f} m/s 没有可用于空间均匀性分析的点位数据。")
            continue

        print(f"\n--- 目标风速 {target_speed:.2f} m/s 的空间均匀性统计 ---")
        
        spatial_df_sorted = spatial_df.sort_values(by=['Y', 'Z'])
        spatial_stats_output = ["空间均匀性点位统计 (各点平均风速):", spatial_df_sorted[['Y', 'Z', 'PointName', 'MeanSpeed', 'PercentageDeviation']].to_string()]

        overall_mean_speed_spatial = spatial_df['MeanSpeed'].mean()
        overall_avg_dev_perc_spatial = ((overall_mean_speed_spatial - target_speed) / target_speed) * 100
        overall_std_spatial = spatial_df['MeanSpeed'].std() 
        overall_cv_spatial = (overall_std_spatial / overall_mean_speed_spatial) * 100 if overall_mean_speed_spatial != 0 else np.nan

        spatial_stats_output.append(f"\n整体平均风速: {overall_mean_speed_spatial:.3f} m/s")
        spatial_stats_output.append(f"整体平均风速相对目标偏差: {overall_avg_dev_perc_spatial:.2f}%")
        spatial_stats_output.append(f"整体风速标准差 (空间均匀性): {overall_std_spatial:.3f} m/s")
        spatial_stats_output.append(f"整体风速变异系数 (空间均匀性CoV): {overall_cv_spatial:.2f}%")
        
        save_text_to_file("\n".join(spatial_stats_output), 
                           f'spatial_uniformity_stats_{target_speed:.2f}m_s.txt', 
                           OUTPUT_DIR)
        print("\n".join(spatial_stats_output)) 

        # 绘制相对偏差等值线图 (现有图表)
        plt.figure(figsize=(10, 8))
        Y_coords = spatial_df['Y'].values
        Z_coords = spatial_df['Z'].values
        deviations_perc = spatial_df['PercentageDeviation'].values

        if len(Y_coords) >= 3 and len(np.unique(Y_coords)) > 1 and len(np.unique(Z_coords)) > 1:
            y_range = Y_coords.max() - Y_coords.min()
            z_range = Z_coords.max() - Z_coords.min()
            y_min, y_max = Y_coords.min() - 0.1 * y_range if y_range > 0 else Y_coords.min() - 0.1, Y_coords.max() + 0.1 * y_range if y_range > 0 else Y_coords.max() + 0.1
            z_min, z_max = Z_coords.min() - 0.1 * z_range if z_range > 0 else Z_coords.min() - 0.1, Z_coords.max() + 0.1 * z_range if z_range > 0 else Z_coords.max() + 0.1
            
            if y_range == 0: y_max = y_min + 0.5 
            if z_range == 0: z_max = z_min + 0.5 

            grid_y, grid_z = np.meshgrid(
                np.linspace(y_min, y_max, 100),
                np.linspace(z_min, z_max, 100)
            )
            grid_percentage_deviation = griddata((Y_coords, Z_coords), deviations_perc, (grid_y, grid_z), method='linear')

            deviation_max_abs = np.nanmax(np.abs(deviations_perc[~np.isnan(deviations_perc)])) if not np.all(np.isnan(deviations_perc)) else 10 
            dev_levels = np.linspace(-deviation_max_abs, deviation_max_abs, 20)
            
            dev_contourf = plt.contourf(grid_y, grid_z, grid_percentage_deviation, levels=dev_levels, cmap='RdBu_r', extend='both')
            plt.colorbar(dev_contourf, label='相对偏差 (%)')
            plt.contour(grid_y, grid_z, grid_percentage_deviation, levels=[0], colors='black', linestyles='--', linewidths=1.5, label='0%偏差线')
        else:
            print(f"目标风速 {target_speed:.2f} m/s 的点位不足以绘制偏差等值线图。")

        scatter = plt.scatter(Y_coords, Z_coords, c=deviations_perc, cmap='RdBu_r', edgecolors='black', s=100, zorder=2)
        for idx, row in spatial_df.iterrows():
            plt.text(row['Y'], row['Z'] + 0.05, f'{row["PercentageDeviation"]:.2f}%', 
                     ha='center', va='bottom', fontsize=8, color='black')

        plt.xlabel('横向位置 (Y) / m')
        plt.ylabel('高度 (Z) / m')
        plt.title(f'目标风速 {target_speed:.2f} m/s - 空间相对偏差分布图')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        handles = [scatter]
        labels = ['测量点 (颜色表示偏差)']
        if len(Y_coords) >= 3 and len(np.unique(Y_coords)) > 1 and len(np.unique(Z_coords)) > 1:
             handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
             labels.append('0%偏差线')

        plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0.15, 1, 1]) 
        plt.savefig(os.path.join(OUTPUT_DIR, f'spatial_deviation_contour_{target_speed:.2f}m_s.png'), dpi=300, bbox_inches='tight')
        print(f"空间相对偏差分布图已保存为 '{os.path.join(OUTPUT_DIR, f'spatial_deviation_contour_{target_speed:.2f}m_s.png')}'")
        plt.close()

        # 风速剖面图 (Wind Speed Profile Plot) 
        plt.figure(figsize=(8, 6))
        sorted_points_by_Z = spatial_df.sort_values(by='Z')
        unique_Y_coords = sorted(spatial_df['Y'].unique())
        
        for y_val in unique_Y_coords:
            subset = sorted_points_by_Z[sorted_points_by_Z['Y'] == y_val]
            if not subset.empty:
                plt.plot(subset['MeanSpeed'], subset['Z'], marker='o', label=f'Y={y_val:.1f} m')
        
        plt.xlabel('平均风速 (m/s)')
        plt.ylabel('高度 (Z) / m')
        plt.title(f'目标风速 {target_speed:.2f} m/s - 风速剖面图')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(title='横向位置')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'wind_speed_profile_{target_speed:.2f}m_s.png'), dpi=300, bbox_inches='tight')
        print(f"风速剖面图已保存为 '{os.path.join(OUTPUT_DIR, f'wind_speed_profile_{target_speed:.2f}m_s.png')}'")
        plt.close()

        # 风速横向分布图 (Wind Speed Lateral Distribution Plot) 
        plt.figure(figsize=(10, 6))
        sorted_points_by_Y = spatial_df.sort_values(by='Y')
        unique_Z_coords = sorted(spatial_df['Z'].unique())

        for z_val in unique_Z_coords:
            subset = sorted_points_by_Y[sorted_points_by_Y['Z'] == z_val]
            if not subset.empty:
                plt.plot(subset['Y'], subset['MeanSpeed'], marker='o', label=f'Z={z_val:.1f} m')

        plt.xlabel('横向位置 (Y) / m')
        plt.ylabel('平均风速 (m/s)')
        plt.title(f'目标风速 {target_speed:.2f} m/s - 风速横向分布图')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(title='高度')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'wind_speed_lateral_distribution_{target_speed:.2f}m_s.png'), dpi=300, bbox_inches='tight')
        print(f"风速横向分布图已保存为 '{os.path.join(OUTPUT_DIR, f'wind_speed_lateral_distribution_{target_speed:.2f}m_s.png')}'")
        plt.close()


# --- 4. 风场稳定性分析 (时间稳定性) ---
def analyze_temporal_stability(sim_data_list):
    """
    分析每个测量点的时间稳定性。
    主要通过绘制时序图和箱线图。
    （CoVT柱状图已移至compare_across_simulations进行集中展示）
    """
    print(f"\n--- 正在进行时间稳定性分析 ---")

    # 存储所有点的时间稳定性数据，供后续统一绘制CoVT散点图
    all_temporal_stability_data = []

    for item in sim_data_list:
        df = item['df'] 
        y_coord = item['y_coordinate']
        target_speed = item['target_speed']
        point_coords = item['point_coords']
        filename = item['filename']

        data_columns = [col for col in df.columns if col != 'Time']

        # 绘制每个点的风速时序图 (调整图例位置)
        plt.figure(figsize=(12, 7))
        if df.empty or not data_columns: 
            plt.text(0.5, 0.5, "无风速数据可绘制", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        else:
            for col in data_columns:
                if col in point_coords:
                    plt.plot(df['Time'], df[col], label=f'Y={y_coord:.1f} Z={point_coords[col]["Z"]:.1f} ({col})', alpha=0.7)
            
            plt.xlabel('时间')
            plt.ylabel('风速 (m/s)')
            plt.title(f'文件: {filename} - 各点风速时序图 (目标风速: {target_speed:.2f} m/s)')
            plt.grid(True, linestyle=':', alpha=0.7)
            
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       fancybox=True, shadow=True, ncol=len(data_columns) // 2 + 1) 
            plt.tight_layout(rect=[0, 0.1, 1, 1]) 

        save_name = f'temporal_wind_speed_series_{filename.replace(".xlsx", "").replace(".xls", "")}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300, bbox_inches='tight')
        print(f"风速时序图已保存为 '{os.path.join(OUTPUT_DIR, save_name)}'")
        plt.close()

        # 箱线图来展示时间稳定性
        # 备注：箱线图集中处理会导致X轴标签过于密集且难以辨认，故维持每个文件一张。
        plt.figure(figsize=(12, 7))
        plot_data = []
        plot_labels = []

        sensors_in_this_file = {k: v for k, v in SENSOR_Z_MAPPING.items() if k in data_columns}
        sorted_sensor_names_by_Z = [name for name, _ in sorted(sensors_in_this_file.items(), key=lambda item: item[1])]

        if not sorted_sensor_names_by_Z:
            plt.text(0.5, 0.5, "无风速数据可绘制箱线图", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        else:
            for col_name in sorted_sensor_names_by_Z:
                plot_data.append(df[col_name].dropna()) 
                plot_labels.append(f'Y={y_coord:.1f} Z={point_coords[col_name]["Z"]:.1f}')
            
            if plot_data: 
                plt.boxplot(plot_data, labels=plot_labels, patch_artist=True) 

                plt.xlabel('测量点 (Y, Z 坐标)')
                plt.ylabel('风速 (m/s)')
                plt.title(f'文件: {filename} - 各点风速分布箱线图 (目标风速: {target_speed:.2f} m/s)')
                plt.grid(axis='y', linestyle=':', alpha=0.7)
                plt.xticks(rotation=45, ha='right') 
                plt.tight_layout()

        box_plot_save_name = f'temporal_wind_speed_boxplot_{filename.replace(".xlsx", "").replace(".xls", "")}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, box_plot_save_name), dpi=300, bbox_inches='tight')
        print(f"风速分布箱线图已保存为 '{os.path.join(OUTPUT_DIR, box_plot_save_name)}'")
        plt.close()

        # 统计每个点的时间波动性，并添加到总列表中
        temporal_stability_results_for_file = []
        for col_name in data_columns:
            if col_name in point_coords:
                coords = point_coords[col_name]
                wind_series = df[col_name]
                mean_speed = wind_series.mean()
                std_dev = wind_series.std()
                cv = (std_dev / mean_speed) * 100 if mean_speed != 0 else np.nan
                
                point_data = {
                    'File': filename,
                    'TargetSpeed': target_speed,
                    'Y_Coord': y_coord,
                    'Z_Coord': coords['Z'],
                    'PointName': col_name,
                    'MeanSpeed_TimeSeries': mean_speed,
                    'StdDev_TimeSeries': std_dev, 
                    'CoV_TimeSeries': cv 
                }
                temporal_stability_results_for_file.append(point_data)
                all_temporal_stability_data.append(point_data) # 添加到总列表
            else:
                 print(f"警告: 文件 '{filename}' 中点位 '{col_name}' 未找到对应的Y,Z坐标，跳过时间稳定性分析。")

        temporal_df = pd.DataFrame(temporal_stability_results_for_file)
        if not temporal_df.empty:
            temporal_stats_output = [f"\n文件: {filename} - 时间稳定性指标 (每个点):"]
            temporal_df_sorted = temporal_df.sort_values(by=['Y_Coord', 'Z_Coord'])
            temporal_stats_output.append(temporal_df_sorted[['Y_Coord', 'Z_Coord', 'PointName', 'MeanSpeed_TimeSeries', 'StdDev_TimeSeries', 'CoV_TimeSeries']].to_string())
            
            save_text_to_file("\n".join(temporal_stats_output), 
                               f'temporal_stability_stats_{filename.replace(".xlsx", "").replace(".xls", "")}.txt', 
                               OUTPUT_DIR)
            print("\n".join(temporal_stats_output)) 
        else:
            print(f"文件: {filename} - 没有可用于时间稳定性分析的点位数据。")
    
    # 将所有收集到的时间稳定性数据返回，供 compare_across_simulations 使用
    return all_temporal_stability_data


# --- 5. 跨工况比较分析 (均匀性和稳定性随目标风速的变化) ---
def compare_across_simulations(sim_data_list, all_temporal_stability_data):
    """
    比较不同模拟风速下风场均匀性和稳定性的趋势，并绘制所有点的CoVT散点图。
    """
    print("\n--- 正在进行跨工况比较分析 ---")

    # 按目标风速分组聚合数据，用于计算总体指标
    grouped_by_target_speed = {}
    for item in sim_data_list:
        target_speed = item['target_speed']
        if target_speed not in grouped_by_target_speed:
            grouped_by_target_speed[target_speed] = {
                'all_mean_speeds_spatial': [], 
                'all_temporal_covs_for_avg': [] # 仅用于计算平均CoV
            }
        
        df = item['df'] 
        point_coords = item['point_coords']

        mean_speeds_for_file = df.drop(columns='Time').mean()
        for col_name, mean_speed in mean_speeds_for_file.items():
            if col_name in point_coords:
                grouped_by_target_speed[target_speed]['all_mean_speeds_spatial'].append(mean_speed)
        
        # 收集时间稳定性所需的所有时间变异系数 (用于计算平均值)
        for col_name in df.drop(columns='Time').columns:
            if col_name in point_coords:
                wind_series = df[col_name]
                mean_s = wind_series.mean()
                std_s = wind_series.std()
                if mean_s != 0:
                    grouped_by_target_speed[target_speed]['all_temporal_covs_for_avg'].append((std_s / mean_s) * 100)

    summary_data = []
    for target_speed, data in grouped_by_target_speed.items():
        all_mean_speeds_spatial = np.array(data['all_mean_speeds_spatial'])
        overall_mean_speed_spatial = np.mean(all_mean_speeds_spatial) if all_mean_speeds_spatial.size > 0 else np.nan
        overall_spatial_std_dev = np.std(all_mean_speeds_spatial) if all_mean_speeds_spatial.size > 0 else np.nan
        overall_spatial_cv = (overall_spatial_std_dev / overall_mean_speed_spatial) * 100 if overall_mean_speed_spatial != 0 else np.nan

        all_temporal_covs_for_avg = np.array(data['all_temporal_covs_for_avg'])
        overall_avg_temporal_cov = np.mean(all_temporal_covs_for_avg) if all_temporal_covs_for_avg.size > 0 else np.nan

        summary_data.append({
            'TargetSpeed': target_speed,
            'OverallMeanSpeed': overall_mean_speed_spatial,
            'OverallSpatialStdDev': overall_spatial_std_dev, 
            'OverallSpatialCoV': overall_spatial_cv, 
            'OverallAvgTemporalCoV': overall_avg_temporal_cov 
        })

    summary_df = pd.DataFrame(summary_data).sort_values(by='TargetSpeed')
    
    overall_summary_output = ["\n跨工况均匀性与稳定性汇总:", summary_df.to_string()]
    
    save_text_to_file("\n".join(overall_summary_output), 
                       'overall_uniformity_stability_summary.txt', 
                       OUTPUT_DIR)
    print("\n".join(overall_summary_output)) 

    # 绘制趋势图
    if len(summary_df) > 1: 
        plt.figure(figsize=(10, 6))
        plt.plot(summary_df['TargetSpeed'], summary_df['OverallSpatialStdDev'], marker='o', linestyle='-', label='整体空间标准差 (m/s)')
        plt.plot(summary_df['TargetSpeed'], summary_df['OverallSpatialCoV'], marker='s', linestyle='--', label='整体空间变异系数 (%)')
        plt.plot(summary_df['TargetSpeed'], summary_df['OverallAvgTemporalCoV'], marker='x', linestyle=':', label='平均时间变异系数 (%)')
        plt.xlabel('目标风速 (m/s)')
        plt.ylabel('指标值')
        plt.title('风场均匀性与稳定性随目标风速的变化')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'overall_uniformity_stability_trend.png'), dpi=300, bbox_inches='tight')
        print(f"跨工况趋势图已保存为 '{os.path.join(OUTPUT_DIR, 'overall_uniformity_stability_trend.png')}'")
        plt.close()
    else:
        print("至少需要两个工况才能进行跨工况比较分析。")

    # *** 所有点位的时间变异系数散点图 (包含数值标注和核心区域) ***
    if all_temporal_stability_data:
        all_cov_df = pd.DataFrame(all_temporal_stability_data)
        # 移除CoV_TimeSeries为NaN的行，这些通常是由于mean_speed为0导致的
        all_cov_df.dropna(subset=['CoV_TimeSeries'], inplace=True) 

        if not all_cov_df.empty:
            plt.figure(figsize=(12, 8))
            
            # 由于去掉了图例，现在直接绘制所有点
            # 颜色映射仍然根据 CoV_TimeSeries
            scatter = plt.scatter(all_cov_df['Y_Coord'], all_cov_df['Z_Coord'], 
                                  c=all_cov_df['CoV_TimeSeries'], cmap='viridis_r', 
                                  marker='o', s=150, edgecolors='black', 
                                  alpha=0.8) # 移除了label参数，因为不再需要图例

            if scatter is not None:
                cbar = plt.colorbar(scatter, label='时间变异系数 (CoV) / %')
            
            # 添加文本标注，调整字体大小和颜色，并添加白色背景框
            for idx, row in all_cov_df.iterrows():
                if pd.notna(row['CoV_TimeSeries']):
                    plt.text(row['Y_Coord'], row['Z_Coord'] + 0.08, # Y坐标稍作偏移，避免与点重叠
                             f'{row["CoV_TimeSeries"]:.2f}', 
                             ha='center', va='bottom', 
                             fontsize=12,               # 增大字体大小
                             color='black',            # 字体颜色改为黑色
                             weight='bold',            # 加粗
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)) # 白色半透明背景框
            
            # *** 绘制核心区域矩形 ***
            rect = patches.Rectangle((CORE_REGION_Y[0], CORE_REGION_Z[0]), # (x,y) bottom-left
                                     CORE_REGION_Y[1] - CORE_REGION_Y[0],  # width
                                     CORE_REGION_Z[1] - CORE_REGION_Z[0],  # height
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--', zorder=3,
                                     label='核心区域') # 核心区域的label可以保留，用于单独的图例或直接理解
            plt.gca().add_patch(rect)
            
            plt.xlabel('横向位置 (Y) / m')
            plt.ylabel('高度 (Z) / m')
            plt.title('各测量点风速时间变异系数 (CoV) 分布图 (含数值与核心区域)')
            plt.grid(True, linestyle=':', alpha=0.7)
            
            # *** 移除图例 ***
            # plt.legend(handles, labels, title='工况/区域') # 这行被注释掉或移除

            plt.tight_layout()
            
            cov_scatter_plot_save_name = 'all_points_temporal_cov_scatter_with_labels_core_region.png'
            plt.savefig(os.path.join(OUTPUT_DIR, cov_scatter_plot_save_name), dpi=300, bbox_inches='tight')
            print(f"所有点位时间变异系数散点图 (含数值与核心区域) 已保存为 '{os.path.join(OUTPUT_DIR, cov_scatter_plot_save_name)}'")
            plt.close()
        else:
            print("没有足够的有效时间变异系数数据来绘制散点图。")
    else:
        print("未收集到任何时间稳定性数据。")

    # *** 计算核心区域内各点时间变异系数的平均值 ***
    if all_temporal_stability_data:
        all_cov_df = pd.DataFrame(all_temporal_stability_data)
        all_cov_df.dropna(subset=['CoV_TimeSeries'], inplace=True) 

        core_region_cov_df = all_cov_df[
            (all_cov_df['Y_Coord'] >= CORE_REGION_Y[0]) &
            (all_cov_df['Y_Coord'] <= CORE_REGION_Y[1]) &
            (all_cov_df['Z_Coord'] >= CORE_REGION_Z[0]) &
            (all_cov_df['Z_Coord'] <= CORE_REGION_Z[1])
        ].copy()

        if not core_region_cov_df.empty:
            avg_core_region_cov_by_speed = core_region_cov_df.groupby('TargetSpeed')['CoV_TimeSeries'].mean().reset_index()
            avg_core_region_cov_by_speed.rename(columns={'CoV_TimeSeries': 'Avg_Core_Region_CoV'}, inplace=True)

            print("\n--- 核心区域时间变异系数平均值 ---")
            core_region_output = [
                "核心区域定义:",
                f"  横向 (Y) 范围: {CORE_REGION_Y[0]} m 到 {CORE_REGION_Y[1]} m",
                f"  高度 (Z) 范围: {CORE_REGION_Z[0]} m 到 {CORE_REGION_Z[1]} m",
                "\n核心区域时间变异系数平均值 (按目标风速):", 
                avg_core_region_cov_by_speed.to_string(index=False) # 不显示DataFrame索引
            ]
            save_text_to_file("\n".join(core_region_output),
                               'core_region_avg_temporal_cov.txt',
                               OUTPUT_DIR)
            print("\n".join(core_region_output))
        else:
            print("\n--- 核心区域时间变异系数平均值 ---")
            print(f"警告: 在核心区域 (Y: {CORE_REGION_Y[0]}~{CORE_REGION_Y[1]}m, Z: {CORE_REGION_Z[0]}~{CORE_REGION_Z[1]}m) 内没有找到测量点。请检查核心区域定义或数据。")


# --- 主函数：执行分析流程 ---
def main_analysis():
    """
    主分析流程控制函数，依次调用数据加载、空间均匀性、时间稳定性
    和跨工况比较分析函数。
    """
    print("当前脚本认为的工作目录是:", os.getcwd())
    setup_chinese_font() 
    print("--- 风场均匀性和稳定性分析开始 ---")

    all_sim_data_list = load_and_preprocess_data(EXCEL_FILES_DIRECTORY, DATA_POINTS_TO_ANALYZE)
    
    if not all_sim_data_list:
        print("未成功加载任何模拟工况数据，分析中止。")
        return

    analyze_spatial_uniformity(all_sim_data_list)

    all_temporal_stability_data = analyze_temporal_stability(all_sim_data_list)

    compare_across_simulations(all_sim_data_list, all_temporal_stability_data)

    print("\n--- 风场均匀性和稳定性分析完成 ---")

# --- 运行主分析函数 ---
if __name__ == "__main__":
    main_analysis()