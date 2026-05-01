import json
import argparse
import os
import pandas as pd

def calc_timestamp_by_gps_time(week, seconds):
    """
    根据GPS时间计算时间戳，实现C++代码中的算法，结果保留3位小数
    
    参数:
        week: 星期数
        seconds: 秒数
        
    返回:
        计算得到的时间戳（保留3位小数）
    """
    try:
        # 将输入转换为适当的数值类型
        week = int(week)
        seconds = float(seconds)
        
        time_diff = 315964800
        seconds_per_week = 604800
        time_offset = 18
        
        # 计算并保留3位小数
        timestamp = time_diff + week * seconds_per_week + seconds - time_offset
        return round(timestamp, 3)
    except (ValueError, TypeError) as e:
        print(f"计算时间戳时出错: {e}")
        return None

def extract_time_segments(json_file):
    """
    从JSON文件中提取时间字段并计算时间戳，生成时间区间
    
    参数:
        json_file: JSON文件路径
        
    返回:
        包含link_code、start_timestamp和end_timestamp的列表
    """
    try:
        # 打开并读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        
        # 遍历每个条目
        for item in data:
            # 检查是否有survey_time字段且为列表
            if 'survey_time' in item and isinstance(item['survey_time'], list) and len(item['survey_time']) > 0:
                # 获取survey_time中的第一个元素
                survey_time = item['survey_time'][0]
                
                # 提取所需字段
                link_code = item.get('link_code', '未知')
                start_week = survey_time.get('start_week', '未找到')
                start_seconds = survey_time.get('start_seconds', '未找到')
                end_week = survey_time.get('end_week', '未找到')
                end_seconds = survey_time.get('end_seconds', '未找到')
                
                # 计算时间戳
                start_timestamp = calc_timestamp_by_gps_time(start_week, start_seconds) if start_week != '未找到' and start_seconds != '未找到' else None
                end_timestamp = calc_timestamp_by_gps_time(end_week, end_seconds) if end_week != '未找到' and end_seconds != '未找到' else None
                
                if start_timestamp is not None and end_timestamp is not None:
                    segments.append({
                        'link_code': link_code,
                        'start_time': start_timestamp,
                        'end_time': end_timestamp
                    })
                else:
                    print(f"警告: {link_code} 缺少有效的时间信息，跳过")
        
        return segments
    
    except FileNotFoundError:
        print(f"错误: 文件 '{json_file}' 未找到")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_file}' 不是有效的JSON格式")
        return []
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return []

def main(task_sel_json_file, parse_temp_file, output_dir):
    # 1. 从JSON文件提取时间区间
    print(f"从 {task_sel_json_file} 提取时间区间...")
    segments = extract_time_segments(task_sel_json_file)
    
    if not segments:
        print("未提取到任何有效的时间区间，程序退出")
        return
    
    # 打印时间区间信息
    print(f"\n发现 {len(segments)} 个时间区间，详细信息如下:")
    print(f"{'序号':<5} | {'link_code':<10} | {'start_time':<20} | {'end_time':<20}")
    print("-" * 60)
    for i, segment in enumerate(segments, 1):
        print(f"{i:<5} | {segment['link_code']:<10} | {segment['start_time']:<20.3f} | {segment['end_time']:<20.3f}")
    
    # 2. 读取原始数据（kelevt.txt），包含高度、分段、时间三个字段
    print(f"\n读取原始数据文件: {parse_temp_file}")
    try:
        # 读取原始数据，不做任何处理，列名为高度(height)、分段(segment)、时间(time)
        df = pd.read_csv(parse_temp_file, header=None, names=['time', 'height', 'segment'])
        print(f"成功读取 {len(df)} 条记录")
    except FileNotFoundError:
        print(f"错误: 数据文件 '{parse_temp_file}' 未找到")
        return
    except Exception as e:
        print(f"读取数据文件时出错: {str(e)}")
        return
    
    # 3. 按时间区间分割并保存数据
    print("\n开始生成对应区间文件...")
    for i, segment in enumerate(segments, 1):
        link_code = segment['link_code']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # 筛选时间在 [start_time, end_time] 之间的数据
        # 注意：这里使用的是数据中的'time'列进行筛选
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        filtered_data = df[mask]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{link_code}.csv"
        file_path = os.path.join(output_dir, filename)
        if not filtered_data.empty:
            # 使用link_code作为文件名
            filtered_data.to_csv(
                file_path,
                index=False,
                header=False,
                float_format='%.4f'
            )
            print(f"生成文件: {filename} (包含 {len(filtered_data)} 条记录)")
        else:
            print(f"警告: 在区间 {link_code} 中未找到任何数据")

    print("\n所有区间文件生成完成")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task_sel", required=True, type=str, default="")
    args.add_argument("--parse_temp", required=True, type=str, default="")
    args.add_argument("--out", required=True, type=str, default="")
    params = args.parse_args()

    task_sel_path = params.task_sel
    parse_temp_path = params.parse_temp
    output = params.out

    if task_sel_path == "" or parse_temp_path == "" or output == "":
        exit(-1)
    if not os.path.exists(task_sel_path) or not os.path.exists(parse_temp_path):
        exit(-1)
    main(task_sel_json_file=task_sel_path, parse_temp_file=parse_temp_path, output_dir=output)
