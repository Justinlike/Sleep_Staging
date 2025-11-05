import os
import glob
import re
import pyedflib

def find_file_pairs_flexible(data_dir):
    """
    在指定目录中灵活地查找成对的 PSG 和 Hypnogram EDF 文件，
    忽略文件名中 E0/E1 的差异。

    参数:
        data_dir (str): 包含 EDF 文件的目录路径。

    返回:
        list: 一个包含元组的列表，每个元组是 (psg_filepath, hypnogram_filepath)。
    """
    # 1. 使用 glob 查找所有 PSG 文件
    psg_files = glob.glob(os.path.join(data_dir, "*PSG.edf"))

    if not psg_files:
        print(f"警告: 在目录 '{data_dir}' 中没有找到任何 *PSG.edf 文件。")
        return []

    # 2. 创建一个字典，用于存储所有 Hypnogram 文件的基础名和完整路径
    #    基础名示例: 'SC4011'
    hypnogram_map = {}
    hypnogram_files = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))
    for hyp_path in hypnogram_files:
        # 使用正则表达式提取基础名 (例如, 从 'SC4011E1-Hypnogram.edf' 提取 'SC4011')
        match = re.match(r"(SC\d+)", os.path.basename(hyp_path))
        if match:
            base_name = match.group(1)
            hypnogram_map[base_name] = hyp_path

    matched_pairs = []
    # 3. 遍历每个 PSG 文件，用其基础名在 Hypnogram 字典中查找匹配项
    for psg_path in psg_files:
        # 同样地，从 PSG 文件名中提取基础名
        match = re.match(r"(SC\d+)", os.path.basename(psg_path))
        if match:
            base_name = match.group(1)
            # 检查这个基础名是否存在于 hypnogram_map 中
            if base_name in hypnogram_map:
                hypnogram_path = hypnogram_map[base_name]
                matched_pairs.append((psg_path, hypnogram_path))
            else:
                print(f"警告: 找到了 {os.path.basename(psg_path)} 但缺少对应的 Hypnogram 文件。")

    return matched_pairs

# --- 使用示例 ---

# 您的数据目录
file_path = r"E:\datasets\sleep-edf-database-expanded-1.0.0\sleep-cassette"

# 调用新的函数查找所有文件对
file_pairs = find_file_pairs_flexible(file_path)

# 打印找到的文件对数量
print(f"\n成功匹配到 {len(file_pairs)} 对文件。\n")

# 遍历并打印所有匹配成功的文件对，以供验证
for psg, hyp in file_pairs:
    print(f"信号文件: {os.path.basename(psg)}")
    print(f"标签文件: {os.path.basename(hyp)}")
    print("-" * 20)

# 如上读取到了数据标签对，可以继续后续处理
def process_and_clean_data(file_pairs, trim_mins=30):
    """
    处理并清洗所有文件对。

    参数:
        file_pairs (list): (psg_path, hyp_path) 的元组列表。
        trim_mins (int): 在睡眠开始前和结束后要包含的分钟数。
    """
    all_cleaned_data = []
    trim_secs = trim_mins * 60

    for psg_path, hyp_path in file_pairs:
        print(f"--- 正在处理: {os.path.basename(psg_path)} ---")
        try:
            # --- 1. 读取 Hypnogram 文件并确定裁剪窗口 ---
            with pyedflib.EdfReader(hyp_path) as f_hyp:
                onsets, durations, annotations = f_hyp.readAnnotations()

            # 找到第一个和最后一个非WAKE睡眠阶段的索引
            sleep_indices = [i for i, ann in enumerate(annotations) if 'Sleep stage W' not in ann]
            if not sleep_indices:
                print(f"警告: {os.path.basename(hyp_path)} 中没有找到睡眠阶段，已跳过。")
                continue

            first_sleep_onset = onsets[sleep_indices[0]]
            last_sleep_onset = onsets[sleep_indices[-1]]
            last_sleep_duration = durations[sleep_indices[-1]]

            # 计算裁剪的开始和结束时间（秒）
            start_trim_sec = max(0, first_sleep_onset - trim_secs)
            end_trim_sec = last_sleep_onset + last_sleep_duration + trim_secs

            # --- 2. 读取 PSG 文件并提取指定通道和时间段的数据 ---
            with pyedflib.EdfReader(psg_path) as f_psg:
                labels = f_psg.getSignalLabels()
                fs = f_psg.getSampleFrequency(0) # 获取采样率

                # 找到 'EEG Fpz-Cz' 通道
                try:
                    channel_idx = labels.index('EEG Fpz-Cz')
                except ValueError:
                    # 备用方案，以防通道名略有不同
                    channel_idx = [i for i, s in enumerate(labels) if 'Fpz-Cz' in s][0]

                # 检查请求的结束时间是否超过文件总时长
                total_duration_sec = f_psg.getFileDuration()
                if end_trim_sec > total_duration_sec:
                    end_trim_sec = total_duration_sec

                # 计算要读取的样本范围
                start_sample = int(start_trim_sec * fs)
                num_samples_to_read = int((end_trim_sec - start_trim_sec) * fs)

                # 读取信号片段
                fpz_cz_signal = f_psg.readSignal(channel_idx, start=start_sample, n=num_samples_to_read)

            # --- 3. 筛选与信号片段对应的睡眠标签 ---
            trimmed_annotations = []
            for i, onset in enumerate(onsets):
                if start_trim_sec <= onset < end_trim_sec:
                    trimmed_annotations.append(annotations[i])

            print(f"原始信号时长: {total_duration_sec / 60:.1f} 分钟")
            print(f"清洗后信号时长: {len(fpz_cz_signal) / fs / 60:.1f} 分钟")
            print(f"清洗后标签数量: {len(trimmed_annotations)}")

            # 将清洗后的数据存储起来（示例）
            all_cleaned_data.append({
                'id': os.path.basename(psg_path).split('-')[0],
                'signal': fpz_cz_signal,
                'annotations': trimmed_annotations,
                'fs': fs
            })

        except Exception as e:
            print(f"处理 {os.path.basename(psg_path)} 时出错: {e}")

    return all_cleaned_data


# --- 使用示例 ---
# 假设 file_pairs 已经由之前的 find_file_pairs_flexible 函数生成
if 'file_pairs' in locals() and file_pairs:
    # 调用清洗函数
    cleaned_data_list = process_and_clean_data(file_pairs)

    print(f"\n\n总共成功处理并清洗了 {len(cleaned_data_list)} 个文件。")

    # 您可以检查第一个处理完的数据
    if cleaned_data_list:
        first_subject = cleaned_data_list[0]
        print(f"\n示例数据 (ID: {first_subject['id']}):")
        print(f"  信号点数: {len(first_subject['signal'])}")
        print(f"  采样率: {first_subject['fs']} Hz")
        print(f"  标签数量: {len(first_subject['annotations'])}")
        print(f"  前5个标签: {first_subject['annotations'][:5]}")
else:
    print("请先运行文件匹配代码以生成 'file_pairs' 列表。")