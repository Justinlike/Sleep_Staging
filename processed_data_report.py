import pandas as pd
from collections import Counter
from logger import get_logger
import os
import glob
import ntpath
import numpy as np
import argparse

def generate_stats_report(output_dir, report_filename="dataset_statistics.xlsx"):
    """
    读取处理后的 .npz 文件，生成一个包含数据集统计信息的 Excel 报告。

    :param output_dir: 包含 .npz 文件的目录。
    :param report_filename: 输出的 Excel 文件名。
    """
    logger = get_logger("stats_logger", level="info")
    logger.info(f"开始生成统计报告，扫描目录: {output_dir}")

    npz_files = glob.glob(os.path.join(output_dir, "*.npz"))
    if not npz_files:
        logger.warning("在输出目录中未找到 .npz 文件，无法生成报告。")
        return

    stats_list = []
    # 睡眠阶段标签到名称的映射 (W, N1, N2, N3, R)
    stage_mapping = {0: "W", 1: "N1", 2: "N2", 3: "N3/N4", 4: "R"}

    for file_path in sorted(npz_files):
        try:
            # 修正点 1: 添加 allow_pickle=True
            with np.load(file_path, allow_pickle=True) as data:
                filename = ntpath.basename(file_path)
                y = data['y']

                # 统计每个睡眠阶段的数量
                stage_counts = Counter(y)

                file_stats = {
                    "文件名": filename,
                    "总时期数 (清洗后)": data['n_epochs'].item(),
                    "采样率 (Hz)": data['fs'].item(),
                    "时期时长 (秒)": data['epoch_duration'].item(),
                    "记录开始时间": str(data['start_datetime'].item()),
                }

                # 添加每个阶段的计数
                for label, name in stage_mapping.items():
                    file_stats[f"时期数_{name}"] = stage_counts.get(label, 0)

                stats_list.append(file_stats)
                logger.info(f"已处理文件: {filename}")

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")

    if not stats_list:
        logger.warning("未能从任何文件中提取统计数据。")
        return

    # 创建 DataFrame
    df = pd.DataFrame(stats_list)

    # 调整列顺序
    column_order = [
        "文件名", "总时期数 (清洗后)",
        "时期数_W", "时期数_N1", "时期数_N2", "时期数_N3/N4", "时期数_R",
        "采样率 (Hz)", "时期时长 (秒)", "记录开始时间"
    ]
    # 确保所有预期的列都存在
    df = df.reindex(columns=column_order)

    # 修正点 2: 将报告保存在指定的输出目录中
    report_path = os.path.join("", report_filename)
    try:
        df.to_excel(report_path, index=False)
        logger.info(f"统计报告已成功保存到: {report_path}")
    except Exception as e:
        logger.error(f"保存 Excel 文件失败: {e}")

        logger.info("\n=======================================\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str,
                        default="E:/datasets/sleep-edf-database-expanded-1.0.0/sleep-cassette/eeg_fpz_cz",
                        help="Directory where to save outputs.")

    args = parser.parse_args()
    # 在所有文件处理完毕后，生成统计报告
    generate_stats_report(args.output_dir)
