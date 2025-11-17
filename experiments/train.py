# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4)  # aviod high cpu avg usage

# 定义要批量运行的数据集列表
DATASETS = ['DZM','WLSQ']


def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STID/',
                        help='model config directory (e.g., baselines/PatchTST/)')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()


def main():
    args = parse_args()

    # 确保cfg路径以/结尾
    model_dir = args.cfg
    if not model_dir.endswith('/'):
        model_dir += '/'

    total_datasets = len(DATASETS)

    print(f"开始批量训练 {total_datasets} 个数据集...")
    print("-" * 50)

    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{total_datasets}] 正在训练数据集: {dataset}")
        print(f"进度: {i}/{total_datasets} ({i / total_datasets * 100:.1f}%)")

        # 构造完整的配置文件路径
        config_path = f"{model_dir}{dataset}.py"

        # 调用basicts进行训练
        basicts.launch_training(config_path, args.gpus, node_rank=0)

        print(f"✓ 数据集 {dataset} 训练完成")

    print("\n" + "=" * 50)
    print(f"🎉 所有 {total_datasets} 个数据集训练完成！")


if __name__ == '__main__':
    main()
