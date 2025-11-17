#!/usr/bin/env python3
"""
水质数据增强插值处理脚本
基于原有的时间序列插值代码，针对水质数据特点进行优化
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from enhanced_ts_imputation import (
    load_data, clean_data, advanced_imputation, 
    format_decimal_places, save_data, generate_data_quality_report
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_quality_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def process_water_quality_dataset(input_file, output_file, dataset_name):
    """
    处理单个水质数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径  
        dataset_name: 数据集名称（用于日志）
    """
    try:
        logging.info(f"开始处理 {dataset_name} 数据集...")
        logging.info(f"输入文件: {input_file}")
        logging.info(f"输出文件: {output_file}")
        
        # 1. 加载数据
        df = load_data(input_file)
        
        # 2. 清理数据（检测异常值、处理0值和负值）
        df_cleaned = clean_data(df, time_col='DATE', outlier_method='iqr', outlier_factor=1.5)
        
        # 3. 高级插值（考虑水质数据特点）
        df_imputed = advanced_imputation(df_cleaned, time_col='DATE')
        
        # 4. 格式化小数位数
        df_formatted = format_decimal_places(df_imputed, time_col='DATE')
        
        # 5. 保存数据
        save_data(df_formatted, output_file, format='csv')
        
        # 6. 生成数据质量报告
        report_file = Path(output_file).with_suffix('.quality_report.txt')
        generate_data_quality_report(df_formatted, time_col='DATE')
        
        logging.info(f"{dataset_name} 数据集处理完成！")
        logging.info(f"处理后的数据形状: {df_formatted.shape}")
        
        return df_formatted
        
    except Exception as e:
        logging.error(f"处理 {dataset_name} 数据集时出错: {e}")
        raise

def main():
    """主函数 - 处理所有水质数据集"""
    
    # 数据集配置
    datasets = {
        'WLSQ': {
            'input': r'E:\data-ji\WH/WLSQ.csv',
            'output': r'E:\data-ji\WH/WLSQ_enhanced.csv'
        },
        'XFZ': {
            'input': r'E:\data-ji\WH/XFZ.csv',
            'output': r'E:\data-ji\WH/XFZ_enhanced.csv'
        },
        'XP': {
            'input': r'E:\data-ji\WH/XP.csv',
            'output': r'E:\data-ji\WH/XP_enhanced.csv'
        },
        'XYTQ': {
            'input': r'E:\data-ji\WH/XYTQ.csv',
            'output': r'E:\data-ji\WH/XYTQ_enhanced.csv'
        }
    }
    
    processed_datasets = {}
    
    try:
        for dataset_name, config in datasets.items():
            input_file = config['input']
            output_file = config['output']
            
            # 检查输入文件是否存在
            if not Path(input_file).exists():
                logging.warning(f"输入文件不存在，跳过 {dataset_name}: {input_file}")
                continue
            
            # 处理数据集
            df_processed = process_water_quality_dataset(
                input_file, output_file, dataset_name
            )
            processed_datasets[dataset_name] = df_processed
            
        # 生成总体处理报告
        logging.info("=" * 60)
        logging.info("所有数据集处理完成！")
        logging.info("=" * 60)
        
        for dataset_name, df in processed_datasets.items():
            logging.info(f"{dataset_name}: {df.shape[0]} 行, {df.shape[1]} 列")
            
        logging.info("\n处理后的文件:")
        for dataset_name, config in datasets.items():
            if dataset_name in processed_datasets:
                logging.info(f"  - {config['output']}")
                logging.info(f"  - {config['output']}.quality_report.txt")
        
    except Exception as e:
        logging.error(f"批量处理时出错: {e}")
        raise

if __name__ == "__main__":
    main()