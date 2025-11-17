#!/usr/bin/env python3
"""
批量处理水质数据脚本
提供三种处理策略：保守、激进、分段
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from smart_water_quality_processor import SmartWaterQualityProcessor
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_water_quality_processing.log'),
        logging.StreamHandler()
    ]
)

def batch_process_datasets():
    """批量处理所有数据集"""
    
    # 数据集配置
    datasets = {
        'WLSQ': r'E:\YL\submit-ss\BasicTS\WH\WLSQ.csv',
        'XFZ': r'E:\YL\submit-ss\BasicTS\WH\XFZ.csv',
        'XP': r'E:\YL\submit-ss\BasicTS\WH\XP.csv',
        'XYTQ': r'E:\YL\submit-ss\BasicTS\WH\XYTQ.csv',
        'WJB': r'E:\YL\submit-ss\BasicTS\WH\WJB.csv',
        'TGDQ': r'E:\YL\submit-ss\BasicTS\WH\TGDQ.csv',
        'SWD': r'E:\YL\submit-ss\BasicTS\WH\SWD.csv'
    }
    
    # 输出目录
    output_base = Path(r'E:\YL\submit-ss\BasicTS\WH_processed')
    output_base.mkdir(exist_ok=True)
    
    # 三种处理策略
    strategies = ['conservative', 'aggressive', 'segmented']
    
    print("开始批量处理水质数据...")
    print(f"输出目录: {output_base}")
    print(f"处理策略: {', '.join(strategies)}")
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"使用策略: {strategy}")
        print(f"{'='*80}")
        
        # 创建处理器
        processor = SmartWaterQualityProcessor(strategy=strategy)
        
        strategy_results = {}
        
        for dataset_name, input_file in datasets.items():
            if not Path(input_file).exists():
                print(f"⚠ 文件不存在: {input_file}")
                continue
            
            # 输出文件路径
            output_file = output_base / f"{dataset_name}_{strategy}.csv"
            
            print(f"\n处理 {dataset_name}...")
            try:
                # 处理数据
                result_df = processor.process_dataset(input_file, str(output_file))
                
                if result_df is not None:
                    strategy_results[dataset_name] = {
                        'status': 'success',
                        'shape': result_df.shape,
                        'output_file': str(output_file)
                    }
                    print(f"✓ {dataset_name} 处理成功")
                else:
                    strategy_results[dataset_name] = {
                        'status': 'failed',
                        'error': '处理失败'
                    }
                    print(f"✗ {dataset_name} 处理失败")
                    
            except Exception as e:
                strategy_results[dataset_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"✗ {dataset_name} 处理出错: {e}")
        
        results[strategy] = strategy_results
    
    # 生成处理报告
    print(f"\n{'='*80}")
    print("批量处理报告")
    print(f"{'='*80}")
    
    for strategy, strategy_results in results.items():
        print(f"\n策略: {strategy}")
        success_count = sum(1 for r in strategy_results.values() if r['status'] == 'success')
        total_count = len(strategy_results)
        
        print(f"  成功: {success_count}/{total_count}")
        
        for dataset, result in strategy_results.items():
            if result['status'] == 'success':
                print(f"  ✓ {dataset}: {result['shape']}")
            else:
                print(f"  ✗ {dataset}: {result.get('error', '未知错误')}")
    
    # 推荐最佳策略
    print(f"\n{'='*80}")
    print("处理建议")
    print(f"{'='*80}")
    
    print("基于数据质量分析，推荐使用以下策略：")
    print("\n1. 保守策略 (conservative):")
    print("   - 适用于: 模型训练、预测分析")
    print("   - 特点: 保留更多数据，插值质量较高")
    print("   - 数据保留率: 约60-70%")
    
    print("\n2. 激进策略 (aggressive):")
    print("   - 适用于: 数据清洗、质量控制")
    print("   - 特点: 删除更多低质量数据，保留高质量数据")
    print("   - 数据保留率: 约40-50%")
    
    print("\n3. 分段策略 (segmented):")
    print("   - 适用于: 时间序列分析、趋势研究")
    print("   - 特点: 保留所有时间段，但质量不同")
    print("   - 数据保留率: 约80-90%")
    
    print(f"\n处理完成！结果保存在: {output_base}")
    
    return results

if __name__ == "__main__":
    batch_process_datasets()