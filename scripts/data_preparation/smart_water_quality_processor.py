#!/usr/bin/env python3
"""
智能水质数据处理脚本
基于数据质量分析结果，提供多种处理策略
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_water_quality_processing.log'),
        logging.StreamHandler()
    ]
)

class SmartWaterQualityProcessor:
    """智能水质数据处理器"""
    
    def __init__(self, strategy='conservative'):
        """
        初始化处理器
        
        Args:
            strategy: 处理策略 ('conservative', 'aggressive', 'segmented')
        """
        self.strategy = strategy
        self.water_quality_config = {
            '水温': {'unit': '°C', 'decimals': 1, 'min_val': 0, 'max_val': 50, 'interpolation': 'cubic'},
            'pH': {'unit': '', 'decimals': 2, 'min_val': 0, 'max_val': 14, 'interpolation': 'linear'},
            '溶解氧': {'unit': 'mg/L', 'decimals': 2, 'min_val': 0, 'max_val': 20, 'interpolation': 'cubic'},
            '高锰酸盐指数': {'unit': 'mg/L', 'decimals': 2, 'min_val': 0, 'max_val': 50, 'interpolation': 'cubic'},
            '氨氮': {'unit': 'mg/L', 'decimals': 3, 'min_val': 0, 'max_val': 10, 'interpolation': 'cubic'},
            '总磷': {'unit': 'mg/L', 'decimals': 3, 'min_val': 0, 'max_val': 5, 'interpolation': 'cubic'},
            '总氮': {'unit': 'mg/L', 'decimals': 2, 'min_val': 0, 'max_val': 20, 'interpolation': 'cubic'},
            '电导率': {'unit': 'μS/cm', 'decimals': 1, 'min_val': 0, 'max_val': 2000, 'interpolation': 'linear'},
            '浊度': {'unit': 'NTU', 'decimals': 1, 'min_val': 0, 'max_val': 1000, 'interpolation': 'cubic'}
        }
        
        # 根据策略设置参数
        if strategy == 'conservative':
            self.max_consecutive_missing = 50
            self.outlier_threshold = 3.0
            self.min_data_ratio = 0.3
        elif strategy == 'aggressive':
            self.max_consecutive_missing = 30
            self.outlier_threshold = 2.5
            self.min_data_ratio = 0.5
        else:  # segmented
            self.max_consecutive_missing = 100
            self.outlier_threshold = 3.5
            self.min_data_ratio = 0.2
    
    def detect_encoding(self, file_path):
        """检测文件编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'
    
    def load_data(self, file_path):
        """加载数据"""
        try:
            encoding = self.detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding)
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.sort_values('DATE').reset_index(drop=True)
            logging.info(f"✓ 成功加载 {Path(file_path).name}，形状: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"✗ 加载失败: {e}")
            return None
    
    def analyze_missing_patterns(self, df):
        """分析缺失值模式"""
        missing_stats = {}
        
        for col in df.columns:
            if col != 'DATE':
                missing_mask = df[col].isnull()
                total_count = len(df)
                missing_count = missing_mask.sum()
                missing_rate = missing_count / total_count * 100
                
                # 检测连续缺失
                consecutive_lengths = []
                current_length = 0
                
                for is_missing in missing_mask:
                    if is_missing:
                        current_length += 1
                    else:
                        if current_length > 0:
                            consecutive_lengths.append(current_length)
                            current_length = 0
                
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                
                missing_stats[col] = {
                    'missing_rate': missing_rate,
                    'max_consecutive': max(consecutive_lengths) if consecutive_lengths else 0,
                    'avg_consecutive': np.mean(consecutive_lengths) if consecutive_lengths else 0,
                    'consecutive_gaps': len(consecutive_lengths)
                }
        
        return missing_stats
    
    def remove_long_missing_periods(self, df):
        """删除长时间连续缺失的时间段"""
        logging.info(f"删除连续缺失超过 {self.max_consecutive_missing} 个数据点的时间段...")
        
        # 为每列标记需要删除的时间段
        to_remove = pd.Series([False] * len(df))
        
        for col in df.columns:
            if col != 'DATE' and col in self.water_quality_config:
                missing_mask = df[col].isnull()
                
                # 找到连续缺失的起始和结束位置
                consecutive_missing = []
                start_idx = None
                
                for i, is_missing in enumerate(missing_mask):
                    if is_missing and start_idx is None:
                        start_idx = i
                    elif not is_missing and start_idx is not None:
                        if i - start_idx >= self.max_consecutive_missing:
                            consecutive_missing.append((start_idx, i-1))
                        start_idx = None
                
                # 处理最后一个连续缺失
                if start_idx is not None and len(df) - start_idx >= self.max_consecutive_missing:
                    consecutive_missing.append((start_idx, len(df)-1))
                
                # 标记需要删除的行
                for start, end in consecutive_missing:
                    to_remove.iloc[start:end+1] = True
                    logging.info(f"  删除 {col} 的连续缺失时间段: {df.iloc[start]['DATE']} 到 {df.iloc[end]['DATE']}")
        
        # 删除标记的行
        df_cleaned = df[~to_remove].reset_index(drop=True)
        removed_count = to_remove.sum()
        logging.info(f"删除了 {removed_count} 行数据，剩余 {len(df_cleaned)} 行")
        
        return df_cleaned
    
    def detect_and_remove_outliers(self, df):
        """检测并处理异常值"""
        logging.info("检测和处理异常值...")
        
        for col in df.columns:
            if col != 'DATE' and col in self.water_quality_config:
                valid_data = df[col].dropna()
                if len(valid_data) < 10:
                    continue
                
                # 使用Z-score检测异常值
                z_scores = np.abs((valid_data - valid_data.mean()) / valid_data.std())
                outliers = z_scores > self.outlier_threshold
                
                if outliers.sum() > 0:
                    outlier_indices = valid_data[outliers].index
                    df.loc[outlier_indices, col] = np.nan
                    logging.info(f"  {col}: 标记了 {outliers.sum()} 个异常值为缺失值")
    
    def advanced_imputation(self, df):
        """高级插值方法"""
        logging.info("执行高级插值...")
        
        for col in df.columns:
            if col != 'DATE' and col in self.water_quality_config:
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue
                
                config = self.water_quality_config[col]
                valid_mask = df[col].notna()
                
                if valid_mask.sum() < 2:
                    logging.warning(f"  {col}: 有效数据点不足，跳过插值")
                    continue
                
                # 根据配置选择插值方法
                if config['interpolation'] == 'cubic' and valid_mask.sum() >= 4:
                    try:
                        # 三次样条插值
                        valid_indices = df[valid_mask].index
                        valid_values = df.loc[valid_mask, col].values
                        all_indices = df.index
                        
                        f = interpolate.interp1d(valid_indices, valid_values, 
                                               kind='cubic', bounds_error=False, 
                                               fill_value='extrapolate')
                        interpolated = f(all_indices)
                        df.loc[df[col].isnull(), col] = interpolated[df[col].isnull()]
                        
                    except:
                        # 回退到线性插值
                        df[col] = df[col].interpolate(method='linear')
                else:
                    # 线性插值
                    df[col] = df[col].interpolate(method='linear')
                
                # 前后填充
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # 确保值在合理范围内
                min_val, max_val = config['min_val'], config['max_val']
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                
                logging.info(f"  {col}: 插值完成，剩余缺失值: {df[col].isnull().sum()}")
    
    def knn_imputation(self, df):
        """KNN插值（用于处理复杂缺失模式）"""
        logging.info("执行KNN插值...")
        
        # 准备数值数据
        numeric_cols = [col for col in df.columns if col != 'DATE' and col in self.water_quality_config]
        df_numeric = df[numeric_cols].copy()
        
        # 标准化数据
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric), 
            columns=numeric_cols, 
            index=df_numeric.index
        )
        
        # KNN插值
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_scaled),
            columns=numeric_cols,
            index=df_numeric.index
        )
        
        # 反标准化
        df_imputed = pd.DataFrame(
            scaler.inverse_transform(df_imputed),
            columns=numeric_cols,
            index=df_numeric.index
        )
        
        # 更新原数据框
        for col in numeric_cols:
            df[col] = df_imputed[col]
        
        logging.info("KNN插值完成")
    
    def format_decimal_places(self, df):
        """格式化小数位数"""
        logging.info("格式化小数位数...")
        
        for col in df.columns:
            if col != 'DATE' and col in self.water_quality_config:
                decimal_places = self.water_quality_config[col]['decimals']
                df[col] = df[col].round(decimal_places)
    
    def process_dataset(self, input_file, output_file):
        """处理单个数据集"""
        logging.info(f"\n{'='*60}")
        logging.info(f"处理数据集: {Path(input_file).name}")
        logging.info(f"策略: {self.strategy}")
        logging.info(f"{'='*60}")
        
        # 1. 加载数据
        df = self.load_data(input_file)
        if df is None:
            return None
        
        original_shape = df.shape
        logging.info(f"原始数据形状: {original_shape}")
        
        # 2. 分析缺失值模式
        missing_stats = self.analyze_missing_patterns(df)
        avg_missing_rate = np.mean([stats['missing_rate'] for stats in missing_stats.values()])
        max_consecutive = max([stats['max_consecutive'] for stats in missing_stats.values()])
        
        logging.info(f"平均缺失率: {avg_missing_rate:.1f}%")
        logging.info(f"最大连续缺失: {max_consecutive}")
        
        # 3. 删除长时间连续缺失的时间段
        df = self.remove_long_missing_periods(df)
        
        # 4. 检测和处理异常值
        self.detect_and_remove_outliers(df)
        
        # 5. 插值处理
        if self.strategy == 'segmented':
            # 分段处理：对每个时间段单独插值
            self.advanced_imputation(df)
        else:
            # 保守/激进处理：使用KNN插值
            self.knn_imputation(df)
        
        # 6. 格式化
        self.format_decimal_places(df)
        
        # 7. 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # 8. 生成处理报告
        final_missing_rate = np.mean([df[col].isnull().sum() / len(df) * 100 
                                    for col in df.columns if col != 'DATE'])
        
        logging.info(f"\n处理完成!")
        logging.info(f"原始形状: {original_shape}")
        logging.info(f"处理后形状: {df.shape}")
        logging.info(f"数据保留率: {len(df)/original_shape[0]*100:.1f}%")
        logging.info(f"最终缺失率: {final_missing_rate:.1f}%")
        logging.info(f"输出文件: {output_path}")
        
        return df

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='智能水质数据处理工具')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--strategy', choices=['conservative', 'aggressive', 'segmented'], 
                       default='conservative', help='处理策略')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = SmartWaterQualityProcessor(strategy=args.strategy)
    
    # 处理数据
    result = processor.process_dataset(args.input, args.output)
    
    if result is not None:
        print(f"\n✓ 处理完成！结果保存在: {args.output}")
    else:
        print(f"\n✗ 处理失败！")

if __name__ == "__main__":
    main()