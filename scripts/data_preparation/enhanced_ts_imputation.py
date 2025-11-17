import pandas as pd
import numpy as np
import logging
import re
import sys
from pathlib import Path
from scipy import interpolate
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ts_imputation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 水质指标配置
WATER_QUALITY_CONFIG = {
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

def load_data(file_path):
    """加载数据，支持CSV和Excel格式，自动检测编码"""
    try:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            # 尝试多种编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logging.info(f"成功使用 {encoding} 编码加载CSV文件")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"无法使用任何编码格式读取文件: {file_path}")
                
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
        logging.info(f"成功加载数据，DataFrame 形状: {df.shape}")
        logging.info(f"列名称: {df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        raise

def clean_cell_content(cell):
    """清理单元格内容：去除不可见字符（保留空格）"""
    if isinstance(cell, str):
        return re.sub(r'[\x00-\x1F\x7F]', '', cell)
    return cell

def detect_outliers_iqr(series, factor=1.5):
    """使用IQR方法检测异常值"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (series < lower_bound) | (series > upper_bound)

def detect_outliers_zscore(series, threshold=3):
    """使用Z-score方法检测异常值"""
    z_scores = np.abs(zscore(series.dropna()))
    return z_scores > threshold

def clean_data(df, time_col='DATE', outlier_method='iqr', outlier_factor=1.5):
    """清理数据：去除不可见字符，转换数值类型，检测和处理异常值"""
    try:
        time_data = df[time_col].copy()
        df = df.drop(columns=[time_col]).applymap(clean_cell_content)
        logging.info("已去除所有单元格中的不可见字符（排除时间列）。")

        # 转换为数值类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        logging.info("已将数据转换为数值类型，无法转换的值设为 NaN（排除时间列）。")

        # 处理0值和负值
        for col in df.columns:
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                logging.info(f"列 {col}: 发现 {zero_count} 个0值和 {negative_count} 个负值，将转换为NaN")
                df.loc[df[col] <= 0, col] = np.nan

        # 异常值检测和处理
        for col in df.columns:
            if col in WATER_QUALITY_CONFIG:
                config = WATER_QUALITY_CONFIG[col]
                valid_data = df[col].dropna()
                
                if len(valid_data) > 10:  # 确保有足够的数据点
                    if outlier_method == 'iqr':
                        outliers = detect_outliers_iqr(valid_data, outlier_factor)
                    elif outlier_method == 'zscore':
                        outliers = detect_outliers_zscore(valid_data, outlier_factor)
                    else:
                        outliers = pd.Series([False] * len(valid_data), index=valid_data.index)
                    
                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        logging.warning(f"列 {col}: 检测到 {outlier_count} 个异常值，将转换为NaN")
                        outlier_indices = valid_data[outliers].index
                        df.loc[outlier_indices, col] = np.nan
                    
                    # 物理范围检查
                    min_val, max_val = config['min_val'], config['max_val']
                    out_of_range = (df[col] < min_val) | (df[col] > max_val)
                    out_of_range_count = out_of_range.sum()
                    if out_of_range_count > 0:
                        logging.warning(f"列 {col}: 发现 {out_of_range_count} 个超出物理范围的值，将转换为NaN")
                        df.loc[out_of_range, col] = np.nan

        logging.info("数据清理完成（已处理0值、负值、异常值和超出范围的值）。")
        df[time_col] = time_data
        return df
    except Exception as e:
        logging.error(f"清理数据时出错: {e}")
        raise

def advanced_imputation(df, time_col='DATE'):
    """高级时间序列插值方法"""
    try:
        time_data = df[time_col].copy()
        df_numeric = df.drop(columns=[time_col])
        
        # 转换时间列并确保有序
        df_numeric['__timestamp'] = pd.to_datetime(time_data)
        df_numeric = df_numeric.sort_values('__timestamp')
        
        # 对每个数值列进行高级插值
        for col in df_numeric.columns:
            if col != '__timestamp' and col in WATER_QUALITY_CONFIG:
                logging.info(f"正在对列 {col} 进行高级时间序列插值...")
                
                config = WATER_QUALITY_CONFIG[col]
                missing_before = df_numeric[col].isna().sum()
                
                if missing_before == 0:
                    logging.info(f"列 {col}: 无缺失值，跳过插值")
                    continue
                
                # 获取有效数据点
                valid_mask = df_numeric[col].notna()
                valid_indices = df_numeric[valid_mask].index
                valid_values = df_numeric.loc[valid_mask, col].values
                valid_times = df_numeric.loc[valid_mask, '__timestamp'].values
                
                if len(valid_values) < 2:
                    logging.warning(f"列 {col}: 有效数据点不足，使用前向填充")
                    df_numeric[col] = df_numeric[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # 根据配置选择插值方法
                    interpolation_method = config['interpolation']
                    
                    if interpolation_method == 'cubic' and len(valid_values) >= 4:
                        # 三次样条插值
                        try:
                            f = interpolate.interp1d(
                                valid_times.astype(np.int64), 
                                valid_values, 
                                kind='cubic',
                                bounds_error=False,
                                fill_value='extrapolate'
                            )
                            all_times = df_numeric['__timestamp'].values.astype(np.int64)
                            interpolated = f(all_times)
                            df_numeric.loc[df_numeric[col].isna(), col] = interpolated[df_numeric[col].isna()]
                        except:
                            # 如果三次样条失败，回退到线性插值
                            logging.warning(f"列 {col}: 三次样条插值失败，回退到线性插值")
                            df_numeric[col] = df_numeric.set_index('__timestamp')[col].interpolate(
                                method='time', limit_direction='both'
                            ).values
                    else:
                        # 线性插值
                        df_numeric[col] = df_numeric.set_index('__timestamp')[col].interpolate(
                            method='time', limit_direction='both'
                        ).values
                    
                    # 对于仍然缺失的值，使用前后最近的有效值填充
                    if df_numeric[col].isna().any():
                        df_numeric[col] = df_numeric[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # 确保值在合理范围内
                    min_val, max_val = config['min_val'], config['max_val']
                    df_numeric[col] = df_numeric[col].clip(lower=min_val, upper=max_val)
                
                missing_after = df_numeric[col].isna().sum()
                min_val = df_numeric[col].min()
                max_val = df_numeric[col].max()
                
                logging.info(f"列 {col}: 缺失值 {missing_before} -> {missing_after}, 值范围: {min_val:.6f} - {max_val:.6f}")
        
        # 移除时间戳列并恢复原始顺序
        df_numeric = df_numeric.sort_index()
        df_numeric = df_numeric.drop(columns=['__timestamp'])
        
        # 合并时间列
        df_imputed = pd.concat([time_data, df_numeric], axis=1)
        
        logging.info("高级时间序列插值完成。")
        return df_imputed
    except Exception as e:
        logging.error(f"高级插值时出错: {e}")
        raise

def format_decimal_places(df, time_col='DATE'):
    """根据配置设置各列的小数位数"""
    try:
        logging.info("开始设置各列的小数位数...")
        
        for col in df.columns:
            if col != time_col and col in WATER_QUALITY_CONFIG:
                decimal_places = WATER_QUALITY_CONFIG[col]['decimals']
                df[col] = df[col].round(decimal_places)
                logging.info(f"列 {col}: 设置为 {decimal_places} 位小数")
            elif col != time_col:
                # 对于未在配置中的列，默认保留4位小数
                df[col] = df[col].round(4)
                logging.info(f"列 {col}: 使用默认4位小数")
        
        logging.info("小数位数设置完成")
        return df
    except Exception as e:
        logging.error(f"设置小数位数时出错: {e}")
        raise

def save_data(df, output_file_path, format='csv'):
    """保存数据，支持CSV和Excel格式"""
    try:
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format.lower() in ['xlsx', 'excel']:
            df.to_excel(output_path, index=False, engine='openpyxl')
        else:
            raise ValueError(f"不支持的输出格式: {format}")
            
        logging.info(f"插值后的数据已保存到: {output_path}")
    except Exception as e:
        logging.error(f"保存数据时出错: {e}")
        raise

def generate_data_quality_report(df, time_col='DATE'):
    """生成数据质量报告"""
    try:
        report = []
        report.append("=" * 60)
        report.append("数据质量报告")
        report.append("=" * 60)
        
        for col in df.columns:
            if col != time_col:
                series = df[col]
                total_count = len(series)
                missing_count = series.isna().sum()
                missing_rate = missing_count / total_count * 100
                
                if col in WATER_QUALITY_CONFIG:
                    config = WATER_QUALITY_CONFIG[col]
                    valid_data = series.dropna()
                    if len(valid_data) > 0:
                        min_val = valid_data.min()
                        max_val = valid_data.max()
                        mean_val = valid_data.mean()
                        std_val = valid_data.std()
                        
                        report.append(f"\n列名: {col}")
                        report.append(f"  单位: {config['unit']}")
                        report.append(f"  总数据点: {total_count}")
                        report.append(f"  缺失值: {missing_count} ({missing_rate:.2f}%)")
                        report.append(f"  有效值: {total_count - missing_count}")
                        report.append(f"  最小值: {min_val:.6f}")
                        report.append(f"  最大值: {max_val:.6f}")
                        report.append(f"  平均值: {mean_val:.6f}")
                        report.append(f"  标准差: {std_val:.6f}")
                        report.append(f"  物理范围: [{config['min_val']}, {config['max_val']}]")
                        
                        # 检查是否在物理范围内
                        out_of_range = ((valid_data < config['min_val']) | 
                                      (valid_data > config['max_val'])).sum()
                        if out_of_range > 0:
                            report.append(f"  超出范围: {out_of_range} 个值")
                        else:
                            report.append(f"  范围检查: 通过")
        
        report.append("\n" + "=" * 60)
        report_text = "\n".join(report)
        
        # 保存报告到文件
        report_path = Path("data_quality_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logging.info(f"数据质量报告已保存到: {report_path}")
        print(report_text)
        
    except Exception as e:
        logging.error(f"生成数据质量报告时出错: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版时间序列插值工具')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--format', choices=['csv', 'xlsx'], default='csv', help='输出格式')
    parser.add_argument('--outlier_method', choices=['iqr', 'zscore'], default='iqr', help='异常值检测方法')
    parser.add_argument('--outlier_factor', type=float, default=1.5, help='异常值检测因子')
    parser.add_argument('--time_col', default='DATE', help='时间列名')
    parser.add_argument('--report', action='store_true', help='生成数据质量报告')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        df = load_data(args.input)
        
        # 清理数据
        df_cleaned = clean_data(df, time_col=args.time_col, 
                               outlier_method=args.outlier_method, 
                               outlier_factor=args.outlier_factor)
        
        # 高级插值
        df_imputed = advanced_imputation(df_cleaned, time_col=args.time_col)
        
        # 格式化小数位数
        df_formatted = format_decimal_places(df_imputed, time_col=args.time_col)
        
        # 保存数据
        save_data(df_formatted, args.output, format=args.format)
        
        # 生成数据质量报告
        if args.report:
            generate_data_quality_report(df_formatted, time_col=args.time_col)
        
        logging.info("处理完成！")
        
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
        raise

if __name__ == "__main__":
    main()