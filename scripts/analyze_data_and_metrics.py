import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def summarize_series(s: pd.Series) -> Dict:
    s = s.replace([np.inf, -np.inf], np.nan)
    desc = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        'count': float(desc.get('count', np.nan)),
        'mean': float(desc.get('mean', np.nan)),
        'std': float(desc.get('std', np.nan)),
        'min': float(desc.get('min', np.nan)),
        'p01': float(desc.get('1%') if '1%' in desc else np.nan),
        'p05': float(desc.get('5%') if '5%' in desc else np.nan),
        'median': float(desc.get('50%') if '50%' in desc else np.nan),
        'p95': float(desc.get('95%') if '95%' in desc else np.nan),
        'p99': float(desc.get('99%') if '99%' in desc else np.nan),
        'max': float(desc.get('max', np.nan)),
        'missing_ratio': float(s.isna().mean()),
        'zero_ratio': float((s == 0).mean()) if s.dtype != 'O' else np.nan,
        'skew': float(s.skew(skipna=True)) if s.dtype != 'O' else np.nan,
        'kurt': float(s.kurt(skipna=True)) if s.dtype != 'O' else np.nan,
    }


def load_timeseries_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ['date', 'timestamp', 'time', 'datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.sort_values(col)
            break
    return df


def analyze_dataset(name: str, csv_path: Path) -> Dict:
    report = {'name': name, 'path': str(csv_path)}
    if not csv_path.exists():
        report['error'] = 'file not found'
        return report
    df = load_timeseries_csv(csv_path)
    report['num_rows'] = int(len(df))
    report['num_columns'] = int(df.shape[1])
    report['columns'] = list(df.columns)

    # numeric columns only
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    report['numeric_columns'] = num_cols
    report['distributions'] = {c: summarize_series(df[c]) for c in num_cols[:20]}

    return report


def read_metrics_csv(csv_file: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_file)
    except Exception:
        return pd.DataFrame()


def aggregate_model_metrics(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    return {split: read_metrics_csv(eval_dir / f'{split}_metrics_unnorm.csv')
            for split in ['train', 'val', 'test']
            if (eval_dir / f'{split}_metrics_unnorm.csv').exists()}


def diagnose_split_metrics(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    if df.empty:
        return ['metrics file empty or unreadable']

    if 'r2_val' in df.columns:
        neg = int((df['r2_val'] < 0).sum())
        if neg > 0:
            issues.append(f'{neg} feature(s) r2<0 (worse-than-mean)')
        low = float((df['r2_val'] < 0.5).mean())
        if low > 0.3:
            issues.append('>30% features r2<0.5 (weak generalization)')

    if 'nrmse_val' in df.columns:
        if float(df['nrmse_val'].max()) > 0.5:
            issues.append('high nrmse on some features (>0.5)')

    if 'per_less_5' in df.columns:
        if float(df['per_less_5'].mean()) < 0.3:
            issues.append('low <5% relative error proportion (<0.3)')

    return issues


def summarize_models(root: Path) -> Dict[str, Dict[str, List[str]]]:
    summary: Dict[str, Dict[str, List[str]]] = {}
    if not root.exists():
        return summary
    for model_dir in sorted(root.iterdir()):
        eval_dir = model_dir / '评价指标'
        if not eval_dir.exists():
            continue
        metrics = aggregate_model_metrics(eval_dir)
        if not metrics:
            continue
        summary[model_dir.name] = {split: diagnose_split_metrics(df) for split, df in metrics.items()}
    return summary


def main():
    # 1) 数据分布/质量报告
    datasets = {
        'WLSQ': Path(r'E:\YL\submit-ss\BasicTS/datasets/data_source/WLSQ.csv'),
        'XFZ': Path(r'E:\YL\submit-ss\BasicTS/datasets/data_source/XFZ.csv'),
        'XP': Path(r'E:\YL\submit-ss\BasicTS/datasets/data_source/XP.csv'),
        'XYTQ': Path(r'E:\YL\submit-ss\BasicTS/datasets/data_source/XYTQ.csv'),
    }

    print('==== 数据集分布与质量报告 ====')
    all_reports = []
    for name, path in datasets.items():
        rep = analyze_dataset(name, path)
        all_reports.append(rep)
        if 'error' in rep:
            print(f'{name}: {rep["error"]} ({path})')
            continue
        print(f'{name}: rows={rep["num_rows"]}, cols={rep["num_columns"]}, numeric_cols={len(rep["numeric_columns"])})')
        for col, d in list(rep['distributions'].items())[:6]:
            print(f'  - {col}: mean={d["mean"]:.4f}, std={d["std"]:.4f}, miss={d["missing_ratio"]:.3f}, p01={d["p01"]:.4f}, p99={d["p99"]:.4f}')

    # 2) 模型效果诊断（遍历各模型/评价指标）
    print('\n==== 各模型效果诊断（评价指标） ====')
    models_root = Path(r'E:\YL\submit-ss\BasicTS/checkpoints/')
    summary = summarize_models(models_root)
    if not summary:
        print('未找到任何模型的评价指标目录。')
    else:
        for model, splits in summary.items():
            print(f'[{model}]')
            for split, issues in splits.items():
                print(f'  {split}: ' + ('; '.join(issues) if issues else 'OK'))

    # 可选：保存为CSV
    out_dir = Path('checkpoints/_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    # 保存简要汇总
    pd.DataFrame([{
        'dataset': r['name'],
        'path': r['path'],
        'rows': r.get('num_rows'),
        'cols': r.get('num_columns'),
        'num_numeric_cols': len(r.get('numeric_columns', []))
    } for r in all_reports]).to_csv(out_dir / 'dataset_summary.csv', index=False)


if __name__ == '__main__':
    main()

