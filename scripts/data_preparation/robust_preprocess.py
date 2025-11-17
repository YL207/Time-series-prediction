import argparse
from pathlib import Path
import json
from typing import List

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    q_low, q_high = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)


def robust_scale_series(s: pd.Series):
    median = float(s.median())
    iqr = float(s.quantile(0.75) - s.quantile(0.25))
    iqr = iqr if iqr != 0 else (s.std() if s.std() != 0 else 1.0)
    return (s - median) / iqr, {'median': median, 'iqr': float(iqr)}


def minmax_scale_series(s: pd.Series):
    s_min, s_max = float(s.min()), float(s.max())
    if s_max == s_min:
        return s * 0.0, {'min': s_min, 'max': s_max}
    return (s - s_min) / (s_max - s_min), {'min': s_min, 'max': s_max}


def transform_dataframe(
    df: pd.DataFrame,
    log1p_cols: List[str],
    standardize: bool = True,
    rescale_minmax: bool = False
) -> pd.DataFrame:
    out = df.copy()
    meta = {}
    # 常见时间列不做数值变换
    time_cols = {'date', 'timestamp', 'time', 'datetime'}

    for col in out.columns:
        if col in time_cols:
            continue
        if not np.issubdtype(out[col].dtype, np.number):
            continue
        # 先截断极端值（不可逆，仅用于稳健性）
        before = out[col].copy()
        out[col] = winsorize_series(out[col], 0.01, 0.99)
        q01, q99 = float(before.quantile(0.01)), float(before.quantile(0.99))
        meta[col] = meta.get(col, {})
        meta[col]['winsor'] = {'q01': q01, 'q99': q99}
        # 针对长尾做 log1p
        if col in log1p_cols:
            out[col] = np.log1p(np.maximum(out[col], 0))
            meta[col]['log1p'] = True
        else:
            meta[col]['log1p'] = False
        # 稳健标准化（可选，会产生负值是正常现象）
        if standardize:
            out[col], rmeta = robust_scale_series(out[col])
            meta[col]['robust'] = rmeta
        # 压到 [0,1] 区间（可选，若启用会消除负值）
        if rescale_minmax:
            out[col], mmeta = minmax_scale_series(out[col])
            meta[col]['minmax'] = mmeta
    out._transform_meta = meta  # attach for caller
    return out


def process_file(
    input_csv: Path,
    output_csv: Path,
    log1p_cols: List[str],
    standardize: bool,
    rescale_minmax: bool,
    encoding: str,
    float_format: str
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    df_out = transform_dataframe(df, log1p_cols, standardize=standardize, rescale_minmax=rescale_minmax)
    df_out.to_csv(output_csv, index=False, encoding=encoding, float_format=float_format)
    print(f'Saved: {output_csv}')
    # 保存变换参数（用于反变换近似恢复到 winsorized-raw 空间）
    meta = getattr(df_out, '_transform_meta', {})
    meta_path = output_csv.with_suffix(output_csv.suffix + '.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'columns': list(df.columns), 'transform': meta}, f, ensure_ascii=False, indent=2)
    print(f'Saved meta: {meta_path}')


def main():
    parser = argparse.ArgumentParser(description='Robust preprocessing for water-quality datasets')
    parser.add_argument('--inputs', nargs='*', default=[], help='Input CSV paths')
    parser.add_argument('--outputs', nargs='*', default=[], help='Output CSV paths (same length as inputs)')
    parser.add_argument('--log1p', nargs='*', default=['氨氮', '总磷', '浊度', '高锰酸盐指数'], help='Columns to apply log1p before scaling')
    parser.add_argument('--no_standardize', action='store_true', help='Disable robust standardization (keeps values non-negative if combined with minmax)')
    parser.add_argument('--rescale_minmax', action='store_true', help='After transforms, rescale each column to [0,1]')
    parser.add_argument('--encoding', default='utf-8', help='Output CSV encoding (default utf-8; previously utf-8-sig)')
    parser.add_argument('--float_format', default='%.6f', help='Float format for saving to reduce file size (default %.6f)')
    args = parser.parse_args()

    standardize = not args.no_standardize

    if args.inputs and args.outputs:
        assert len(args.inputs) == len(args.outputs), 'inputs and outputs must have same length'
        for inp, outp in zip(args.inputs, args.outputs):
            process_file(
                Path(inp), Path(outp), args.log1p,
                standardize=standardize,
                rescale_minmax=args.rescale_minmax,
                encoding=args.encoding,
                float_format=args.float_format
            )
        return

    # 默认批处理四个内置数据集
    base = Path(r'E:\YL\submit-ss\BasicTS/datasets/data_source')
    pairs = [
        (base / 'WLSQ.csv', base / 'WLSQ_robust.csv'),
        (base / 'XFZ.csv', base / 'XFZ_robust.csv'),
        (base / 'XP.csv', base / 'XP_robust.csv'),
        (base / 'XYTQ.csv', base / 'XYTQ_robust.csv'),
    ]
    for inp, outp in pairs:
        if inp.exists():
            process_file(
                inp, outp, args.log1p,
                standardize=standardize,
                rescale_minmax=args.rescale_minmax,
                encoding=args.encoding,
                float_format=args.float_format
            )
        else:
            print(f'Skip (not found): {inp}')


if __name__ == '__main__':
    main()

