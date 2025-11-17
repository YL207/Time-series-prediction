import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def inverse_transform(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    out = df.copy()
    tr = meta.get('transform', {})
    for col in out.columns:
        if col not in tr:
            continue
        info = tr[col]
        # 逆 minmax（如果存在）
        if 'minmax' in info:
            mn, mx = info['minmax'].get('min', 0.0), info['minmax'].get('max', 1.0)
            out[col] = out[col] * (mx - mn) + mn
        # 逆 robust 标准化（如果存在）
        if 'robust' in info:
            median = info['robust'].get('median', 0.0)
            iqr = info['robust'].get('iqr', 1.0)
            iqr = iqr if iqr != 0 else 1.0
            out[col] = out[col] * iqr + median
        # 逆 log1p（如果存在）
        if info.get('log1p', False):
            out[col] = np.expm1(out[col])
        # 注意：winsorize 是截断，无法完全恢复，只能保持当前值。
    return out


def main():
    parser = argparse.ArgumentParser(description='Inverse-transform data/predictions using saved meta')
    parser.add_argument('--inputs', nargs='+', required=True, help='CSV files to inverse-transform')
    parser.add_argument('--meta', required=True, help='Meta JSON saved by robust_preprocess.py')
    parser.add_argument('--outputs', nargs='+', required=True, help='Output CSV paths (same length as inputs)')
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), 'inputs and outputs must have same length'

    with open(args.meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    for inp, outp in zip(args.inputs, args.outputs):
        df = pd.read_csv(inp)
        inv = inverse_transform(df, meta)
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(outp, index=False, encoding='utf-8')
        print(f'Saved inverse-transformed: {outp}')


if __name__ == '__main__':
    main()

