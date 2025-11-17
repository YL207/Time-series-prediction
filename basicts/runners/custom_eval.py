import torch
import numpy as np
import pandas as pd


def nse(y_true, y_pred):
    """
    计算 Nash-Sutcliffe Efficiency (NSE).
    更常用于水文/流量等时间序列场景。
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - numerator / denominator if denominator != 0 else np.nan

def rmse(y_true, y_pred):
    """
    计算 RMSE (Root Mean Squared Error).
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true, y_pred):
    """
    计算 NRMSE. 常见公式: RMSE / (max(y_true) - min(y_true)).
    """
    range_y = np.max(y_true) - np.min(y_true)
    return rmse(y_true, y_pred) / range_y if range_y != 0 else np.nan

def r2_score(y_true, y_pred):
    """
    计算 R² (Coefficient of Determination).
    """
    sstotal = np.sum((y_true - np.mean(y_true)) ** 2)
    ssres = np.sum((y_true - y_pred) ** 2)
    return 1 - ssres / sstotal if sstotal != 0 else np.nan

def eval_csv_unnorm(returns_all, target_columns, path): 
    # pred = returns_all['prediction'][:, i, :, :]
    # real = returns_all['target'][:, i, :, :]
    
    y_pred = returns_all['prediction_unnorm'].cpu().numpy()
    y_true = returns_all['target_unnorm'].cpu().numpy()

    # 为避免除以 0，可以设置一个小阈值 (epsilon)，如 1e-8
    epsilon = 1e-8
    # 计算占比：(y_pred - y_true) / y_true
    # 这里取绝对值，表示误差占真实值的百分比
    ratios = np.abs(y_pred - y_true) / (y_true + epsilon)

    # 统计各维度下占比小于 5% 和 10% 的样本数
    threshold_5 = 0.05
    threshold_10 = 0.10
    num_dimensions = returns_all['prediction'].shape[-2]

    count_less_5 = []
    count_less_10 = []
    nse_val = []
    rmse_val = []
    nrmse_val = []
    r2_val = []

    total_samples_per_dimension = y_true.shape[0] * y_true.shape[1]
    csv_dt = {}
    for d in range(num_dimensions):
        dimension_ratios = ratios[:, :, d, :]  # 取出第 d 个维度的误差占比
        # 统计占比小于5%的样本数
        cnt_5 = np.sum(dimension_ratios < threshold_5)
        # 统计占比小于10%的样本数
        cnt_10 = np.sum(dimension_ratios < threshold_10)
        
        count_less_5.append(cnt_5)
        count_less_10.append(cnt_10)
        
        y_true_d = y_true[:, :, d, :]
        y_pred_d = y_pred[:, :, d, :]
        csv_dt[target_columns[d]] = y_true_d.reshape(-1).tolist()
        csv_dt[f'{target_columns[d]}_pred'] = y_pred_d.reshape(-1).tolist()
        
        nse_val_d = nse(y_true_d, y_pred_d)
        rmse_val_d = rmse(y_true_d, y_pred_d)
        nrmse_val_d = nrmse(y_true_d, y_pred_d)
        r2_val_d = r2_score(y_true_d, y_pred_d)

        nse_val.append(nse_val_d)
        rmse_val.append(rmse_val_d)
        nrmse_val.append(nrmse_val_d)
        r2_val.append(r2_val_d)
        
    df = pd.DataFrame(csv_dt)
    df.to_csv(path[:-4]+'_result_unnorm.csv', index=False)

    per_less_5 = [c / total_samples_per_dimension for c in count_less_5]
    per_less_10 = [c / total_samples_per_dimension for c in count_less_10]

    # average
    per_less_5.append(np.average(per_less_5))
    per_less_10.append(np.average(per_less_10))
    count_less_5.append(np.sum(count_less_5))
    count_less_10.append(np.sum(count_less_10))
    nse_val.append(np.average(nse_val))
    rmse_val.append(np.average(rmse_val))
    nrmse_val.append(np.average(nrmse_val))
    r2_val.append(np.average(r2_val))
    
    # all
    per_less_5.append(per_less_5[-1])
    per_less_10.append(per_less_10[-1])
    count_less_5.append(count_less_5[-1])
    count_less_10.append(count_less_10[-1])
    nse_val.append(nse(y_true, y_pred))
    rmse_val.append(rmse(y_true, y_pred))
    nrmse_val.append(nrmse(y_true, y_pred))
    r2_val.append(r2_score(y_true, y_pred))
    
    features = target_columns + ['avg','all']
    sample_counts = [total_samples_per_dimension] * (num_dimensions+2)

    df = pd.DataFrame({
        "feature": features,
        "count_less_5": count_less_5,
        "count_less_10": count_less_10,
        'samples_count': sample_counts,
        "per_less_5": per_less_5,
        "per_less_10": per_less_10,
        "nse_val": nse_val,
        "rmse_val": rmse_val,
        "nrmse_val": nrmse_val,
        "r2_val": r2_val
    })
    
    df.to_csv(path[:-4]+'_unnorm.csv', index=False)


def eval_csv_norm(returns_all, target_columns, path): 
    # pred = returns_all['prediction'][:, i, :, :]
    # real = returns_all['target'][:, i, :, :]
    
    y_pred = returns_all['prediction'].cpu().numpy()
    y_true = returns_all['target'].cpu().numpy()

    # 为避免除以 0，可以设置一个小阈值 (epsilon)，如 1e-8
    epsilon = 1e-8
    # 计算占比：(y_pred - y_true) / y_true
    # 这里取绝对值，表示误差占真实值的百分比
    ratios = np.abs(y_pred - y_true) / (y_true + epsilon)

    # 统计各维度下占比小于 5% 和 10% 的样本数
    threshold_5 = 0.05
    threshold_10 = 0.10
    num_dimensions = returns_all['prediction'].shape[-2]

    count_less_5 = []
    count_less_10 = []
    nse_val = []
    rmse_val = []
    nrmse_val = []
    r2_val = []

    total_samples_per_dimension = y_true.shape[0] * y_true.shape[1]
    csv_dt = {}
    for d in range(num_dimensions):
        dimension_ratios = ratios[:, :, d, :]  # 取出第 d 个维度的误差占比
        # 统计占比小于5%的样本数
        cnt_5 = np.sum(dimension_ratios < threshold_5)
        # 统计占比小于10%的样本数
        cnt_10 = np.sum(dimension_ratios < threshold_10)
        
        count_less_5.append(cnt_5)
        count_less_10.append(cnt_10)
        
        y_true_d = y_true[:, :, d, :]
        y_pred_d = y_pred[:, :, d, :]
        csv_dt[target_columns[d]] = y_true_d.reshape(-1).tolist()
        csv_dt[f'{target_columns[d]}_pred'] = y_pred_d.reshape(-1).tolist()
        
        nse_val_d = nse(y_true_d, y_pred_d)
        rmse_val_d = rmse(y_true_d, y_pred_d)
        nrmse_val_d = nrmse(y_true_d, y_pred_d)
        r2_val_d = r2_score(y_true_d, y_pred_d)

        nse_val.append(nse_val_d)
        rmse_val.append(rmse_val_d)
        nrmse_val.append(nrmse_val_d)
        r2_val.append(r2_val_d)
        
    df = pd.DataFrame(csv_dt)
    df.to_csv(path[:-4]+'_result.csv', index=False)

    per_less_5 = [c / total_samples_per_dimension for c in count_less_5]
    per_less_10 = [c / total_samples_per_dimension for c in count_less_10]

    # average
    per_less_5.append(np.average(per_less_5))
    per_less_10.append(np.average(per_less_10))
    count_less_5.append(np.sum(count_less_5))
    count_less_10.append(np.sum(count_less_10))
    nse_val.append(np.average(nse_val))
    rmse_val.append(np.average(rmse_val))
    nrmse_val.append(np.average(nrmse_val))
    r2_val.append(np.average(r2_val))
    
    # all
    per_less_5.append(per_less_5[-1])
    per_less_10.append(per_less_10[-1])
    count_less_5.append(count_less_5[-1])
    count_less_10.append(count_less_10[-1])
    nse_val.append(nse(y_true, y_pred))
    rmse_val.append(rmse(y_true, y_pred))
    nrmse_val.append(nrmse(y_true, y_pred))
    r2_val.append(r2_score(y_true, y_pred))
    
    features = target_columns + ['avg','all']
    sample_counts = [total_samples_per_dimension] * (num_dimensions+2)

    df = pd.DataFrame({
        "feature": features,
        "count_less_5": count_less_5,
        "count_less_10": count_less_10,
        'samples_count': sample_counts,
        "per_less_5": per_less_5,
        "per_less_10": per_less_10,
        "nse_val": nse_val,
        "rmse_val": rmse_val,
        "nrmse_val": nrmse_val,
        "r2_val": r2_val
    })
    
    df.to_csv(path, index=False)