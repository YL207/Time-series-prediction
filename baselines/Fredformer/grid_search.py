import os
import sys
import json
from itertools import product
from easydict import EasyDict

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# 切换到项目根目录（确保相对路径正确）
os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))

from basicts.runners import SimpleTimeSeriesForecastingRunner
from easytorch.config import init_cfg
from easytorch.device import set_device_type
from easytorch.utils import set_visible_devices

import torch


def create_config_copy(base_cfg, param_updates):
    """
    创建配置的副本，通过重新构建所有嵌套EasyDict结构。
    不使用copy.deepcopy，而是手动重新构建，确保EasyDict的点号键访问功能正常。
    
    Args:
        base_cfg: 原始配置对象
        param_updates: 需要更新的参数字典（例如 {'patch_len': 4, 'cf_depth': 2}）
        
    Returns:
        EasyDict: 新的配置对象
    """
    # 创建新的顶层EasyDict
    new_cfg = EasyDict()
    
    # 复制所有顶层键
    for key in base_cfg.keys():
        if key == 'MODEL' and 'PARAM' in base_cfg.MODEL:
            # 特殊处理MODEL部分，更新PARAM中的参数
            new_cfg.MODEL = EasyDict()
            new_cfg.MODEL.NAME = base_cfg.MODEL.NAME
            new_cfg.MODEL.ARCH = base_cfg.MODEL.ARCH
            new_cfg.MODEL.FORWARD_FEATURES = base_cfg.MODEL.FORWARD_FEATURES
            new_cfg.MODEL.TARGET_FEATURES = base_cfg.MODEL.TARGET_FEATURES
            
            # 复制MODEL.PARAM并更新参数
            new_cfg.MODEL.PARAM = EasyDict()
            for param_key, param_value in base_cfg.MODEL.PARAM.items():
                # 如果参数在更新列表中，使用新值
                if param_key in param_updates:
                    new_cfg.MODEL.PARAM[param_key] = param_updates[param_key]
                else:
                    new_cfg.MODEL.PARAM[param_key] = param_value
        elif isinstance(base_cfg[key], (EasyDict, dict)):
            # 递归复制嵌套的EasyDict
            new_cfg[key] = EasyDict()
            for sub_key, sub_value in base_cfg[key].items():
                if isinstance(sub_value, (EasyDict, dict)):
                    new_cfg[key][sub_key] = EasyDict(sub_value)
                else:
                    new_cfg[key][sub_key] = sub_value
        else:
            # 直接复制非字典值
            new_cfg[key] = base_cfg[key]
    
    return new_cfg


def run_fredformer_grid_search(gpu_ids: str = '0'):
    """
    执行Fredformer模型的网格搜索，寻找最佳超参数组合。
    
    该函数会遍历所有超参数组合，对每个组合进行完整训练，
    并记录最佳验证指标，最后按指标排序输出结果。
    
    Args:
        gpu_ids (str): 要使用的GPU设备ID，例如 '0' 或 '0,1'。默认为 '0'。
    """
    # 检查数据集路径，如果需要则创建符号链接
    dataset_source = 'datasets/data_source/WLSQ'
    dataset_target = 'datasets/WLSQ'
    
    if os.path.exists(dataset_source) and not os.path.exists(dataset_target):
        # 如果源目录存在但目标目录不存在，创建符号链接或复制
        try:
            # 尝试创建符号链接（Windows需要管理员权限）
            if sys.platform == 'win32':
                # Windows下创建目录链接
                import subprocess
                subprocess.run(['mklink', '/D', dataset_target, os.path.abspath(dataset_source)], 
                             shell=True, check=False)
            else:
                os.symlink(os.path.abspath(dataset_source), dataset_target)
            print(f"创建数据集链接: {dataset_target} -> {dataset_source}")
        except Exception:
            # 如果符号链接失败，尝试复制文件
            import shutil
            if os.path.exists(dataset_target):
                shutil.rmtree(dataset_target)
            shutil.copytree(dataset_source, dataset_target)
            print(f"复制数据集文件: {dataset_source} -> {dataset_target}")
    elif not os.path.exists(dataset_target):
        raise FileNotFoundError(
            f"数据集目录不存在: {dataset_target}。"
            f"请确保数据集在 {dataset_source} 或 {dataset_target} 目录下。"
        )
    
    # 设置GPU设备
    if torch.cuda.is_available():
        set_device_type('gpu')
        set_visible_devices(gpu_ids)
        print(f"使用GPU: {gpu_ids}")
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
    else:
        set_device_type('cpu')
        print("警告: 未检测到GPU，将使用CPU训练（速度会很慢）")
    
    # 定义待搜索的超参数网格
    param_grid = {
        "patch_len": [4, 8],          # 子频段长度
        "cf_depth": [2, 3],           # 模型深度（编码器层数）
        "cf_heads": [24, 48],         # 多头数量
        "cf_mlp": [8, 16],            # 前馈层特征维度
        "cf_head_dim": [128, 256],    # 多头内特征维度
        "cf_dim": [128, 256]          # 自注意力特征维度
    }

    # 计算总组合数
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"开始网格搜索，总共 {total_combinations} 个超参数组合需要训练")

    # 导入基础配置（每次循环都重新导入，确保获得原始配置）
    from baselines.Fredformer.WLSQ import CFG as base_cfg

    # 记录所有组合的结果
    results = []
    failed_combinations = []

    # 使用itertools.product生成所有组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    combination_idx = 0
    for param_combination in product(*param_values):
        combination_idx += 1
        # 将组合转换为字典
        params_dict = dict(zip(param_names, param_combination))
        
        patch_len = params_dict["patch_len"]
        cf_depth = params_dict["cf_depth"]
        cf_heads = params_dict["cf_heads"]
        cf_head_dim = params_dict["cf_head_dim"]
        cf_mlp = params_dict["cf_mlp"]
        cf_dim = params_dict["cf_dim"]
        
        print(f"\n{'='*80}")
        print(f"组合 {combination_idx}/{total_combinations}: patch_len={patch_len}, "
              f"cf_depth={cf_depth}, cf_heads={cf_heads}, cf_head_dim={cf_head_dim}, "
              f"cf_mlp={cf_mlp}, cf_dim={cf_dim}")
        print(f"{'='*80}")
        
        try:
            # 创建配置副本并更新参数
            param_updates = {
                'patch_len': patch_len,
                'cf_depth': cf_depth,
                'cf_heads': cf_heads,
                'cf_head_dim': cf_head_dim,
                'cf_mlp': cf_mlp,
                'cf_dim': cf_dim
            }
            cfg = create_config_copy(base_cfg, param_updates)
            
            # 动态生成checkpoint路径，区分不同参数组合
            cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
                'checkpoints',
                cfg.MODEL.NAME,
                f"WLSQ_{cfg.TRAIN.NUM_EPOCHS}_{cfg.DATASET.PARAM.input_len}_{cfg.DATASET.PARAM.output_len}_"
                f"pl{patch_len}_cd{cf_depth}_ch{cf_heads}_chd{cf_head_dim}_cm{cf_mlp}_cdim{cf_dim}"
            )

            # 使用init_cfg确保配置正确初始化（这可能会修复EasyDict的点号键访问）
            try:
                cfg = init_cfg(cfg, save=False)
            except Exception as e:
                # 如果init_cfg失败，继续使用原配置
                print(f"警告: init_cfg失败，继续使用原配置: {e}")

            # 初始化执行器并启动训练
            runner = cfg.RUNNER(cfg)
            
            # 验证模型是否在GPU上
            if torch.cuda.is_available():
                # 检查模型参数是否在GPU上
                model_device = next(runner.model.parameters()).device
                if model_device.type == 'cuda':
                    print(f"✓ 模型已加载到GPU: {model_device}")
                else:
                    print(f"⚠ 警告: 模型在 {model_device} 上，未使用GPU！")
            
            runner.train(cfg)

            # 获取目标指标名称（例如 'MAE'）
            target_metric_name = cfg.METRICS.TARGET
            # 构建完整的指标键名（例如 'val/MAE'）
            val_metric_key = f'val/{target_metric_name}'
            
            # 记录最佳验证指标
            best_val_metric = runner.best_metrics.get(val_metric_key, None)
            
            if best_val_metric is None:
                print(f"警告: 组合 {combination_idx} 未找到最佳指标 {val_metric_key}")
                failed_combinations.append({
                    "params": params_dict,
                    "error": f"未找到指标 {val_metric_key}"
                })
            else:
                results.append({
                    "params": params_dict,
                    "best_val_metric": best_val_metric,
                    "metric_name": target_metric_name,
                    "ckpt_dir": cfg.TRAIN.CKPT_SAVE_DIR
                })
                print(f"✓ 组合 {combination_idx} 训练完成，最佳 {target_metric_name}: {best_val_metric:.4f}")

        except Exception as e:
            error_msg = f"组合 {combination_idx} 训练失败: {str(e)}"
            print(f"✗ {error_msg}")
            failed_combinations.append({
                "params": params_dict,
                "error": str(e)
            })
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # 保存结果到文件
    results_dir = os.path.join('checkpoints', base_cfg.MODEL.NAME, 'grid_search_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 按目标指标排序结果（MAE越小越好，其他指标根据实际情况调整）
    if results:
        results.sort(key=lambda x: x["best_val_metric"])
        
        # 保存所有结果
        results_file = os.path.join(results_dir, 'grid_search_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "successful_results": results,
                "failed_combinations": failed_combinations,
                "total_combinations": total_combinations,
                "successful_count": len(results),
                "failed_count": len(failed_combinations)
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
        
        # 输出结果摘要
        print("\n" + "="*80)
        print("网格搜索结果（按最优指标排序）：")
        print("="*80)
        for i, res in enumerate(results[:10], 1):  # 只显示前10名
            print(f"第{i}名: 参数={res['params']}, "
                  f"最佳{res['metric_name']}={res['best_val_metric']:.4f}, "
                  f"模型路径={res['ckpt_dir']}")
        
        if len(results) > 10:
            print(f"... (还有 {len(results) - 10} 个结果，详见结果文件)")
        
        print(f"\n最优参数组合：{results[0]['params']}")
        print(f"最优指标值 ({results[0]['metric_name']}): {results[0]['best_val_metric']:.4f}")
        
        if failed_combinations:
            print(f"\n失败的组合数量: {len(failed_combinations)}")
            print("失败详情已保存到结果文件中")
    else:
        print("\n警告: 所有组合都训练失败，请检查错误信息")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Grid search for Fredformer hyperparameters')
    parser.add_argument('-g', '--gpus', type=str, default='0', 
                        help='GPU device IDs to use (e.g., "0" or "0,1"). Default: "0"')
    args = parser.parse_args()
    
    run_fredformer_grid_search(gpu_ids=args.gpus)
