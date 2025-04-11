import os
import sys
import json
import copy
import shutil
import warnings

import torch
import torch.nn as nn
import dgl
import numpy as np
from tqdm import tqdm



# 将项目目录添加到 sys.path
sys.path.append('..')

# ======= 这里是你项目的内部导入 (需保证文件路径正确) =======
from EarlyStopping import EarlyStopping
from metrics import get_metric
from utils import (
    set_random_seed,
    convert_to_gpu,
    get_optimizer_and_lr_scheduler,
    get_llm_data_loader,
    get_n_params,
    load_dgl_data
)

from llm import *
# ========================================


###############################################################################
# 超参数设置
###############################################################################
args = {
    'dataset': 'heterogeneous_graph',   # 数据集名称
    'model_name': 'Qwen2.5_7B',               # 模型名称
    'label': 'share',
    'mode': 'val',
    'seed': 2,
    'predict_category': 'subtopic',
    'cuda': 4,
    'learning_rate': 1e-5,
    'hidden_units': [512, 512],
    'n_layers': 2,
    'num_heads': 4, 
    'residual': True,
    'dropout': 0.5,
    'n_bases': -1,
    'use_self_loop': True,
    'batch_size': 40,
    'node_neighbors_min_num': 15,
    'optimizer': 'adam',
    'weight_decay': 0,
    'epochs': 500,
    'patience': 5,
    'lora': True,
    'freeze': False,
    'prompt_count': 0,
}

# 设置训练设备
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
if args["cuda"] >= 0:
    torch.cuda.set_device(args["cuda"])


###############################################################################
# 评估函数：evaluate
# - 在 validate/test 阶段使用
###############################################################################
@torch.no_grad()
def evaluate(model: nn.Module,
             loader: dgl.dataloading.DataLoader,
             loss_func: nn.Module,
             labels: torch.Tensor,
             encoding: dict,
             predict_category: str,
             device: str,
             mode: str = 'validate'):
    """
    Evaluate the model's performance on the validation/test set.
    """
    model.eval()
    y_trues = []
    y_predicts = []
    total_loss = 0.0

    loader_tqdm = tqdm(loader, ncols=120)
    for batch_idx, (graph_emb, label_idx) in enumerate(loader_tqdm):
        # Move graph_emb to the specified device
        graph_emb = convert_to_gpu(graph_emb, device=device)
        graph_feature = {'graph_feature': graph_emb}

        # Repeat encoding for the current batch size
        batch_encoding = {key: value.repeat(graph_emb.size(0), 1) for key, value in encoding.items()}
        
        # Forward pass: pass graph features and encoding to the model
        y_pred = model(**{**graph_feature, **batch_encoding})[0]
        
        # True labels: get the labels for the batch
        y_true = convert_to_gpu(labels[label_idx], device=device)

        # Calculate loss
        loss = loss_func(y_pred.float(), y_true.float())
        total_loss += loss.item()

        # Collect results
        y_trues.append(y_true.detach().cpu())
        y_predicts.append(y_pred.detach().cpu())

        loader_tqdm.set_description(f'{mode} [{batch_idx}] - loss: {loss.item():.4f}')

    total_loss /= (batch_idx + 1)
    y_trues = torch.cat(y_trues, dim=0)
    y_predicts = torch.cat(y_predicts, dim=0)

    return total_loss, y_trues, y_predicts


@torch.no_grad()
def final_batch_evaluation(model: nn.Module,
                           train_loader: dgl.dataloading.DataLoader,
                           val_loader: dgl.dataloading.DataLoader,
                           test_loader: dgl.dataloading.DataLoader,
                           labels: torch.Tensor,
                           encoding: dict,
                           args: dict):
    """
    Perform the final evaluation using mini-batch inference on train, validation, and test sets.
    """
    model.eval()
    device = args['device']
    predict_category = args['predict_category']

    def batch_inference(loader, predict_category, device):
        y_trues, y_preds = [], []
        loader_tqdm = tqdm(loader, ncols=120, desc="Final evaluation")
        for batch_idx, (graph_emb, label_idx) in enumerate(loader_tqdm):
            # Move graph_emb to the specified device
            graph_emb = convert_to_gpu(graph_emb, device=device)
            graph_feature = {'graph_feature': graph_emb}

            # Repeat encoding for the current batch size
            batch_encoding = {key: value.repeat(graph_emb.size(0), 1) for key, value in encoding.items()}

            # Forward pass to get node representations
            y_pred = model(**{**graph_feature, **batch_encoding})[0]

            # True labels for the batch
            y_true = convert_to_gpu(labels[label_idx], device=device)


            y_trues.append(y_true.cpu())
            y_preds.append(y_pred.cpu())

        # Concatenate all results
        y_trues = torch.cat(y_trues, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        return y_trues, y_preds

    # Inference for train, validation, and test loaders
    train_y_true, train_y_pred = batch_inference(train_loader, predict_category, device)
    val_y_true, val_y_pred = batch_inference(val_loader, predict_category, device)
    test_y_true, test_y_pred = batch_inference(test_loader, predict_category, device)

    # Calculate metrics
    train_scores = get_metric(
        y_true=train_y_true.numpy(),
        y_pred=train_y_pred.numpy()
    )
    val_scores = get_metric(
        y_true=val_y_true.numpy(),
        y_pred=val_y_pred.numpy()
    )
    test_scores = get_metric(
        y_true=test_y_true.numpy(),
        y_pred=test_y_pred.numpy()
    )

    result_json = {
        "train_scores": train_scores,
        "valid_scores": val_scores,
        "test_scores": test_scores
    }
    print("Final evaluation:", result_json)

    # Save results
    save_result_folder = f"../../result/{args['dataset']}/{args['label']}"
    os.makedirs(save_result_folder, exist_ok=True)
    if args['lora'] and not args['freeze']:
        save_result_path = os.path.join(save_result_folder, f"{args['model_name']}_{args['seed']}_{args['prompt_count']}.json")
    elif args['freeze']:
        save_result_path = os.path.join(save_result_folder, f"{args['model_name']}_{args['seed']}_freeze.json")
    else:
        save_result_path = os.path.join(save_result_folder, f"{args['model_name']}_{args['seed']}_nolora.json")
    with open(save_result_path, 'w') as f:
        json.dump(result_json, f, indent=4)

    print(f"Results saved to {save_result_path}")
    print("Done.")


###############################################################################
# main 逻辑
###############################################################################
def main(args):
    # 忽略一些警告
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)  # 避免多线程过载

    # 设置随机种子，保证实验可复现
    set_random_seed(args['seed'])

    print(f"========== Loading dataset: {args['dataset']} ==========")

    # 从文件加载图数据
    with torch.no_grad():
        target_node_type, labels = load_dgl_data(args['device'], label=args['label'])

        num_classes = 1  # 如果是回归或只有1维输出, num_classes=1

    print(f"Graph loaded. Node type to predict: {target_node_type}")
    print(f"Labels shape: {labels.shape}, #classes={num_classes}")

    # 构建 DataLoader
    train_loader, val_loader, test_loader = get_llm_data_loader(
        args['dataset'],
        args['label'],
        args['seed'],
        args['batch_size'],
    )

    # 1) 构建 模型
    model_name = "./" + args['model_name']
    model, tokenizer, encoding = initialize_model(model_name, args['label'], args['device'], args['lora'], args['freeze'], args['prompt_count'])

    # 3) 组合
    model = convert_to_gpu(model, device=args['device'])
    print(model)
    print(f"Model #Params: {get_n_params(model)}.")

    # 优化器、scheduler
    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model, args['optimizer'], args['learning_rate'], args['weight_decay'],
        steps_per_epoch=len(train_loader), epochs=args['epochs']
    )

    # EarlyStopping
    save_model_folder = f"../../result/{args['dataset']}/{args['label']}"
    os.makedirs(save_model_folder, exist_ok=True)
    save_model_name = args['model_name'] + str(args['prompt_count'])
    if args['freeze']:
        save_model_name += "_freeze"
    elif args['lora'] == False:
        save_model_name += "_nolora"
    early_stopping = EarlyStopping(
        patience=args['patience'],
        save_model_folder=save_model_folder,
        save_model_name=save_model_name
    )

    # 定义损失函数 (MSE)
    # loss_func = nn.HuberLoss()

    loss_func = nn.MSELoss()

    train_steps = 0

    # ===================== TRAIN =====================
    if args['mode'] == 'train':
        for epoch in range(args['epochs']):
            model.train()
            train_loader_tqdm = tqdm(train_loader, ncols=120)

            train_loss_accum = 0.0
            y_trues, y_preds = [], []

            for batch_idx, (graph_emb, label_idx) in enumerate(train_loader_tqdm):
                graph_emb = convert_to_gpu(graph_emb, device=args['device'])
                graph_feature = {'graph_feature': graph_emb}

                # Repeat encoding for the current batch size
                batch_encoding = {key: value.repeat(graph_emb.size(0), 1) for key, value in encoding.items()}
                y_pred = model(**{**graph_feature, **batch_encoding})[0]
                y_true = convert_to_gpu(labels[label_idx], device=args['device'])
                loss = loss_func(y_pred.float(), y_true.float())

                train_loss_accum += loss.item()
                y_trues.append(y_true.detach().cpu())
                y_preds.append(y_pred.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loader_tqdm.set_description(
                    f'Epoch {epoch+1}/{args["epochs"]} [batch {batch_idx}] train loss: {loss.item():.4f}'
                )

                train_steps += 1
                scheduler.step(train_steps)

            train_loss_accum /= (batch_idx + 1)
            y_trues = torch.cat(y_trues, dim=0)
            y_preds = torch.cat(y_preds, dim=0)

            # 评估：验证集
            val_loss, val_y_true, val_y_pred = evaluate(
                model, val_loader, loss_func, labels, encoding,
                predict_category=args['predict_category'],
                device=args['device'], mode='validate'
            )

            # 评估：测试集
            test_loss, test_y_true, test_y_pred = evaluate(
                model, test_loader, loss_func, labels, encoding, 
                predict_category=args['predict_category'],
                device=args['device'], mode='test'
            )

            # 计算一些metric
            val_scores = get_metric(
                y_true=val_y_true.cpu().numpy(),
                y_pred=val_y_pred.cpu().detach().numpy(),
            )
            test_scores = get_metric(
                y_true=test_y_true.cpu().numpy(),
                y_pred=test_y_pred.cpu().detach().numpy(),
            )

            print(f"========== Epoch {epoch+1}/{args['epochs']} Summary ==========")
            print(f"Train Loss: {train_loss_accum:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}")
            print(f"Val Scores: {val_scores}, Test Scores: {test_scores}")

            early_stop = early_stopping.step([('MAPE', val_scores['MAPE'], False)], model)
            if early_stop:
                print("Early stopping triggered.")
                break

    else:
        # 如果不是训练模式，直接加载已保存的最佳模型
        param_path = os.path.join(save_model_folder, f"{args['model_name']}.pkl")
        params = torch.load(param_path, map_location='cpu')
        # print(model.state_dict().keys())
        model.load_state_dict(params, strict=False)
        model = convert_to_gpu(model, device=args['device'])

    # 加载最优模型
    early_stopping.load_checkpoint(model)

    # ===================== 最终评估并保存结果 =====================
    final_batch_evaluation(model, train_loader, val_loader, test_loader, labels, encoding, args)


###############################################################################
# 直接运行
###############################################################################
if __name__ == '__main__':
    main(args)
