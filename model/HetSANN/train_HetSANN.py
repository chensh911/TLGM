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
from HetSANN import HetSANN
from Classifier import Classifier
from EarlyStopping import EarlyStopping
from metrics import get_metric
from utils import (
    set_random_seed,
    convert_to_gpu,
    get_optimizer_and_lr_scheduler,
    get_node_data_loader,
    get_n_params,
    load_dgl_data
)
# ========================================


###############################################################################
# 超参数设置
###############################################################################
args = {
    'dataset': 'heterogeneous_graph',   # 数据集名称
    'model_name': 'HetSANN',               # 模型名称
    'label': 'share',
    'mode': 'val',
    'seed': 2,
    'predict_category': 'subtopic',
    'cuda': 3,
    'learning_rate': 1e-5,
    'hidden_units': [512, 512],
    'n_layers': 2,
    'num_heads': 4, 
    'residual': True,
    'dropout': 0.5,
    'n_bases': -1,
    'use_self_loop': True,
    'batch_size': 128,
    'node_neighbors_min_num': 15,
    'optimizer': 'adam',
    'weight_decay': 0,
    'epochs': 500,
    'patience': 30
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
             predict_category: str,
             device: str,
             mode: str = 'validate'):
    """
    评估模型在验证集或测试集上的性能。

    Parameters
    ----------
    model : nn.Module
        模型 (包含 HetSANN + Classifier)
    loader : dgl.dataloading.DataLoader
        验证或测试集的 DataLoader
    loss_func : nn.Module
        损失函数
    labels : torch.Tensor
        所有节点的标签张量
    predict_category : str
        本次预测的节点类型
    device : str
        设备, 'cpu' or 'cuda:X'
    mode : str
        'validate' 或 'test'，只是用来打印日志

    Returns
    -------
    total_loss : float
        该数据集上的平均 Loss
    y_trues : torch.Tensor
        真实标签
    y_predicts : torch.Tensor
        模型预测值
    """
    model.eval()
    y_trues = []
    y_predicts = []
    total_loss = 0.0

    loader_tqdm = tqdm(loader, ncols=120)
    for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
        # 将 blocks, input_features 等搬到 GPU
        blocks = [convert_to_gpu(b, device=device) for b in blocks]
        input_features = {
            ntype: blocks[0].srcnodes[ntype].data['feat']
            for ntype in input_nodes.keys()
        }

        # 前向
        node_repr = model[0](blocks, copy.deepcopy(input_features))
        y_pred = model[1](node_repr[predict_category])

        # 获取真值
        y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

        # 计算 Loss
        loss = loss_func(y_pred, y_true)
        total_loss += loss.item()

        # 收集结果
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
                           args: dict):
    """
    使用批量推断方式对训练/验证/测试集进行最终评估，并保存结果。

    参数：
        model: nn.Module
            包含 RGCN 与 Classifier 的组合模型。
        train_loader, val_loader, test_loader: dgl.dataloading.DataLoader
            分别对应训练、验证、测试集的 DataLoader。
        labels: torch.Tensor
            所有节点的标签张量。
        args: dict
            包含设备、预测节点类型、数据集名称、模型名称、随机种子等配置信息。

    作用：
        1. 对每个 loader 分别采用 mini-batch 方式获取前向预测结果。
        2. 将各批次的预测结果拼接起来，与真实标签比较后计算各项指标。
        3. 将评估结果保存为 JSON 文件。
    """

    model.eval()
    device = args['device']
    predict_category = args['predict_category']

    def batch_inference(loader, predict_category, device):
        y_trues, y_preds = [], []
        loader_tqdm = tqdm(loader, ncols=120, desc="Final evaluation")
        for input_nodes, output_nodes, blocks in loader_tqdm:
            # 将每个 block 移动到指定设备
            blocks = [convert_to_gpu(b, device=device) for b in blocks]

            # 提取输入特征（注意这里按照每个节点类型获取对应特征）
            input_features = {
                ntype: blocks[0].srcnodes[ntype].data['feat']
                for ntype in input_nodes.keys()
            }

            # 前向传播得到节点表征
            # 注意这里用 copy.deepcopy 避免在多层传播中修改原始特征
            node_repr = model[0](blocks, copy.deepcopy(input_features))
            # 仅取目标节点类型的表征，经过分类器得到预测值
            y_pred = model[1](node_repr[predict_category])

            # 获取该批次中目标节点的真实标签，并搬到相同设备
            y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

            y_trues.append(y_true.cpu())
            y_preds.append(y_pred.cpu())
            # print(output_nodes[predict_category])
            # print(y_pred)
            # exit()

        # 拼接所有批次的结果
        y_trues = torch.cat(y_trues, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        return y_trues, y_preds

    # 对训练、验证、测试集分别进行批量推断
    train_y_true, train_y_pred = batch_inference(train_loader, predict_category, device)
    val_y_true, val_y_pred = batch_inference(val_loader, predict_category, device)
    test_y_true, test_y_pred = batch_inference(test_loader, predict_category, device)

    # 计算各数据集上的指标（例如：MAPE、RMSE 等，根据 get_metric 内部实现）
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

    # 保存结果到指定文件夹
    save_result_folder = f"../../result/{args['dataset']}/{args['label']}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args['model_name']}_{args['seed']}.json")
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
    # 说明: load_dgl_data() 是你自定义的函数, 需根据实际实现修改
    with torch.no_grad():
        # 这里 embedding_size=args['hidden_units'][0] 用于指示初始特征大小?
        graph, target_node_type, idx_tuple, labels = load_dgl_data(args['device'], label=args['label'])

        (train_idx, valid_idx, test_idx) = idx_tuple
        num_classes = 1  # 如果是回归或只有1维输出, num_classes=1

    print(f"Graph loaded. Node type to predict: {target_node_type}")
    print(f"Labels shape: {labels.shape}, #classes={num_classes}")

    # 构建 DataLoader
    train_loader, val_loader, test_loader = get_node_data_loader(
        args['node_neighbors_min_num'],  # 每层最少邻居数
        args['n_layers'],               # HetSANN 层数
        graph,
        batch_size=args['batch_size'],
        sampled_node_type=args['predict_category'],
        train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx,
        num_workers=0
    )

    # 1) 构建 HetSANN
    hetsann = HetSANN(graph=graph,
                      input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                      hidden_dim=args['hidden_units'][-1], num_layers=args['n_layers'],
                      n_heads=args['num_heads'], dropout=args['dropout'], residual=args['residual'])

    # 2) 构建分类器 (回归/分类等)
    classifier = Classifier(
        n_hid=args['hidden_units'][-1] * args['num_heads'],
        n_out=num_classes
    )

    # 3) 组合
    model = nn.Sequential(hetsann, classifier)
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
    early_stopping = EarlyStopping(
        patience=args['patience'],
        save_model_folder=save_model_folder,
        save_model_name=args['model_name']
    )

    # 定义损失函数 (MSE)
    loss_func = nn.MSELoss()

    train_steps = 0

    # ===================== TRAIN =====================
    if args['mode'] == 'train':
        for epoch in range(args['epochs']):
            model.train()
            train_loader_tqdm = tqdm(train_loader, ncols=120)

            train_loss_accum = 0.0
            y_trues, y_preds = [], []

            for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
                blocks = [convert_to_gpu(b, device=args['device']) for b in blocks]
                input_features = {
                    ntype: blocks[0].srcnodes[ntype].data['feat']
                    for ntype in input_nodes.keys()
                }

                node_repr = model[0](blocks, copy.deepcopy(input_features))
                y_pred = model[1](node_repr[args['predict_category']])

                y_true = convert_to_gpu(labels[output_nodes[args['predict_category']]], device=args['device'])
                loss = loss_func(y_pred.float(), y_true.float())

                train_loss_accum += loss.item()
                y_trues.append(y_true.detach().cpu())
                y_preds.append(y_pred.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
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
                model, val_loader, loss_func, labels,
                predict_category=args['predict_category'],
                device=args['device'], mode='validate'
            )

            # 评估：测试集
            test_loss, test_y_true, test_y_pred = evaluate(
                model, test_loader, loss_func, labels,
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
        model.load_state_dict(params)
        model = convert_to_gpu(model, device=args['device'])

    # 加载最优模型
    early_stopping.load_checkpoint(model)

    # ===================== 最终评估并保存结果 =====================
    final_batch_evaluation(model, train_loader, val_loader, test_loader, labels, args)


###############################################################################
# 直接运行
###############################################################################
if __name__ == '__main__':
    main(args)
