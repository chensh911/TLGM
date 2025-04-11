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
    'model_name': 'HetSANN',              # 模型名称
    'label': 'comment',
    'mode': 'val',
    'seed': 2,
    'predict_category': 'subtopic',
    'cuda': 6,
    'learning_rate': 1e-5,
    'hidden_units': [512, 512],
    'n_layers': 2,
    'num_heads': 4, 
    'residual': True,
    'dropout': 0.5,
    'n_bases': -1,
    'use_self_loop': True,
    'batch_size': 32,
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

        # 前向：调用 HetSANN 提取表征，然后通过 classifier 得到预测值
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


###############################################################################
# 新增函数：保存表征 representation（仅提取 HetSANN 部分，不经过 Classifier）
###############################################################################
@torch.no_grad()
def save_representations(model: nn.Module,
                         train_loader: dgl.dataloading.DataLoader,
                         val_loader: dgl.dataloading.DataLoader,
                         test_loader: dgl.dataloading.DataLoader,
                         args: dict):
    """
    对 train/val/test 三个数据集使用 mini-batch 推断方式，
    仅调用模型的 HetSANN 部分提取表征，然后保存成 .npy 文件。
    同时保存对应的 label_idx。
    """
    model.eval()
    device = args['device']
    predict_category = args['predict_category']

    def get_representation_and_labels(loader, set_name: str):
        representations = []
        label_indices = []  # 用于保存每个 batch 对应的 label 索引
        loader_tqdm = tqdm(loader, desc=f"Extracting {set_name} representations", ncols=120)
        for input_nodes, output_nodes, blocks in loader_tqdm:
            # 将每个 block 移动到指定设备
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            # 获取输入特征（各节点类型）
            input_features = {
                ntype: blocks[0].srcnodes[ntype].data['feat']
                for ntype in input_nodes.keys()
            }
            # 仅调用 HetSANN 提取表征，不经过 classifier
            node_repr = model[0](blocks, copy.deepcopy(input_features))
            # 只提取目标节点类型的表征
            rep = node_repr[predict_category]
            representations.append(rep.detach().cpu())

            # 保存 output_nodes 对应的 label 索引
            label_indices.append(output_nodes[predict_category].cpu())

        representations = torch.cat(representations, dim=0)
        label_indices = torch.cat(label_indices, dim=0)
        return representations, label_indices

    # 分别提取三个数据集的表征和标签索引
    train_rep, train_label_idx = get_representation_and_labels(train_loader, "train")
    val_rep, val_label_idx = get_representation_and_labels(val_loader, "val")
    test_rep, test_label_idx = get_representation_and_labels(test_loader, "test")

    # 保存为 .npy 文件
    save_folder = f"../../result/{args['dataset']}/{args['label']}"
    os.makedirs(save_folder, exist_ok=True)

    # 保存表征
    train_rep_path = os.path.join(save_folder, f"{args['model_name']}_{args['seed']}_train_rep.npy")
    val_rep_path = os.path.join(save_folder, f"{args['model_name']}_{args['seed']}_val_rep.npy")
    test_rep_path = os.path.join(save_folder, f"{args['model_name']}_{args['seed']}_test_rep.npy")

    np.save(train_rep_path, train_rep.numpy())
    np.save(val_rep_path, val_rep.numpy())
    np.save(test_rep_path, test_rep.numpy())

    # 保存 label 索引
    train_label_idx_path = os.path.join(save_folder, f"{args['model_name']}_{args['seed']}_train_label_idx.npy")
    val_label_idx_path = os.path.join(save_folder, f"{args['model_name']}_{args['seed']}_val_label_idx.npy")
    test_label_idx_path = os.path.join(save_folder, f"{args['model_name']}_{args['seed']}_test_label_idx.npy")

    np.save(train_label_idx_path, train_label_idx.numpy())
    np.save(val_label_idx_path, val_label_idx.numpy())
    np.save(test_label_idx_path, test_label_idx.numpy())

    print(f"Representations and label indices saved to:\n  {train_rep_path}\n  {val_rep_path}\n  {test_rep_path}")
    print(f"Label indices saved to:\n  {train_label_idx_path}\n  {val_label_idx_path}\n  {test_label_idx_path}")


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

    # 3) 组合：model[0] 为 HetSANN，model[1] 为 Classifier
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

                # 前向传播：调用 HetSANN 提取表征，再由 classifier 计算预测值
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

            # 计算一些 metric
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

    # ===================== 新增：保存各数据集的节点表征 =====================
    save_representations(model, train_loader, val_loader, test_loader, args)


###############################################################################
# 直接运行
###############################################################################
if __name__ == '__main__':
    main(args)
