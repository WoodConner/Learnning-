import argparse
import random
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, accuracy_score, roc_curve
import sklearn.metrics as m
import math
from torch.utils.data import DataLoader, Dataset
import collections
import scipy.sparse as sp
import re
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import  warnings
warnings.filterwarnings("ignore")

# ========== 评估函数 ==========
def precision(y_true, y_pred):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred)


def auc(y_true, y_scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)


def accuracy(y_true, y_scores):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_scores)


def eval_classification(labels, logits):
    auc_score = roc_auc_score(y_true=labels, y_score=logits)
    p, r, t = precision_recall_curve(y_true=labels, y_score=logits)
    aupr = m.auc(r, p)
    fpr, tpr, threshold = roc_curve(labels, logits)
    spc = 1 - fpr
    j_scores = tpr - fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]

    predicted_label = [1 if i >= youden_thresh else 0 for i in logits]
    p_1 = precision(y_true=labels, y_pred=predicted_label)
    r_1 = recall(y_true=labels, y_pred=predicted_label)
    acc = accuracy_score(y_true=labels, y_pred=predicted_label)
    f1 = f1_score(y_true=labels, y_pred=predicted_label)
    return p_1, r_1, acc, auc_score, aupr, f1


# ========== 数据预处理和工具函数 ==========
class TCRDataset(Dataset):
    def __init__(self, file_path, task_type, entity_dict, max_length=20):
        self.task_type = task_type
        self.entity_dict = entity_dict
        self.max_length = max_length
        self.data = self.load_data_from_file(file_path)

    def load_data_from_file(self, file_path):
        """从CSV文件加载数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        df = pd.read_csv(file_path)
        print(f"从 {file_path} 加载了 {len(df)} 条数据")

        # 根据任务类型处理数据
        if self.task_type == 'pep_hla_label':
            # HLA, Peptide, label, HLA_sequence
            required_cols = ['HLA', 'Peptide', 'label', 'HLA_sequence']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"文件 {file_path} 缺少必要的列: {required_cols}")

            # 为pep-hla任务生成对应的tcr（但此任务不使用）
            df['TCR'] = [f"TCR_{i}" for i in range(len(df))]

        elif self.task_type == 'pep_tcr_label':
            # TCR, Peptide, label
            required_cols = ['TCR', 'Peptide', 'label']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"文件 {file_path} 缺少必要的列: {required_cols}")

            # 为pep-tcr任务生成对应的hla（但此任务不使用）
            df['HLA'] = [f"HLA_{i}" for i in range(len(df))]
            df['HLA_sequence'] = [f"HLA_SEQ_{i}" for i in range(len(df))]

        elif self.task_type == 'tcr_pep_hla_label':
            # Peptide, HLA, TCR, label, HLA_sequence
            required_cols = ['Peptide', 'HLA', 'TCR', 'label', 'HLA_sequence']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"文件 {file_path} 缺少必要的列: {required_cols}")

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # 根据任务类型获取不同的字段
        if self.task_type == 'pep_hla_label':
            pep = sample['Peptide']
            hla = sample['HLA_sequence']
            tcr = sample['TCR']  # 不使用，但需要存在
            label = sample['label']

        elif self.task_type == 'pep_tcr_label':
            pep = sample['Peptide']
            tcr = sample['TCR']
            hla = sample['HLA_sequence']  # 不使用，但需要存在
            label = sample['label']

        elif self.task_type == 'tcr_pep_hla_label':
            pep = sample['Peptide']
            tcr = sample['TCR']
            hla = sample['HLA_sequence']
            label = sample['label']

        # 将序列转换为实体ID
        pep_id = self.sequence_to_entity_id(pep, 'pep')
        tcr_id = self.sequence_to_entity_id(tcr, 'tcr')
        hla_id = self.sequence_to_entity_id(hla, 'hla')

        # 根据任务类型返回不同的数据格式
        if self.task_type == 'pep_hla_label':
            # Task 1: pep-hla序列判断label - hla作为drug1，pep作为cellline
            return torch.tensor(hla_id), torch.tensor(0), torch.tensor(pep_id), torch.tensor(float(label))
        elif self.task_type == 'pep_tcr_label':
            # Task 2: pep和tcr判断label - tcr作为drug1，pep作为cellline
            return torch.tensor(tcr_id), torch.tensor(0), torch.tensor(pep_id), torch.tensor(float(label))
        elif self.task_type == 'tcr_pep_hla_label':
            # Task 3: tcr-pep-hla判断label - tcr作为drug1，hla作为drug2，pep作为cellline
            return torch.tensor(tcr_id), torch.tensor(hla_id), torch.tensor(pep_id), torch.tensor(float(label))
        else:
            raise ValueError(f"未知任务类型: {self.task_type}")

    def sequence_to_entity_id(self, sequence, seq_type):
        """将序列转换为实体ID"""
        hash_val = hash(sequence) % 10000
        entity_key = f"{seq_type}_{hash_val}"

        if entity_key not in self.entity_dict:
            self.entity_dict[entity_key] = len(self.entity_dict)

        return self.entity_dict[entity_key]


def load_real_data():
    """使用DataLoader加载真实数据"""
    # 定义文件路径映射
    file_paths = {
        'pep_hla_label': 'data_random1x/pep_hla_randaom1x.csv',
        'pep_tcr_label': 'data_random1x/pep_tcr_randaom1x.csv',
        'tcr_pep_hla_label': 'data_random1x/trimer_randaom1x.csv'
    }

    # 创建统一的实体字典（用于知识图谱）
    entity_dict = {}
    all_datasets = {}
    all_data_frames = []

    # 为每个任务创建数据集
    for task_name, file_path in file_paths.items():
        try:
            dataset = TCRDataset(file_path, task_name, entity_dict.copy())
            all_datasets[task_name] = dataset
            all_data_frames.append(dataset.data)
            print(f"成功创建 {task_name} 数据集，包含 {len(dataset)} 个样本")
        except Exception as e:
            print(f"创建 {task_name} 数据集失败: {e}")
            # 创建空数据集作为后备
            all_datasets[task_name] = None

    # 合并所有数据用于构建知识图谱
    if all_data_frames:
        all_data_df = pd.concat(all_data_frames, ignore_index=True)
    else:
        all_data_df = pd.DataFrame(columns=['Peptide', 'TCR', 'HLA_sequence', 'label'])

    return all_data_df, all_datasets


def build_kg_from_data(data):
    """从数据构建知识图谱"""
    kg_triples = []
    entity_dict = {}

    for _, row in data.iterrows():
        pep = row['Peptide']
        tcr = row['TCR']
        hla = row['HLA_sequence']

        # 为每个序列创建实体ID
        pep_id = sequence_to_entity_id(pep, 'pep', entity_dict)
        tcr_id = sequence_to_entity_id(tcr, 'tcr', entity_dict)
        hla_id = sequence_to_entity_id(hla, 'hla', entity_dict)

        # 添加三元组关系
        kg_triples.append([pep_id, 0, tcr_id])  # pep-interact-tcr
        kg_triples.append([pep_id, 1, hla_id])  # pep-bind-hla
        kg_triples.append([tcr_id, 2, hla_id])  # tcr-recognize-hla

    return kg_triples, entity_dict


def sequence_to_entity_id(sequence, seq_type, entity_dict):
    """将序列转换为实体ID"""
    hash_val = hash(sequence) % 10000
    entity_key = f"{seq_type}_{hash_val}"

    if entity_key not in entity_dict:
        entity_dict[entity_key] = len(entity_dict)

    return entity_dict[entity_key]


def getKgIndexsFromKgTriples(kg_triples):
    kg_indexs = collections.defaultdict(list)
    for h, r, t in kg_triples:
        kg_indexs[str(h)].append([int(t), int(r)])
    return kg_indexs


def construct_adj(neighbor_sample_size, kg_indexes, entity_num, device):
    print('生成实体邻接列表和关系邻接列表')
    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        if str(entity) in kg_indexes:
            neighbors = kg_indexes[str(entity)]
            n_neighbors = len(neighbors)
            if n_neighbors >= neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)),
                                                   size=neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)),
                                                   size=neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    # 转换为PyTorch张量并移动到设备
    adj_entity = torch.LongTensor(adj_entity).to(device)
    adj_relation = torch.LongTensor(adj_relation).to(device)
    return adj_entity, adj_relation


# ========== KGANS模型 (修改以支持GPU) ==========
class KGANS(nn.Module):
    def __init__(self, args, n_entitys, n_relations, e_dim, r_dim,
                 adj_entity, adj_relation, agg_method='Bi-Interaction'):
        super(KGANS, self).__init__()

        self.entity_embs = nn.Embedding(n_entitys, e_dim, max_norm=1)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1)
        self.dropout = args.dropout
        self.n_iter = args.n_iter
        self.dim = e_dim
        self.WW = nn.Linear(256, 128, bias=False)
        self.n_heads = args.n_heads
        self.adj_entity = adj_entity  # 现在已经是GPU张量
        self.adj_relation = adj_relation  # 现在已经是GPU张量
        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self._init_weight()

        self.dropout_layer = nn.Dropout(self.dropout)

        self.agg_method = agg_method
        self.agg = 'concat'

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        if agg_method == 'concat':
            self.W_concat = nn.Linear(e_dim * 2, e_dim)
        else:
            self.W1 = nn.Linear(e_dim * self.n_heads, e_dim * 2)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(e_dim * self.n_heads, e_dim * 2)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)

        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def get_neighbors_cell(self, cells):
        cells = cells.view((-1, 1))
        entities = [cells]
        relations = []

        for h in range(self.n_iter):
            # 使用索引选择，现在adj_entity已经在GPU上
            neighbor_entities = self.adj_entity[entities[h]].view((entities[h].shape[0], -1))
            neighbor_relations = self.adj_relation[entities[h]].view((entities[h].shape[0], -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
        neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
        return neighbor_entities_embs, neighbor_relations_embs

    def get_neighbors_drug(self, drugs):
        drugs = drugs.view((-1, 1))
        entities = [drugs]
        relations = []

        for h in range(self.n_iter):
            # 使用索引选择，现在adj_entity已经在GPU上
            neighbor_entities = self.adj_entity[entities[h]].view((entities[h].shape[0], -1))
            neighbor_relations = self.adj_relation[entities[h]].view((entities[h].shape[0], -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
        neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
        return neighbor_entities_embs, neighbor_relations_embs

    def sum_aggregator(self, embs):
        e_u = embs[0]
        if self.agg == 'concat':
            for i in range(1, len(embs)):
                e_u = torch.cat((embs[i], e_u), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(embs)):
                e_u += self.WW(embs[i])
        return e_u

    def GATMessagePass(self, h_embs, r_embs, t_embs):
        """
        多注意力消息传递，兼容 CPU / CUDA，不改动模型结构
        """
        device = h_embs.device  # 拿到当前 batch 所在的设备
        muti = []
        for i in range(self.n_heads):
            att_weights = self.attention(torch.cat((h_embs, r_embs), dim=-1)).squeeze(-1)
            att_weights_norm = F.softmax(att_weights, dim=-1)
            emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_embs)
            # 临时 Linear，显式搬到同一设备
            Wx = nn.Linear(self.dim, self.dim).to(device)
            emb_i = Wx(emb_i.sum(dim=1))
            muti.append(emb_i)
        all_emb_i = torch.cat(muti, dim=-1)
        return all_emb_i

    def aggregate(self, h_embs, Nh_embs, agg_method='Bi-Interaction'):
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) \
                + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:
            return self.leakyRelu(self.W1(h_embs + Nh_embs))

    def forward(self, u1, u2, c):
        # Cell line embedding learning process
        t_embs, r_embs = self.get_neighbors_cell(c)
        h_embs = self.entity_embs(c)
        t_vectors_next_iter = [h_embs]

        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                h_embs = torch.cat([h_embs for _ in range(self.n_heads)], dim=1)
                cell_embs_1 = self.aggregate(h_embs, vector, self.agg_method)
                t_vectors_next_iter.append(cell_embs_1)
            else:
                h_broadcast_embs = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs[i], dim=1), 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                embs = torch.cat([torch.sum(t_embs[i], dim=1) for _ in range(self.n_heads)], dim=1)
                cell_embs_1 = self.aggregate(embs, vector, self.agg_method)
                t_vectors_next_iter.append(cell_embs_1)

        Nh_embs_list = t_vectors_next_iter
        self.cell_embs = self.sum_aggregator(Nh_embs_list)

        # Drug2 embedding learning process
        t_embs_drug2, r_embs_drug2 = self.get_neighbors_drug(u2)
        h_embs_drug2 = self.entity_embs(u2)
        drug2_t_vectors_next_iter = [h_embs_drug2]

        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs_drug2 = torch.cat(
                    [torch.unsqueeze(h_embs_drug2, 1) for _ in range(t_embs_drug2[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug2, r_embs_drug2[i], t_embs_drug2[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_drug2 = torch.cat([h_embs_drug2 for _ in range(self.n_heads)], dim=1)
                drug2_embs_1 = self.aggregate(h_embs_drug2, vector, self.agg_method)
                drug2_t_vectors_next_iter.append(drug2_embs_1)
            else:
                h_broadcast_embs_drug2 = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs_drug2[i], dim=1), 1) for _ in
                     range(t_embs_drug2[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug2, r_embs_drug2[i], t_embs_drug2[i + 1])
                vector = self.leakyRelu(vector)
                embs_d2 = torch.cat([torch.sum(t_embs_drug2[i], dim=1) for _ in range(self.n_heads)], dim=1)
                drug2_embs_1 = self.aggregate(embs_d2, vector, self.agg_method)
                drug2_t_vectors_next_iter.append(drug2_embs_1)

        Nh_embs_drug2_list = drug2_t_vectors_next_iter
        self.drug2_embs = self.sum_aggregator(Nh_embs_drug2_list)

        # Drug1 embedding learning process
        t_embs_drug1, r_embs_drug1 = self.get_neighbors_drug(u1)
        h_embs_drug1 = self.entity_embs(u1)
        drug1_t_vectors_next_iter = [h_embs_drug1]

        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs_drug1 = torch.cat(
                    [torch.unsqueeze(h_embs_drug1, 1) for _ in range(t_embs_drug1[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug1, r_embs_drug1[i], t_embs_drug1[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_drug1 = torch.cat([h_embs_drug1 for _ in range(self.n_heads)], dim=1)
                drug1_embs_1 = self.aggregate(h_embs_drug1, vector, self.agg_method)
                drug1_t_vectors_next_iter.append(drug1_embs_1)
            else:
                h_broadcast_embs_drug1 = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs_drug1[i], dim=1), 1) for _ in
                     range(t_embs_drug1[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug1, r_embs_drug1[i], t_embs_drug1[i + 1])
                vector = self.leakyRelu(vector)
                embs_d1 = torch.cat([torch.sum(t_embs_drug1[i], dim=1) for _ in range(self.n_heads)], dim=1)
                drug1_embs_1 = self.aggregate(embs_d1, vector, self.agg_method)
                drug1_t_vectors_next_iter.append(drug1_embs_1)

        Nh_embs_drug1_list = drug1_t_vectors_next_iter
        drug1_embs = self.sum_aggregator(Nh_embs_drug1_list)

        # Calculate combination prediction score
        combine_drug = torch.max(drug1_embs, self.drug2_embs)
        logits = torch.sigmoid((combine_drug * self.cell_embs).sum(dim=1))
        return logits


# ========== 元学习框架 ==========
class MAML:
    def __init__(self, model, tasks, args):
        self.model = model
        self.tasks = tasks
        self.args = args
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

        # 添加余弦退火调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.meta_optimizer,
            T_0=args.T_0,  # 初始周期
            T_mult=args.T_mult,  # 周期倍增因子
            eta_min=args.eta_min  # 最小学习率
        )

    def meta_train(self, data_loaders):
        """元训练过程"""
        print("开始元训练...")
        print(f"任务数据量分布: { {task: len(loader.dataset) for task, loader in data_loaders.items()} }")

        best_auc = 0

        for epoch in range(self.args.meta_epochs):
            total_meta_loss = 0
            task_count = 0
            epoch_auc_scores = []

            # 对每个任务进行训练
            for task_name, task_data in data_loaders.items():
                if task_name == 'pep_hla_generate_tcr':
                    # 跳过生成任务（需要特殊处理）
                    continue

                # 内循环梯度更新
                fast_weights = self.inner_update(task_data, task_name)

                # 外循环梯度更新
                meta_loss, task_auc = self.meta_update(task_data, task_name, fast_weights, epoch)
                total_meta_loss += meta_loss
                task_count += 1
                if task_auc > 0:
                    epoch_auc_scores.append(task_auc)

            if task_count > 0:
                avg_meta_loss = total_meta_loss / task_count
                avg_auc = np.mean(epoch_auc_scores) if epoch_auc_scores else 0

                # 更新学习率
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

                print(f"Epoch {epoch + 1}, Average Meta Loss: {avg_meta_loss:.4f}, "
                      f"Average AUC: {avg_auc:.4f}, LR: {current_lr:.6f}")

                # 基于AUC的早停逻辑
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), f'best_auc_model_{int(time.time())}.pt')

        # 加载最佳模型
        # self.model.load_state_dict(torch.load(f'best_auc_model_{int(time.time())}.pt'))
        print("训练完成，加载最佳模型")

    def inner_update(self, task_data, task_name):
        """内循环梯度更新"""
        # 复制模型参数
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # 内循环训练步骤（只使用少量批次）
        inner_steps = min(3, len(task_data))  # 限制内循环步数
        data_iter = iter(task_data)

        for step in range(inner_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            u1, u2, c, labels = batch
            u1, u2, c, labels = u1.to(self.args.device), u2.to(self.args.device), c.to(self.args.device), labels.to(
                self.args.device)

            # 前向传播
            logits = self.forward_with_weights(u1, u2, c, fast_weights, task_name)
            loss = F.binary_cross_entropy(logits, labels)

            # 计算梯度并更新fast_weights
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True, allow_unused=True)

            # 过滤掉None梯度（未使用的参数）
            valid_grads = []
            valid_weights = {}
            for (name, weight), grad in zip(fast_weights.items(), grads):
                if grad is not None:
                    valid_grads.append(grad)
                    valid_weights[name] = weight

            # 只更新有梯度的参数
            if valid_grads:
                updated_weights = {name: weight - self.args.inner_lr * grad
                                   for (name, weight), grad in zip(valid_weights.items(), valid_grads)}
                # 保持其他参数不变
                for name, weight in fast_weights.items():
                    if name not in updated_weights:
                        updated_weights[name] = weight
                fast_weights = updated_weights

        return fast_weights

    def meta_update(self, task_data, task_name, fast_weights, epoch):
        """外循环梯度更新"""
        meta_loss = 0
        batch_count = 0
        all_labels = []
        all_logits = []

        # 使用少量批次进行元更新
        meta_steps = min(2, len(task_data))
        data_iter = iter(task_data)

        for step in range(meta_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            u1, u2, c, labels = batch
            u1, u2, c, labels = u1.to(self.args.device), u2.to(self.args.device), c.to(self.args.device), labels.to(
                self.args.device)

            # 使用fast_weights进行前向传播
            logits = self.forward_with_weights(u1, u2, c, fast_weights, task_name)
            loss = F.binary_cross_entropy(logits, labels)
            meta_loss += loss
            batch_count += 1

            # 收集预测结果用于计算AUC
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.detach().cpu().numpy())

        # 平均损失并更新元参数
        if batch_count > 0:
            meta_loss = meta_loss / batch_count
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            # 计算AUC
            task_auc = 0
            if len(all_labels) > 0:
                try:
                    task_auc = roc_auc_score(all_labels, all_logits)
                except:
                    task_auc = 0

            return meta_loss.item(), task_auc
        else:
            return 0.0, 0

    def forward_with_weights(self, u1, u2, c, weights, task_name):
        """使用给定的权重进行前向传播"""
        # 根据任务类型调整输入映射
        if task_name == 'pep_hla_label':
            # hla作为drug1，固定drug2为0，pep作为cellline
            pass  # 输入顺序已经正确
        elif task_name == 'pep_tcr_label':
            # tcr作为drug1，固定drug2为0，pep作为cellline
            pass  # 输入顺序已经正确
        elif task_name == 'tcr_pep_hla_label':
            # tcr作为drug1，hla作为drug2，pep作为cellline
            pass  # 输入顺序已经正确
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        # 使用模型的标准前向传播
        return self.model(u1, u2, c)

    def meta_test(self, data_loaders, support_ratio=0.2, adaptation_steps=5):
        """元学习测试：在支持集上微调，在查询集上测试"""
        print("\n=== 元学习测试开始 ===")

        results = {}

        for task_name, task_data in data_loaders.items():
            if task_name == 'pep_hla_generate_tcr':
                continue

            print(f"\n测试任务: {task_name}")

            # 分割支持集和查询集
            dataset = task_data.dataset
            dataset_size = len(dataset)
            support_size = int(dataset_size * support_ratio)
            query_size = dataset_size - support_size

            # 随机分割
            indices = list(range(dataset_size))
            random.shuffle(indices)
            support_indices = indices[:support_size]
            query_indices = indices[support_size:]

            # 创建支持集和查询集
            support_dataset = torch.utils.data.Subset(dataset, support_indices)
            query_dataset = torch.utils.data.Subset(dataset, query_indices)

            support_loader = DataLoader(support_dataset, batch_size=self.args.batch_size, shuffle=True)
            query_loader = DataLoader(query_dataset, batch_size=self.args.batch_size, shuffle=False)

            # 在支持集上快速适应
            adapted_weights = self.fast_adaptation(support_loader, task_name, adaptation_steps)

            # 在查询集上评估
            task_results = self.evaluate_on_query_set(query_loader, task_name, adapted_weights)
            results[task_name] = task_results

            print(f"任务 {task_name} 结果: "
                  f"Precision: {task_results['precision']:.4f}, "
                  f"Recall: {task_results['recall']:.4f}, "
                  f"ACC: {task_results['accuracy']:.4f}, "
                  f"AUC: {task_results['auc']:.4f}, "
                  f"AUPR: {task_results['aupr']:.4f}, "
                  f"F1: {task_results['f1']:.4f}")

        return results

    def fast_adaptation(self, support_loader, task_name, adaptation_steps):
        """在支持集上进行快速适应"""
        # 复制模型参数
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # 快速适应步骤
        for step in range(adaptation_steps):
            total_loss = 0
            batch_count = 0

            for batch in support_loader:
                u1, u2, c, labels = batch
                u1, u2, c, labels = u1.to(self.args.device), u2.to(self.args.device), c.to(self.args.device), labels.to(
                    self.args.device)

                # 前向传播
                logits = self.forward_with_weights(u1, u2, c, fast_weights, task_name)
                loss = F.binary_cross_entropy(logits, labels)

                # 计算梯度并更新fast_weights
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=False, allow_unused=True)

                # 过滤掉None梯度
                valid_grads = []
                valid_weights = {}
                for (name, weight), grad in zip(fast_weights.items(), grads):
                    if grad is not None:
                        valid_grads.append(grad)
                        valid_weights[name] = weight

                # 只更新有梯度的参数
                if valid_grads:
                    updated_weights = {name: weight - self.args.inner_lr * grad
                                       for (name, weight), grad in zip(valid_weights.items(), valid_grads)}
                    # 保持其他参数不变
                    for name, weight in fast_weights.items():
                        if name not in updated_weights:
                            updated_weights[name] = weight
                    fast_weights = updated_weights

                total_loss += loss.item()
                batch_count += 1

        return fast_weights

    def evaluate_on_query_set(self, query_loader, task_name, adapted_weights):
        """在查询集上评估适应后的模型"""
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in query_loader:
                u1, u2, c, labels = batch
                u1, u2, c, labels = u1.to(self.args.device), u2.to(self.args.device), c.to(self.args.device), labels.to(
                    self.args.device)

                # 使用适应后的权重进行预测
                logits = self.forward_with_weights(u1, u2, c, adapted_weights, task_name)

                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        # 评估性能
        if len(all_labels) > 0:
            p, r, acc, auc_score, aupr, f1 = eval_classification(np.array(all_labels), np.array(all_logits))
            return {
                'precision': p,
                'recall': r,
                'accuracy': acc,
                'auc': auc_score,
                'aupr': aupr,
                'f1': f1
            }
        else:
            return {
                'precision': 0,
                'recall': 0,
                'accuracy': 0,
                'auc': 0,
                'aupr': 0,
                'f1': 0
            }


# ========== 主训练函数 ==========
def train_maml():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--meta_epochs', type=int, default=1500)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--n_neighbors', type=int, default=6)
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--r_dim', type=int, default=64)
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--l2_weight', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--meta_lr', type=float, default=0.01)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # 添加余弦退火参数
    parser.add_argument('--T_0', type=int, default=10, help='Cosine annealing initial period')
    parser.add_argument('--T_mult', type=int, default=2, help='Period multiplier')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate')

    args = parser.parse_args()

    # 设置随机种子
    seed = 55
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(seed)

    print(f"使用设备: {args.device}")

    # 加载真实数据
    print("加载数据...")
    all_data, task_datasets = load_real_data()

    # 构建知识图谱（使用所有数据）
    print("构建知识图谱...")
    all_data_frames = []
    for task_name, dataset in task_datasets.items():
        if dataset is not None:
            all_data_frames.append(dataset.data)

    if all_data_frames:
        all_data_df = pd.concat(all_data_frames, ignore_index=True)
    else:
        all_data_df = pd.DataFrame(columns=['Peptide', 'TCR', 'HLA_sequence', 'label'])

    kg_triples, entity_dict = build_kg_from_data(all_data_df)
    kg_indexes = getKgIndexsFromKgTriples(kg_triples)

    ## 构建邻接矩阵（现在会返回GPU张量）
    adj_entity, adj_relation = construct_adj(args.n_neighbors, kg_indexes, len(entity_dict), args.device)

    # 创建模型
    n_entities = len(entity_dict)
    n_relations = 3  # 三种关系

    model = KGANS(args, n_entities, n_relations, args.e_dim, args.r_dim,
                  adj_entity, adj_relation)
    model.to(args.device)

    # 准备任务数据加载器
    tasks = ['pep_hla_label', 'pep_tcr_label', 'tcr_pep_hla_label']
    data_loaders = {}

    for task in tasks:
        if task in task_datasets and task_datasets[task] is not None:
            dataset = task_datasets[task]

            # 更新数据集的实体字典为统一的字典
            dataset.entity_dict = entity_dict.copy()

            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            data_loaders[task] = data_loader
            print(f"任务 {task} 创建了数据加载器，包含 {len(dataset)} 个样本")
        else:
            print(f"警告: 任务 {task} 的数据集不存在或为空")

    # 创建MAML训练器
    maml = MAML(model, tasks, args)

    # 元训练
    maml.meta_train(data_loaders)

    # 使用元学习方式进行测试（支持集上微调，查询集上测试）
    print("\n=== 元学习评估开始 ===")
    test_results = maml.meta_test(data_loaders)

    # 打印总体结果
    print("\n=== 最终测试结果 ===")
    for task_name, results in test_results.items():
        print(f"{task_name}: "
              f"Precision: {results['precision']:.4f}, "
              f"Recall: {results['recall']:.4f}, "
              f"ACC: {results['accuracy']:.4f}, "
              f"AUC: {results['auc']:.4f}, "
              f"AUPR: {results['aupr']:.4f}, "
              f"F1: {results['f1']:.4f}")

    return model, test_results


# ========== 主函数调用部分 ==========
if __name__ == '__main__':
    # 训练模型
    model, results = train_maml()

    print("元学习训练和评估完成！")