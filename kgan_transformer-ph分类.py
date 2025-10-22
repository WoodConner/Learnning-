import os
import math
import collections
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, \
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


# ========== 评估函数 ==========
def evaluate_model(model, data_loader, device, save_plots=False):
    """评估模型性能"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for hla_ids, pep_ids, labels in data_loader:
            hla_ids = hla_ids.to(device)
            pep_ids = pep_ids.to(device)
            labels = labels.long().to(device)

            outputs = model(hla_ids, pep_ids)  # [B, 2]
            probs = F.softmax(outputs, dim=-1)[:, 1]  # 正类的概率
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算各项指标（对 AUC/AUPRC 做容错）
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = float('nan')
        print(f"[Warning] 计算 AUC 失败: {e}")

    try:
        auprc = average_precision_score(all_labels, all_probs)
    except Exception as e:
        auprc = float('nan')
        print(f"[Warning] 计算 AUPRC 失败: {e}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'auprc': auprc
    }

    # 绘制曲线
    if save_plots:
        plot_curves(all_labels, all_probs, save_path='evaluation_curves.png')

    return metrics, all_probs, all_labels


def plot_curves(true_labels, predicted_probs, save_path='evaluation_curves.png'):
    """绘制AUC和AUPRC曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC曲线（若可计算）
    try:
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = roc_auc_score(true_labels, predicted_probs)
        ax1.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    except Exception:
        ax1.text(0.5, 0.5, 'ROC cannot be computed', horizontalalignment='center', verticalalignment='center')

    ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall曲线
    try:
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
        auprc = average_precision_score(true_labels, predicted_probs)
        ax2.plot(recall, precision, lw=2, label=f'PR curve (AUPRC = {auprc:.4f})')
    except Exception:
        ax2.text(0.5, 0.5, 'PR cannot be computed', horizontalalignment='center', verticalalignment='center')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall (PR) Curve')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"评估曲线已保存至: {save_path}")


def print_metrics(metrics, phase="Validation"):
    """打印评估指标"""
    print(f"\n{phase} Metrics:")
    print("-" * 50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"AUPRC:       {metrics['auprc']:.4f}")
    print("-" * 50)


# ========== HLA等位基因名称到伪序列的映射 ==========
class HLAMapper:
    def __init__(self):
        self.hla_name_to_sequence = {}
        self.sequence_to_hla_name = defaultdict(list)

    def load_hla_mapping(self, mapping_file):
        """加载HLA等位基因名称到伪序列的映射"""
        # 添加文件存在性检查
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"HLA映射文件不存在: {mapping_file}")
        
        print(f"正在加载HLA映射文件: {mapping_file}")
        
        # 使用更稳健的方式读取CSV
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(mapping_file, encoding=encoding)
                    print(f"使用编码 {encoding} 成功读取文件")
                    break
                except UnicodeDecodeError:
                    print(f"编码 {encoding} 失败，尝试下一种...")
                    continue
            
            # 如果所有编码都失败，尝试使用错误处理
            if df is None:
                df = pd.read_csv(mapping_file, encoding='utf-8', errors='replace')
                print("使用错误处理模式读取文件")
                
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            # 尝试使用Python内置方式读取
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 解析CSV内容
                if len(lines) > 0:
                    headers = lines[0].strip().split(',')
                    data = []
                    for line in lines[1:]:
                        values = line.strip().split(',')
                        data.append(values)
                    
                    df = pd.DataFrame(data, columns=headers)
                    print("使用Python内置方式成功读取文件")
            except Exception as e2:
                raise ValueError(f"无法读取文件 {mapping_file}: {e2}")
        
        # 检查必要的列是否存在
        required_cols = ['HLA', 'HLA_sequence']
        if not all(col in df.columns for col in required_cols):
            available_cols = list(df.columns)
            raise ValueError(f"文件 {mapping_file} 缺少必要的列。需要: {required_cols}，实际有: {available_cols}")
        
        for _, row in df.iterrows():
            hla_name = row['HLA']
            hla_sequence = row['HLA_sequence']
            self.hla_name_to_sequence[hla_name] = hla_sequence

        # 构建反向映射
        for hla_name, hla_sequence in self.hla_name_to_sequence.items():
            self.sequence_to_hla_name[hla_sequence].append(hla_name)
            
        print(f"成功加载 {len(self.hla_name_to_sequence)} 个HLA映射")

    def get_sequence(self, hla_name):
        """根据HLA等位基因名称获取伪序列"""
        return self.hla_name_to_sequence.get(hla_name)


# ========== 数据加载和知识图谱构建 ==========
class HLApepDataset(Dataset):
    def __init__(self, file_path, entity_dict, hla_mapper):
        self.entity_dict = entity_dict
        self.hla_mapper = hla_mapper
        self.data = self.load_data_from_file(file_path)
        self.entity_id_to_sequence = {}
        self.kg_triples = self.build_kg_triples()

    def load_data_from_file(self, file_path):
        """从CSV文件加载数据"""
        # 添加文件存在性检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
        print(f"正在加载数据文件: {file_path}")
        
        # 使用更稳健的方式读取CSV
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"使用编码 {encoding} 成功读取文件")
                    break
                except UnicodeDecodeError:
                    print(f"编码 {encoding} 失败，尝试下一种...")
                    continue
            
            # 如果所有编码都失败，尝试使用错误处理
            if df is None:
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
                print("使用错误处理模式读取文件")
                
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            # 尝试使用Python内置方式读取
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 解析CSV内容
                if len(lines) > 0:
                    headers = lines[0].strip().split(',')
                    data = []
                    for line in lines[1:]:
                        values = line.strip().split(',')
                        data.append(values)
                    
                    df = pd.DataFrame(data, columns=headers)
                    print("使用Python内置方式成功读取文件")
            except Exception as e2:
                raise ValueError(f"无法读取文件 {file_path}: {e2}")
        
        print(f"从 {file_path} 加载了 {len(df)} 条数据")

        required_cols = ['HLA', 'Peptide', 'label', 'HLA_sequence']
        if not all(col in df.columns for col in required_cols):
            available_cols = list(df.columns)
            raise ValueError(f"文件 {file_path} 缺少必要的列。需要: {required_cols}，实际有: {available_cols}")

        # 统计序列唯一数
        unique_hla_names = df['HLA'].nunique()
        unique_peptides = df['Peptide'].nunique()
        unique_hla_sequences = df['HLA_sequence'].nunique()

        print(f"唯一HLA等位基因名称: {unique_hla_names}")
        print(f"唯一肽段序列: {unique_peptides}")
        print(f"唯一HLA伪序列: {unique_hla_sequences}")

        return df

    def sequence_to_entity_id(self, sequence, seq_type):
        """将序列转换为实体ID，相同序列使用相同ID"""
        entity_key = f"{seq_type}_{sequence}"

        if entity_key not in self.entity_dict:
            entity_id = len(self.entity_dict)
            self.entity_dict[entity_key] = entity_id
            self.entity_id_to_sequence[entity_id] = {
                'type': seq_type,
                'sequence': sequence,
                'key': entity_key
            }

        return self.entity_dict[entity_key]

    def build_kg_triples(self):
        """构建知识图谱三元组"""
        kg_triples = []
        triple_set = set()

        for _, row in self.data.iterrows():
            pep = row['Peptide']
            hla_seq = row['HLA_sequence']
            label = row['label']

            pep_id = self.sequence_to_entity_id(pep, 'pep')
            hla_id = self.sequence_to_entity_id(hla_seq, 'hla')

            triple_key = (pep_id, 1 if label == 1 else 0, hla_id)
            if triple_key not in triple_set:
                triple_set.add(triple_key)
                kg_triples.append([pep_id, 1 if label == 1 else 0, hla_id])

        print(f"构建了 {len(kg_triples)} 个唯一三元组")
        return kg_triples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        pep = sample['Peptide']
        hla_seq = sample['HLA_sequence']
        label = sample['label']

        pep_id = self.sequence_to_entity_id(pep, 'pep')
        hla_id = self.sequence_to_entity_id(hla_seq, 'hla')

        return torch.tensor(hla_id, dtype=torch.long), torch.tensor(pep_id, dtype=torch.long), torch.tensor(int(label), dtype=torch.long)


# ========== 通用工具 ==========
def getKgIndexesFromKgTriples(kg_triples):
    """构建知识图谱索引"""
    kg_indexes = collections.defaultdict(list)
    for h, r, t in kg_triples:
        kg_indexes[str(h)].append([int(t), int(r)])
        kg_indexes[str(t)].append([int(h), int(r)])  # 无向图
    return kg_indexes


def construct_adj(neighbor_sample_size, kg_indexes, entity_num, device):
    """构建邻接矩阵"""
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
        else:
            # 如果没有邻居，用自身填充
            adj_entity[entity] = np.array([entity] * neighbor_sample_size)
            adj_relation[entity] = np.array([0] * neighbor_sample_size)

    adj_entity = torch.LongTensor(adj_entity).to(device)
    adj_relation = torch.LongTensor(adj_relation).to(device)

    print(f"实体邻接列表形状: {adj_entity.shape}")
    print(f"关系邻接列表形状: {adj_relation.shape}")

    return adj_entity, adj_relation


# ========== KGAN 编码器 ==========
class KGANEncoder(nn.Module):
    def __init__(self, args, n_entities, n_relations, e_dim, r_dim, adj_entity, adj_relation,
                 agg_method='Bi-Interaction'):
        super(KGANEncoder, self).__init__()

        self.device = args.device  # 保存设备信息
        self.entity_embs = nn.Embedding(n_entities, e_dim, max_norm=1).to(args.device)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1).to(args.device)
        self.dropout = args.dropout
        self.n_iter = args.n_iter
        self.dim = e_dim
        self.n_heads = args.n_heads
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        # 修复：不要在 forward 中创建线性层 -> 在 __init__ 中创建 ModuleList
        self.Wx = nn.ModuleList([nn.Linear(self.dim, self.dim).to(args.device) for _ in range(self.n_heads)])

        # 将 WW 与 e_dim 一致（避免硬编码）
        self.WW = nn.Linear(self.dim, self.dim, bias=False).to(args.device)

        # 统一 agg 标识
        self.agg_method = agg_method
        self.agg = agg_method

        # 如果 relation dim != entity dim，则投影 relation 到 e_dim（保持注意力 concat 处维度一致）
        if r_dim != e_dim:
            self.rel2ent = nn.Linear(r_dim, e_dim, bias=False).to(args.device)
        else:
            self.rel2ent = None

        # attention 网络：注意力的输入为 [h_emb || r_emb] -> 维度应为 e_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(e_dim * 2, e_dim, bias=False).to(args.device),
            nn.ReLU(),
            nn.Linear(e_dim, e_dim, bias=False).to(args.device),
            nn.ReLU(),
            nn.Linear(e_dim, 1, bias=False).to(args.device),
            nn.Sigmoid(),
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        if agg_method == 'concat':
            self.W_concat = nn.Linear(e_dim * 2, e_dim).to(args.device)
        else:
            self.W1 = nn.Linear(e_dim * self.n_heads, e_dim * 2).to(args.device)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(e_dim * self.n_heads, e_dim * 2).to(args.device)

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)

        # 初始化 attention 中的线性层
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # 初始化 Wx 列表
        for wx in self.Wx:
            nn.init.xavier_uniform_(wx.weight)
            if wx.bias is not None:
                nn.init.constant_(wx.bias, 0)

        # 初始化 WW 与 rel2ent（如存在）
        nn.init.xavier_uniform_(self.WW.weight)
        if hasattr(self, 'rel2ent') and self.rel2ent is not None:
            nn.init.xavier_uniform_(self.rel2ent.weight)

        # 其他可能存在的线性层也初始化
        for m in [getattr(self, 'W1', None), getattr(self, 'W2', None), getattr(self, 'W_concat', None)]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_neighbors(self, entities):
        """统一的邻居获取函数"""
        entities = entities.view((-1, 1))
        entity_list = [entities]
        relation_list = []

        for h in range(self.n_iter):
            neighbor_entities = self.adj_entity[entity_list[h]].view((entity_list[h].shape[0], -1))
            neighbor_relations = self.adj_relation[entity_list[h]].view((entity_list[h].shape[0], -1))
            entity_list.append(neighbor_entities)
            relation_list.append(neighbor_relations)

        neighbor_entities_embs = [self.entity_embs(entity) for entity in entity_list]
        neighbor_relations_embs = [self.relation_embs(relation) for relation in relation_list]

        # 如果需要，把 relation_emb 投影到 e_dim
        if self.rel2ent is not None:
            neighbor_relations_embs = [self.rel2ent(r) for r in neighbor_relations_embs]

        return neighbor_entities_embs, neighbor_relations_embs

    def sum_aggregator(self, embs):
        e_u = embs[0]
        # 使用 self.agg（已与 agg_method 统一）
        if self.agg == 'concat':
            for i in range(1, len(embs)):
                e_u = torch.cat((embs[i], e_u), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(embs)):
                e_u += self.WW(embs[i])
        else:
            # 如果没有匹配则默认返回第一个
            pass

        return e_u

    def GATMessagePass(self, h_embs, r_embs, t_embs):
        """图注意力消息传递
        期望输入:
          - h_embs: [B, num_neighbors, e_dim]
          - r_embs: [B, num_neighbors, e_dim]  (已投影)
          - t_embs: [B, num_neighbors, e_dim]
        """
        multi = []

        for i in range(self.n_heads):
            # 注意力权重（针对每个邻居位置）
            att_weights = self.attention(torch.cat((h_embs, r_embs), dim=-1)).squeeze(-1)  # [B, num_neighbors]
            att_weights_norm = F.softmax(att_weights, dim=-1)
            emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_embs)  # [B, num_neighbors, e_dim]

            # 使用在 __init__ 中注册的线性层（已初始化）
            emb_i = self.Wx[i](emb_i.sum(dim=1))  # [B, e_dim]
            multi.append(emb_i)

        all_emb_i = torch.cat(multi, dim=-1)  # [B, e_dim * n_heads]
        return all_emb_i

    def aggregate(self, h_embs, Nh_embs, agg_method='Bi-Interaction'):
        """聚合头实体和邻居信息"""
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) \
                + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:  # sum
            return self.leakyRelu(self.W1(h_embs + Nh_embs))

    def forward(self, entity_idx):
        """对单个实体进行编码"""
        t_embs, r_embs = self.get_neighbors(entity_idx)
        h_embs = self.entity_embs(entity_idx)  # [batch_size, e_dim]

        t_vectors_next_iter = [h_embs]

        for i in range(self.n_iter):
            if i == 0:
                # 第一次迭代：使用 head 实体的广播
                h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_expanded = torch.cat([h_embs for _ in range(self.n_heads)], dim=1)
                entity_emb = self.aggregate(h_embs_expanded, vector, self.agg_method)
                t_vectors_next_iter.append(entity_emb)
            else:
                # 后续迭代：使用邻居聚合的表示
                h_broadcast_embs = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs[i], dim=1), 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                embs = torch.cat([torch.sum(t_embs[i], dim=1) for _ in range(self.n_heads)], dim=1)
                entity_emb = self.aggregate(embs, vector, self.agg_method)
                t_vectors_next_iter.append(entity_emb)

        Nh_embs_list = t_vectors_next_iter

        # 最终聚合所有层的嵌入
        final_entity_emb = self.sum_aggregator(Nh_embs_list)
        return final_entity_emb

    def encode_pair(self, h_idx, t_idx):
        """编码实体对"""
        h_emb = self.forward(h_idx)  # [batch_size, e_dim*(n_iters? depends on agg)]
        t_emb = self.forward(t_idx)
        # 拼接两个实体的嵌入
        pair_embedding = torch.cat([h_emb, t_emb], dim=-1)
        return pair_embedding


# ---------------- PositionalEncoding（保持） ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# -------------- Causal Multi-Head Self-Attention --------------
class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal (autoregressive) multi-head self-attention with batch_first ([B, T, d_model]).
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B, T, d_model]
        B, T, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        device = x.device
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        return out, attn_weights


# -------------- Multi-Head Cross-Attention --------------
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, q_input, memory, key_padding_mask=None):
        # q_input: [B, T, d_model], memory: [B, M, d_model] or [M, d_model]
        B, T, _ = q_input.size()
        if memory is None:
            # no memory -> return zeros and None weights (cross-attention skipped)
            return torch.zeros_like(q_input), None

        if memory.dim() == 2:
            memory = memory.unsqueeze(0).expand(B, -1, -1)
        M = memory.size(1)

        q = self.q_proj(q_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out, attn


# -------------- Decoder Layer using above attention blocks --------------
class DecoderLayerNew(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = CausalMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, memory_key_padding_mask=None):
        # 1) causal self-attention
        sa_out, sa_weights = self.self_attn(x)
        x = x + self.dropout(sa_out)
        x = self.norm1(x)

        # 2) cross-attention (if memory provided)
        if memory is not None:
            ca_out, ca_weights = self.cross_attn(x, memory, key_padding_mask=memory_key_padding_mask)
            x = x + self.dropout(ca_out)
            x = self.norm2(x)
        else:
            ca_weights = None

        # 3) FFN
        ff_out = self.ffn(x)
        x = x + ff_out
        x = self.norm3(x)

        return x, sa_weights, ca_weights


# -------------- Full Transformer Decoder (stack of layers) --------------
class TransformerDecoderNew(nn.Module):
    """
    Decoder stack that receives a memory vector (projected to d_model) and returns classification logits.
    Keep vocab_size arg for compatibility (not used for now).
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, vocab_size,
                 dropout=0.1, max_len=100, cls_num_classes=2, dim_feedforward=512):
        super().__init__()
        assert d_model % nhead == 0
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)  # 保留但暂时未使用
        self.pos = PositionalEncoding(d_model, max_len)
        self.mem_proj = nn.Linear(input_dim, d_model)  # project mem->d_model
        self.layers = nn.ModuleList([DecoderLayerNew(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, cls_num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x: [B, input_dim]
        x_proj = self.mem_proj(x).unsqueeze(1)  # [B, 1, d_model]
        encoded = self.pos(x_proj)

        # 逐层处理（cross-attention 的 memory 参数都为 None）
        for layer in self.layers:
            encoded, _, _ = layer(encoded, memory=None)

        output = self.classifier(encoded.squeeze(1))  # [B, cls_num_classes]
        return output


# ========== 完整模型 ==========
class HLApepClassifier(nn.Module):
    def __init__(self, kgan_encoder, transformer_decoder, device):
        super().__init__()
        self.kgan_encoder = kgan_encoder
        self.transformer_decoder = transformer_decoder.to(device)
        self.device = device

    def forward(self, hla_ids, pep_ids):
        # 使用KGAN编码器编码HLA-peptide对
        pair_embeddings = self.kgan_encoder.encode_pair(hla_ids, pep_ids)  # [B, input_dim]
        # 使用Transformer解码器进行分类
        logits = self.transformer_decoder(pair_embeddings)
        return logits


# ========== 预测接口 ==========
class HLApepPredictor:
    def __init__(self, model, entity_dict, entity_id_to_sequence, hla_mapper, device):
        self.model = model
        self.entity_dict = entity_dict
        self.entity_id_to_sequence = entity_id_to_sequence
        self.hla_mapper = hla_mapper
        self.device = device
        self.model.eval()

        # 构建序列到ID的映射
        self.sequence_to_id = {}
        for entity_id, info in entity_id_to_sequence.items():
            self.sequence_to_id[info['sequence']] = entity_id

    def predict(self, hla_name, peptide):
        """预测HLA-peptide是否结合"""
        hla_sequence = self.hla_mapper.get_sequence(hla_name)
        if hla_sequence is None:
            raise ValueError(f"未知的HLA等位基因: {hla_name}")

        hla_id = self.sequence_to_id.get(hla_sequence)
        pep_id = self.sequence_to_id.get(peptide)

        if hla_id is None or pep_id is None:
            return 0.5

        hla_tensor = torch.tensor([hla_id], device=self.device)
        pep_tensor = torch.tensor([pep_id], device=self.device)

        with torch.no_grad():
            logits = self.model(hla_tensor, pep_tensor)
            prob = F.softmax(logits, dim=-1)
            binding_prob = prob[0, 1].item()

        return binding_prob


# ========== 训练函数 ==========
def train_model(args, model, train_loader, val_loader, entity_dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0
    train_losses = []
    val_metrics_history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for hla_ids, pep_ids, labels in train_loader:
            hla_ids = hla_ids.to(args.device)
            pep_ids = pep_ids.to(args.device)
            labels = labels.long().to(args.device)

            optimizer.zero_grad()
            outputs = model(hla_ids, pep_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        val_metrics, _, _ = evaluate_model(model, val_loader, args.device)
        val_metrics_history.append(val_metrics)

        scheduler.step(val_metrics['accuracy'])

        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

    plot_training_history(train_losses, val_metrics_history)
    return model


def plot_training_history(train_losses, val_metrics_history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    epochs = range(1, len(val_metrics_history) + 1)
    accuracies = [m['accuracy'] for m in val_metrics_history]
    ax2.plot(epochs, accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# ========== 主函数 ==========
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--neighbor_sample_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--r_dim', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--data_file', type=str, default='C:\\Users\\admin\\OneDrive\\桌面\\科研\\with_neg\\peptide_hla_random1x.csv')
    args = parser.parse_args()

    print(f"使用设备: {args.device}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据文件路径: {os.path.abspath(args.data_file)}")

    # 初始化HLA映射器
    hla_mapper = HLAMapper()
    
    # 检查文件是否存在
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件 '{args.data_file}' 不存在!")
        print("请确保文件存在于当前目录，或使用 --data_file 参数指定正确的文件路径")
        
        # 尝试在当前目录查找可能的CSV文件
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"当前目录下的CSV文件: {csv_files}")
            if len(csv_files) == 1:
                print(f"尝试使用文件: {csv_files[0]}")
                args.data_file = csv_files[0]
            else:
                return
        else:
            return
    
    try:
        hla_mapper.load_hla_mapping(args.data_file)
    except Exception as e:
        print(f"加载HLA映射失败: {e}")
        return

    # 加载数据和构建知识图谱
    entity_dict = {}
    try:
        dataset = HLApepDataset(args.data_file, entity_dict, hla_mapper)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    n_entities = len(entity_dict)
    n_relations = 2

    # 构建邻接矩阵
    kg_indexes = getKgIndexesFromKgTriples(dataset.kg_triples)
    adj_entity, adj_relation = construct_adj(
        args.neighbor_sample_size, kg_indexes, n_entities, args.device
    )

    # 初始化模型 - 确保所有组件都在正确的设备上
    kgan_encoder = KGANEncoder(args, n_entities, n_relations, args.e_dim, args.r_dim,
                               adj_entity, adj_relation)

    # 计算KGAN编码器的输出维度（使用 dummy 测试，稳健）
    dummy_hla = torch.zeros(1, dtype=torch.long).to(args.device)
    dummy_pep = torch.zeros(1, dtype=torch.long).to(args.device)

    with torch.no_grad():
        dummy_pair_embedding = kgan_encoder.encode_pair(dummy_hla, dummy_pep)
        actual_kgan_output_dim = dummy_pair_embedding.shape[-1]

    print(f"KGAN编码器输出维度: {actual_kgan_output_dim}")

    # 注意：传入 vocab_size 占位参数（类中 embedding 未用到，但 __init__ 要求）
    transformer_decoder = TransformerDecoderNew(
        input_dim=actual_kgan_output_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        vocab_size=1,
        cls_num_classes=2
    )

    model = HLApepClassifier(kgan_encoder, transformer_decoder, args.device).to(args.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    # 训练模型
    trained_model = train_model(args, model, train_loader, val_loader, entity_dict)

    # 最终测试集评估
    print("\n" + "=" * 60)
    print("最终测试集评估结果")
    print("=" * 60)

    # 加载最佳模型（检查文件存在）
    if os.path.exists('best_model.pth'):
        trained_model.load_state_dict(torch.load('best_model.pth', map_location=args.device))
    else:
        print("[Warning] best_model.pth 不存在，使用当前训练模型进行评估。")

    test_metrics, test_probs, test_labels = evaluate_model(
        trained_model, test_loader, args.device, save_plots=True
    )
    print_metrics(test_metrics, "Test")

if __name__ == "__main__":
    main()