import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random, os, math
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, \
    f1_score, precision_recall_curve, roc_curve
import argparse
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import torch.cuda as cuda
import pickle
import hashlib


# ========== 数据加载和知识图谱构建 ==========
class ProcessData(Dataset):
    def __init__(self, file_path, task_type, entity_dict, max_length=20):
        self.task_type = task_type
        self.entity_dict = entity_dict
        self.max_length = max_length
        self.data = self.load_data_from_file(file_path)

        # 只存储真实的序列，不存储虚拟序列
        self.real_entity_sequences = {}
        self.kg_triples = self.build_kg_triples()

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
            # 使用有效的氨基酸序列作为虚拟TCR
            df['TCR'] = [f"AVT{''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=7))}" for i in range(len(df))]

        elif self.task_type == 'pep_tcr_label':
            # TCR, Peptide, label
            required_cols = ['TCR', 'Peptide', 'label']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"文件 {file_path} 缺少必要的列: {required_cols}")

            # 为pep-tcr任务生成对应的hla（但此任务不使用）
            # 使用有效的HLA序列作为虚拟HLA
            df['HLA'] = [f"HLA_{i}" for i in range(len(df))]
            df['HLA_sequence'] = [f"GSHSMRYF{''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=10))}" for i in
                                  range(len(df))]

        elif self.task_type == 'tcr_pep_hla_label':
            # Peptide, HLA, TCR, label, HLA_sequence
            required_cols = ['Peptide', 'HLA', 'TCR', 'label', 'HLA_sequence']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"文件 {file_path} 缺少必要的列: {required_cols}")

        return df

    def sequence_to_entity_id(self, sequence, seq_type, is_virtual=False):
        """将序列转换为实体ID，只存储真实序列"""
        entity_key = f"{seq_type}_{sequence}"

        if entity_key not in self.entity_dict:
            entity_id = len(self.entity_dict)
            self.entity_dict[entity_key] = entity_id
            # 只存储真实序列，不存储虚拟序列
            if not is_virtual:
                self.real_entity_sequences[entity_id] = sequence

        return self.entity_dict[entity_key]

    def build_kg_triples(self):
        """构建知识图谱三元组，区分真实和虚拟序列"""
        kg_triples = []

        for _, row in self.data.iterrows():
            pep = row['Peptide']
            tcr = row['TCR']
            hla = row['HLA_sequence']
            label = row['label']  # 0表示不结合，1表示结合

            # 根据任务类型判断是否为虚拟序列
            if self.task_type == 'pep_hla_label':
                # TCR是虚拟的
                pep_id = self.sequence_to_entity_id(pep, 'pep', is_virtual=False)
                tcr_id = self.sequence_to_entity_id(tcr, 'tcr', is_virtual=True)
                hla_id = self.sequence_to_entity_id(hla, 'hla', is_virtual=False)
            elif self.task_type == 'pep_tcr_label':
                # HLA是虚拟的
                pep_id = self.sequence_to_entity_id(pep, 'pep', is_virtual=False)
                tcr_id = self.sequence_to_entity_id(tcr, 'tcr', is_virtual=False)
                hla_id = self.sequence_to_entity_id(hla, 'hla', is_virtual=True)
            else:  # tcr_pep_hla_label
                # 都是真实的
                pep_id = self.sequence_to_entity_id(pep, 'pep', is_virtual=False)
                tcr_id = self.sequence_to_entity_id(tcr, 'tcr', is_virtual=False)
                hla_id = self.sequence_to_entity_id(hla, 'hla', is_virtual=False)

            # 添加三元组关系
            if label == 1:  # 结合
                kg_triples.append([pep_id, 0, tcr_id])  # pep-interact-tcr
                kg_triples.append([pep_id, 1, hla_id])  # pep-bind-hla
                kg_triples.append([tcr_id, 2, hla_id])  # tcr-recognize-hla
            else:  # 不结合
                kg_triples.append([pep_id, 3, tcr_id])  # pep-not-interact-tcr
                kg_triples.append([pep_id, 4, hla_id])  # pep-not-bind-hla
                kg_triples.append([tcr_id, 5, hla_id])  # tcr-not-recognize-hla

        return kg_triples

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


def load_data_and_build_kg(args):
    """加载数据并构建知识图谱，只收集真实序列"""
    print("加载数据并构建知识图谱...")

    file_paths = {
        'pep_hla_label': 'with_neg/pep_hla_random1x.csv',
        'pep_tcr_label': 'with_neg/pep_tcr_random1x.csv',
        'tcr_pep_hla_label': 'with_neg/trimer_random1x.csv'
    }

    entity_dict = {}
    all_real_entity_sequences = {}  # 只收集真实序列

    all_datasets = {}
    all_kg_triples = []

    # 用于统计真实序列
    real_sequences_stats = {
        'Peptide': set(),
        'TCR': set(),
        'HLA': set()
    }

    for task_name, file_path in file_paths.items():
        try:
            dataset = ProcessData(file_path, task_name, entity_dict)
            all_datasets[task_name] = dataset
            all_kg_triples.extend(dataset.kg_triples)

            # 只收集真实序列
            all_real_entity_sequences.update(dataset.real_entity_sequences)

            # 统计真实序列
            df = dataset.data
            if task_name == 'pep_hla_label':
                real_sequences_stats['Peptide'].update(df['Peptide'].unique())
                real_sequences_stats['HLA'].update(df['HLA_sequence'].unique())
                # TCR是虚拟的，不统计
            elif task_name == 'pep_tcr_label':
                real_sequences_stats['Peptide'].update(df['Peptide'].unique())
                real_sequences_stats['TCR'].update(df['TCR'].unique())
                # HLA是虚拟的，不统计
            else:  # tcr_pep_hla_label
                real_sequences_stats['Peptide'].update(df['Peptide'].unique())
                real_sequences_stats['TCR'].update(df['TCR'].unique())
                real_sequences_stats['HLA'].update(df['HLA_sequence'].unique())

        except Exception as e:
            print(f"创建 {task_name} 数据集失败: {e}")
            all_datasets[task_name] = None

    # 计算总真实序列数（去重后）
    total_real_peptides = len(real_sequences_stats['Peptide'])
    total_real_tcrs = len(real_sequences_stats['TCR'])
    total_real_hlas = len(real_sequences_stats['HLA'])
    total_real_sequences = total_real_peptides + total_real_tcrs + total_real_hlas

    n_entitys = len(entity_dict)
    n_relations = 6

    print(f"总实体数 (包含虚拟): {n_entitys}")
    print(f"总真实序列数: {total_real_sequences}")
    print(f"真实肽段数: {total_real_peptides}")
    print(f"真实TCR数: {total_real_tcrs}")
    print(f"真实HLA数: {total_real_hlas}")
    print(f"关系数: {n_relations}")
    print(f"三元组数: {len(all_kg_triples)}")

    # 构建邻接矩阵
    print("构建邻接矩阵...")
    kg_indexs = getKgIndexsFromKgTriples(all_kg_triples)
    adj_entity, adj_relation = construct_adj(
        args.neighbor_sample_size, kg_indexs, n_entitys, args.device
    )

    vocab_size = n_entitys

    return n_entitys, n_relations, adj_entity, adj_relation, vocab_size, all_datasets, all_real_entity_sequences


# ========== KGAN 编码器 ==========
class KGANEncoder(nn.Module):
    def __init__(self, args, n_entitys, n_relations, e_dim, r_dim, adj_entity, adj_relation,
                 agg_method='Bi-Interaction'):
        super(KGANEncoder, self).__init__()

        self.entity_embs = nn.Embedding(n_entitys, e_dim, max_norm=1)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1)
        self.dropout = args.dropout
        self.n_iter = args.n_iter
        self.dim = e_dim
        self.n_heads = args.n_heads
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # 初始化权重
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)

        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

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
        return neighbor_entities_embs, neighbor_relations_embs

    def sum_aggregator(self, embs):
        e_u = embs[0]
        if self.agg == 'concat':
            for i in range(1, len(embs)):
                e_u = torch.cat((embs[i], e_u), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(embs)):
                e_u += embs[i]
        return e_u

    def GATMessagePass(self, h_embs, r_embs, t_embs):
        multi = []
        for i in range(self.n_heads):
            att_weights = self.attention(torch.cat((h_embs, r_embs), dim=-1)).squeeze(-1)
            att_weights_norm = F.softmax(att_weights, dim=-1)
            emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_embs)

            Wx = nn.Linear(self.dim, self.dim).to(h_embs.device)
            emb_i = Wx(emb_i.sum(dim=1))
            multi.append(emb_i)
        all_emb_i = torch.cat(multi, dim=-1)
        return all_emb_i

    def aggregate(self, h_embs, Nh_embs, agg_method='Bi-Interaction'):
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:
            return self.leakyRelu(self.W1(h_embs + Nh_embs))

    def forward(self, entity_idx):
        """对单个实体进行编码"""
        t_embs, r_embs = self.get_neighbors(entity_idx)
        h_embs = self.entity_embs(entity_idx)
        t_vectors_next_iter = [h_embs]

        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_expanded = torch.cat([h_embs for _ in range(self.n_heads)], dim=1)
                entity_emb = self.aggregate(h_embs_expanded, vector, self.agg_method)
                t_vectors_next_iter.append(entity_emb)
            else:
                h_broadcast_embs = torch.cat(
                    [torch.unsqueeze(torch.sum(t_embs[i], dim=1), 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                embs = torch.cat([torch.sum(t_embs[i], dim=1) for _ in range(self.n_heads)], dim=1)
                entity_emb = self.aggregate(embs, vector, self.agg_method)
                t_vectors_next_iter.append(entity_emb)

        Nh_embs_list = t_vectors_next_iter
        final_entity_emb = self.sum_aggregator(Nh_embs_list)
        return final_entity_emb

    def encode_pair(self, h_idx, t_idx):
        """编码实体对"""
        h_emb = self.forward(h_idx)
        t_emb = self.forward(t_idx)
        return torch.cat([h_emb, t_emb], dim=-1)


# ========== Transformer 解码器 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, vocab_size, dropout=0.1, max_len=100):
        super().__init__()
        self.d_model, self.vocab_size, self.max_len = d_model, vocab_size, max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def generate_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def forward(self, mem, tgt=None, max_len=None, teacher_ratio=0.5):
        B = mem.size(0)
        max_len = max_len or self.max_len
        mem_proj = self.proj(mem).unsqueeze(1)

        if tgt is not None and random.random() < teacher_ratio:
            tgt_input = tgt[:, :-1]
            tgt_emb = self.emb(tgt_input) * math.sqrt(self.d_model)
            tgt_emb = self.pos(self.drop(tgt_emb))
            tgt_mask = self.generate_mask(tgt_input.size(1)).to(mem.device)
            out = self.decoder(tgt=tgt_emb, memory=mem_proj, tgt_mask=tgt_mask)
            return self.out(out)
        else:
            tokens = torch.full((B, 1), 0, dtype=torch.long, device=mem.device)
            logits = []
            for i in range(max_len):
                tgt_emb = self.emb(tokens) * math.sqrt(self.d_model)
                tgt_emb = self.pos(self.drop(tgt_emb))
                tgt_mask = self.generate_mask(tokens.size(1)).to(mem.device)
                out = self.decoder(tgt=tgt_emb, memory=mem_proj, tgt_mask=tgt_mask)
                next_logit = self.out(out[:, -1, :])
                logits.append(next_logit.unsqueeze(1))
                next_token = next_logit.argmax(-1).unsqueeze(1)
                tokens = torch.cat([tokens, next_token], dim=1)
                if (next_token == 1).all():
                    break
            return torch.cat(logits, dim=1)


# ========== 统一的多任务模型 ==========
class UnifiedModel(nn.Module):
    def __init__(self, args, n_entitys, n_relations, e_dim, r_dim, adj_entity, adj_relation,
                 vocab_size, max_len=30, device='cpu'):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim_unifier = None

        # KGAN编码器
        self.kgan_encoder = KGANEncoder(
            args, n_entitys, n_relations, e_dim, r_dim, adj_entity, adj_relation
        )

        # Transformer解码器
        self.decoder = TransformerDecoder(
            input_dim=256,
            d_model=128,
            nhead=4,
            num_layers=2,
            vocab_size=vocab_size,
            max_len=max_len
        )

        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(vocab_size, 64),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # 任务嵌入
        self.task_embedding = nn.Embedding(4, e_dim)  # 0:hla_pep, 1:tcr_pep, 2:triple, 3:generation

    def _build_dim_unifier(self, in_features):
        """只在第一次调用时创建 Linear，并注册为模块。"""
        if self.dim_unifier is None or self.dim_unifier.in_features != in_features:
            self.dim_unifier = nn.Linear(in_features, 256).to(self.device)
            # 重新注册参数，保证 state_dict 能追踪
            self.add_module('dim_unifier', self.dim_unifier)

    def encode_task_inputs(self, hla_idx=None, tcr_idx=None, pep_idx=None, task_id=0):
        """编码任务输入"""
        embeddings = []
        with torch.no_grad():
            if hla_idx is not None:
                embeddings.append(self.kgan_encoder(hla_idx))
            if tcr_idx is not None:
                embeddings.append(self.kgan_encoder(tcr_idx))
            if pep_idx is not None:
                embeddings.append(self.kgan_encoder(pep_idx))
        if len(embeddings) == 0:
            raise ValueError("至少需要提供一个实体索引")
        combined_emb = torch.cat(embeddings, dim=-1)
        self._build_dim_unifier(combined_emb.size(-1))  # 懒创建/复用 Linear
        return self.dim_unifier(combined_emb)

    def forward_classification(self, hla_idx=None, tcr_idx=None, pep_idx=None, task_type='hla_pep'):
        """使用Decoder做分类 - 修复参数传递"""
        task_map = {'hla_pep': 0, 'tcr_pep': 1, 'triple': 2}
        task_id = task_map[task_type]

        # 编码输入 - 根据任务类型传递正确的参数
        if task_type == 'hla_pep':
            memory = self.encode_task_inputs(hla_idx=hla_idx, pep_idx=pep_idx, task_id=task_id)
        elif task_type == 'tcr_pep':
            memory = self.encode_task_inputs(tcr_idx=tcr_idx, pep_idx=pep_idx, task_id=task_id)
        else:  # triple
            memory = self.encode_task_inputs(tcr_idx=tcr_idx, hla_idx=hla_idx, pep_idx=pep_idx, task_id=task_id)

        # 为分类任务创建特殊的起始token
        batch_size = memory.size(0)
        start_tokens = torch.full((batch_size, 1), 0, device=self.device)  # <sos>

        # 通过decoder获取表示
        decoder_output = self.decoder(memory, start_tokens, max_len=1, teacher_ratio=0)

        # 使用最后一个隐藏状态进行分类
        cls_representation = decoder_output[:, -1, :]
        logits = self.classification_head(cls_representation).squeeze(-1)

        return logits

    def forward_generation(self, hla_idx=None, pep_idx=None, tcr_idx=None, tgt=None, teacher_ratio=0.5):
        """使用Decoder做生成 - 修复参数传递"""
        task_id = 3  # generation task

        # 编码输入
        memory = self.encode_task_inputs(hla_idx=hla_idx, pep_idx=pep_idx, task_id=task_id)

        # 通过decoder生成TCR序列
        outputs = self.decoder(memory, tgt, teacher_ratio=teacher_ratio)
        return outputs


# ========== 评估函数 ==========
def evaluate_model(model, dataloader, task_type, device, threshold=0.5):
    """评估模型性能"""
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            if task_type == 'hla_pep':
                logits = model.forward_classification(
                    hla_idx=batch[0].to(device),
                    pep_idx=batch[2].to(device),
                    task_type='hla_pep'
                )
            elif task_type == 'tcr_pep':
                logits = model.forward_classification(
                    tcr_idx=batch[0].to(device),
                    pep_idx=batch[2].to(device),
                    task_type='tcr_pep'
                )
            else:  # triple
                logits = model.forward_classification(
                    tcr_idx=batch[0].to(device),
                    hla_idx=batch[1].to(device),
                    pep_idx=batch[2].to(device),
                    task_type='triple'
                )

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            labels = batch[3].cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    # 计算评估指标
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # 计算AUC和AUPRC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    try:
        auprc = average_precision_score(all_labels, all_probs)
    except:
        auprc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'auprc': auprc,
        'probs': all_probs,
        'labels': all_labels
    }


def plot_performance_curves(results_dict, save_path='performance_curves.png'):
    """绘制AUC和AUPRC曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (task_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]

        # ROC曲线
        fpr, tpr, _ = roc_curve(results['labels'], results['probs'])
        ax1.plot(fpr, tpr, label=f'{task_name} (AUC = {results["auc"]:.3f})',
                 color=color, linewidth=2)

        # PR曲线
        precision, recall, _ = precision_recall_curve(results['labels'], results['probs'])
        ax2.plot(recall, precision, label=f'{task_name} (AUPRC = {results["auprc"]:.3f})',
                 color=color, linewidth=2)

    # 设置ROC曲线图
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 设置PR曲线图
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"性能曲线已保存至: {save_path}")


def print_evaluation_metrics(results_dict):
    """打印评估指标"""
    print("模型评估结果")
    headers = ["Task", "Precision", "Recall", "Accuracy", "F1-Score", "AUC", "AUPRC"]
    print(
        f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10} {headers[6]:<10}")

    for task_name, results in results_dict.items():
        print(f"{task_name:<15} {results['precision']:.4f}    {results['recall']:.4f}    "
              f"{results['accuracy']:.4f}    {results['f1']:.4f}    "
              f"{results['auc']:.4f}    {results['auprc']:.4f}")


# ========== 训练函数 ==========
def train_single_classification(model, dataloaders, args, eval_datasets=None):
    """单独训练分类任务，添加内存优化"""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    train_losses = []
    accumulation_steps = 4  # 梯度累积步数

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        optimizer.zero_grad()

        for task_name, dataloader in dataloaders.items():
            if task_name == 'generation':
                continue

            for i, batch in enumerate(dataloader):
                # 清理GPU内存
                if args.device.startswith('cuda') and (i + 1) % 100 == 0:
                    cuda.empty_cache()

                if task_name == 'hla_pep':
                    logits = model.forward_classification(
                        hla_idx=batch[0].to(args.device),
                        pep_idx=batch[2].to(args.device),
                        task_type='hla_pep'
                    )
                elif task_name == 'tcr_pep':
                    logits = model.forward_classification(
                        tcr_idx=batch[0].to(args.device),
                        pep_idx=batch[2].to(args.device),
                        task_type='tcr_pep'
                    )
                else:  # triple
                    logits = model.forward_classification(
                        tcr_idx=batch[0].to(args.device),
                        hla_idx=batch[1].to(args.device),
                        pep_idx=batch[2].to(args.device),
                        task_type='triple'
                    )

                loss = criterion(logits, batch[3].float().to(args.device))
                loss = loss / accumulation_steps  # 梯度累积
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                batch_count += 1

        # 处理剩余的梯度
        if batch_count % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)

        # 每10个epoch评估一次
        if (epoch + 1) % 10 == 0 and eval_datasets:
            print(f"\nEpoch {epoch + 1} 评估结果:")
            results_dict = {}

            for task_name, dataset in eval_datasets.items():
                if dataset is None:
                    continue

                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                task_type_map = {
                    'pep_hla_label': 'hla_pep',
                    'pep_tcr_label': 'tcr_pep',
                    'tcr_pep_hla_label': 'triple'
                }
                task_type = task_type_map[task_name]

                results = evaluate_model(model, dataloader, task_type, args.device)
                results_dict[task_name] = results

                current_auc = results['auc']
                if current_auc > best_auc:
                    best_auc = current_auc
                    torch.save(model.state_dict(), 'best_model.pth')
                    print(f"保存最佳模型，AUC: {best_auc:.4f}")

            print_evaluation_metrics(results_dict)

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")

    return train_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--neighbor_sample_size', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--r_dim', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=30)

    args = parser.parse_args()

    print(f"使用设备: {args.device}")

    # 加载数据并构建知识图谱
    n_entitys, n_relations, adj_entity, adj_relation, vocab_size, all_datasets, real_entity_sequences = load_data_and_build_kg(
        args)

    print(f"实体数量: {n_entitys}")
    print(f"关系数量: {n_relations}")

    # 创建数据加载器
    dataloaders = {}
    eval_datasets = {}

    for task_name, dataset in all_datasets.items():
        if dataset is not None:
            dataloaders[task_name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            eval_datasets[task_name] = dataset

    # 创建模型
    model = UnifiedModel(
        args=args,
        n_entitys=n_entitys,
        n_relations=n_relations,
        e_dim=args.e_dim,
        r_dim=args.r_dim,
        adj_entity=adj_entity,
        adj_relation=adj_relation,
        vocab_size=vocab_size,
        max_len=args.max_len,
        device=args.device
    ).to(args.device)

    print("模型结构:")
    print(model)

    # 训练模型
    print("开始训练模型...")
    train_losses = train_single_classification(model, dataloaders, args, eval_datasets)

    # 最终评估
    print("\n最终评估结果:")
    final_results = {}

    for task_name, dataset in eval_datasets.items():
        if dataset is None:
            continue

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        task_type_map = {
            'pep_hla_label': 'hla_pep',
            'pep_tcr_label': 'tcr_pep',
            'tcr_pep_hla_label': 'triple'
        }
        task_type = task_type_map[task_name]

        results = evaluate_model(model, dataloader, task_type, args.device)
        final_results[task_name] = results

    # 打印最终结果
    print_evaluation_metrics(final_results)

    # 绘制性能曲线
    plot_performance_curves(final_results)

    print("训练完成！")


if __name__ == '__main__':
    main()