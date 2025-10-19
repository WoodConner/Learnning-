# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import random, os, re, math
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, \
    f1_score


# ----------- 1. 数据加载 -----------
def load_hla_pep(file):
    """HLA,peptide,label(0/1)"""
    if not os.path.exists(file):
        print(f"[WARN] {file} 不存在");
        return []
    with open(file) as f:
        lines = f.readlines()
    data = []
    for ln in lines:
        parts = ln.strip().split(',')
        if len(parts) < 3: continue
        hla, pep, lab = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if lab in ['0', '1']:
            data.append((hla, pep, int(lab)))
    return data


def load_tcr_pep(file):
    """TCR,peptide,label(0/1)"""
    if not os.path.exists(file):
        print(f"[WARN] {file} 不存在");
        return []
    with open(file) as f:
        lines = f.readlines()
    data = []
    for ln in lines:
        parts = ln.strip().split(',')
        if len(parts) < 3: continue
        tcr, pep, lab = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if lab in ['0', '1']:
            data.append((tcr, pep, int(lab)))
    return data


def load_tcr_hla_pep(file):
    """pep,hla,tcr,label(0/1) → 转 (TCR,HLA,pep,lab)"""
    if not os.path.exists(file):
        print(f"[WARN] {file} 不存在");
        return []
    with open(file) as f:
        lines = f.readlines()
    data = []
    for ln in lines:
        parts = ln.strip().split(',')
        if len(parts) < 4: continue
        pep, hla, tcr, lab = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        if lab in ['0', '1']:
            data.append((tcr, hla, pep, int(lab)))  # 统一成 (TCR, HLA, pep, lab)
    return data


# ----------- 2. 通用工具 -----------
def build_global_peptide_mapping(hla_data, tcr_data, tcr_hla_data=None):
    pep_set = set()
    for _, pep, _ in hla_data + tcr_data:
        pep_set.add(pep)
    if tcr_hla_data:
        for _, _, pep, _ in tcr_hla_data:
            pep_set.add(pep)
    pep2id = {p: i for i, p in enumerate(sorted(pep_set))}
    id2pep = {i: p for p, i in pep2id.items()}
    return pep2id, id2pep, len(pep2id)


def build_global_tcr_mapping(tcr_data, tcr_hla_data=None):
    """构建TCR的全局映射"""
    tcr_set = set()
    for tcr, _, _ in tcr_data:
        tcr_set.add(tcr)
    if tcr_hla_data:
        for tcr, _, _, _ in tcr_hla_data:
            tcr_set.add(tcr)
    tcr2id = {t: i for i, t in enumerate(sorted(tcr_set))}
    id2tcr = {i: t for t, i in tcr2id.items()}
    return tcr2id, id2tcr, len(tcr2id)


def build_global_hla_mapping(hla_data, tcr_hla_data=None):
    """构建HLA的全局映射"""
    hla_set = set()
    for hla, _, _ in hla_data:
        hla_set.add(hla)
    if tcr_hla_data:
        for _, hla, _, _ in tcr_hla_data:
            hla_set.add(hla)
    hla2id = {h: i for i, h in enumerate(sorted(hla_set))}
    id2hla = {i: h for h, i in hla2id.items()}
    return hla2id, id2hla, len(hla2id)


def create_entity_mapping(data):
    ents = set()
    for triple in data:
        if len(triple) == 3:  # (h, t, lab) 格式
            h, t, _ = triple
            ents.add(h);
            ents.add(t)
    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    id2ent = {i: e for e, i in ent2id.items()}
    return ent2id, id2ent, len(ent2id)


def create_kg_triples(data, ent2id):
    triples = []
    for h, t, lab in data:
        if h in ent2id and t in ent2id:
            triples.append((ent2id[h], lab, ent2id[t]))
    return triples


def build_adjacency_list(triples, n_ent, n_rel):
    kg = defaultdict(list)
    for h, r, t in triples:
        kg[h].append((t, r))
    for e in range(n_ent):
        if not kg[e]:
            kg[e] = [(e, 0)]
    return kg


def construct_adj(neighbor_sample_size, kg, n_ent, device='cpu'):
    # 预分配数组
    adj_ent = np.zeros([n_ent, neighbor_sample_size], dtype=np.int64)
    adj_rel = np.zeros([n_ent, neighbor_sample_size], dtype=np.int64)

    for e in range(n_ent):
        neighbors = kg[e]
        n_nei = len(neighbors)
        if n_nei >= neighbor_sample_size:
            samp = np.random.choice(n_nei, neighbor_sample_size, replace=False)
        else:
            samp = np.random.choice(n_nei, neighbor_sample_size, replace=True)
        adj_ent[e] = [neighbors[i][0] for i in samp]
        adj_rel[e] = [neighbors[i][1] for i in samp]

    # 一次性转换为tensor，避免警告
    adj_ent_tensor = torch.LongTensor(adj_ent).to(device)
    adj_rel_tensor = torch.LongTensor(adj_rel).to(device)

    return adj_ent_tensor, adj_rel_tensor


# ----------- 3. KGAN -----------
class KGAN(nn.Module):
    def __init__(self, n_ent, n_rel, e_dim, r_dim, adj_ent, adj_rel,
                 agg_method='Bi-Interaction', device='cpu'):
        super().__init__()
        self.device = device
        self.n_ent = n_ent
        self.ent_embs = nn.Embedding(n_ent, e_dim, max_norm=1)
        self.rel_embs = nn.Embedding(n_rel, r_dim, max_norm=1)
        # 直接存储tensor，避免重复转换
        self.register_buffer('adj_ent', adj_ent)
        self.register_buffer('adj_rel', adj_rel)
        self.agg_method = agg_method
        self.Wr = nn.Linear(e_dim, r_dim)
        self.leaky = nn.LeakyReLU(0.2)
        if agg_method == 'concat':
            self.Wcat = nn.Linear(e_dim * 2, e_dim)
        else:
            self.W1 = nn.Linear(e_dim, e_dim)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(e_dim, e_dim)
        self.combine_layer = nn.Sequential(
            nn.Linear(e_dim * 2, e_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(e_dim)
        )
        self.rel_classifier = nn.Sequential(
            nn.Linear(e_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.triple_classifier = nn.Sequential(
            nn.Linear(e_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)

    def encode_hla_peptide_pair(self, hla_idx, pep_idx):
        h = self.forward(hla_idx)
        p = self.forward(pep_idx)
        # 确保h和p的维度相同
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        return self.combine_layer(torch.cat([h, p], dim=-1))

    def get_neighbors(self, items):
        # 直接使用索引操作，避免列表操作
        if isinstance(items, torch.Tensor):
            items = items.cpu()

        # 确保索引在有效范围内
        items = torch.clamp(items, 0, self.n_ent - 1)

        # 直接使用高级索引，避免循环
        e_ids = self.adj_ent[items]
        r_ids = self.adj_rel[items]

        return self.ent_embs(e_ids), self.rel_embs(r_ids)

    def GATMessagePass(self, h_embs, r_embs, t_embs):
        h_broadcast = h_embs.unsqueeze(1).expand(-1, t_embs.size(1), -1)
        tr = self.Wr(t_embs)
        hr = self.Wr(h_broadcast)
        att = torch.softmax(torch.sum(torch.tanh(hr + r_embs) * tr, dim=-1, keepdim=True), dim=1)
        return (t_embs * att).sum(dim=1)

    def aggregate(self, h_embs, Nh_embs):
        if self.agg_method == 'Bi-Interaction':
            return self.leaky(self.W1(h_embs + Nh_embs)) + self.leaky(self.W2(h_embs * Nh_embs))
        elif self.agg_method == 'concat':
            return self.leaky(self.Wcat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:
            return self.leaky(self.W1(h_embs + Nh_embs))

    def forward(self, idx):
        idx = idx.to(self.device)
        # 确保索引在有效范围内
        idx = torch.clamp(idx, 0, self.n_ent - 1)
        t_embs, r_embs = self.get_neighbors(idx)
        h_embs = self.ent_embs(idx)
        Nh = self.GATMessagePass(h_embs, r_embs, t_embs)
        return self.aggregate(h_embs, Nh)

    def forward_relation(self, h_idx, t_idx):
        h_e = self.forward(h_idx)
        t_e = self.forward(t_idx)
        concat = torch.cat([h_e, t_e], dim=-1)
        logits = self.rel_classifier(concat).squeeze(-1)
        return logits, h_e, t_e

    def forward_triple(self, tcr_idx, hla_idx, pep_idx):
        """三元组分类：TCR-HLA-肽"""
        tcr_e = self.forward(tcr_idx)
        hla_e = self.forward(hla_idx)
        pep_e = self.forward(pep_idx)

        # 拼接三个实体的嵌入
        triple_emb = torch.cat([tcr_e, hla_e, pep_e], dim=-1)
        logits = self.triple_classifier(triple_emb).squeeze(-1)
        return logits


# ----------- 4. Transformer Decoder -----------
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
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, vocab_size,
                 dropout=0.1, max_len=100):
        super().__init__()
        self.d_model, self.vocab_size, self.max_len = d_model, vocab_size, max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                           dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)

    def generate_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, mem, tgt=None, max_len=None, teacher_ratio=0.5):
        B = mem.size(0)
        max_len = max_len or self.max_len

        # 投影记忆向量
        mem_proj = self.proj(mem).unsqueeze(1)  # [B, 1, d_model]

        if tgt is not None and random.random() < teacher_ratio:
            # 教师强制训练 - 输入不包括最后一个token
            tgt_input = tgt[:, :-1]
            tgt_emb = self.emb(tgt_input) * math.sqrt(self.d_model)
            tgt_emb = self.pos(self.drop(tgt_emb))

            tgt_mask = self.generate_mask(tgt_input.size(1)).to(mem.device)

            out = self.decoder(
                tgt=tgt_emb,
                memory=mem_proj,
                tgt_mask=tgt_mask
            )
            return self.out(out)
        else:
            # 推理模式
            tokens = torch.full((B, 1), 0, dtype=torch.long, device=mem.device)  # 从<sos>开始
            logits = []

            for i in range(max_len):
                tgt_emb = self.emb(tokens) * math.sqrt(self.d_model)
                tgt_emb = self.pos(self.drop(tgt_emb))

                tgt_mask = self.generate_mask(tokens.size(1)).to(mem.device)

                out = self.decoder(
                    tgt=tgt_emb,
                    memory=mem_proj,
                    tgt_mask=tgt_mask
                )

                next_logit = self.out(out[:, -1, :])  # 只取最后一个时间步
                logits.append(next_logit.unsqueeze(1))

                next_token = next_logit.argmax(-1).unsqueeze(1)
                tokens = torch.cat([tokens, next_token], dim=1)

                # 如果所有序列都生成了<eos>，提前停止
                if (next_token == 1).all():  # <eos>=1
                    break

            return torch.cat(logits, dim=1)  # [B, generated_len, vocab_size]


# ----------- 5. TCR Dataset -----------
class TCRGenerationDataset(Dataset):
    def __init__(self, tcr_data, hla_ent2id, pep2id, max_len=30):
        self.max_len = max_len
        self.pep2id, self.id2pep = pep2id, {v: k for k, v in pep2id.items()}

        # 收集所有有效的氨基酸字符
        aas = set()
        for tcr, hla, pep, lab in tcr_data:
            clean_tcr = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', tcr.upper())
            aas.update(clean_tcr)

        self.aas = sorted(aas)
        self.vocab_size = len(self.aas) + 3  # +3 for <sos>=0, <eos>=1, <pad>=2
        self.aa2i = {aa: i + 3 for i, aa in enumerate(self.aas)}  # 氨基酸从3开始
        self.i2aa = {i + 3: aa for i, aa in enumerate(self.aas)}

        # 特殊token
        self.sos, self.eos, self.pad = 0, 1, 2
        self.aa2i['<sos>'] = self.sos
        self.aa2i['<eos>'] = self.eos
        self.aa2i['<pad>'] = self.pad
        self.i2aa[self.sos] = '<sos>'
        self.i2aa[self.eos] = '<eos>'
        self.i2aa[self.pad] = '<pad>'

        # 构建样本
        self.samples = []
        for tcr, hla, pep, lab in tcr_data:
            if hla in hla_ent2id and pep in self.pep2id:
                clean_tcr = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', tcr.upper())
                if clean_tcr and len(clean_tcr) > 3:  # 确保TCR有效且有一定长度
                    encoded_tcr = self.encode_tcr(clean_tcr)
                    self.samples.append((hla_ent2id[hla], self.pep2id[pep], encoded_tcr))

        print(f'[Dataset] 最终样本数: {len(self.samples)}')
        print(f'[Dataset] 词汇表大小: {self.vocab_size}')
        print(f'[Dataset] 最大序列长度: {max_len}')
        assert len(self.samples) > 0, "Dataset 内部样本为空！"

    def encode_tcr(self, seq):
        # 编码: <sos> + 氨基酸序列 + <eos> + <pad>...
        tokens = [self.sos] + [self.aa2i.get(aa, self.pad) for aa in seq] + [self.eos]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            tokens[-1] = self.eos  # 确保以eos结尾
        else:
            tokens += [self.pad] * (self.max_len - len(tokens))
        return tokens

    def decode_tcr(self, tok):
        if isinstance(tok, torch.Tensor):
            tok = tok.cpu().numpy()
        seq = []
        for t in tok:
            if t == self.eos or t == self.pad:
                break
            if t >= 3 and t in self.i2aa:  # 只解码氨基酸token
                seq.append(self.i2aa[t])
        return ''.join(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h, p, t = self.samples[idx]
        return torch.LongTensor([h]), torch.LongTensor([p]), torch.LongTensor(t)


class TCRGenerator(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, hla, pep, tgt=None, max_len=20, teacher_ratio=0.5):
        mem = self.enc.encode_hla_peptide_pair(hla, pep)
        return self.dec(mem, tgt, max_len, teacher_ratio)


# ----------- 6. 训练逻辑 -----------
def calculate_metrics(y_true, y_pred, y_score):
    """计算所有评估指标"""
    metrics = {}

    # 二分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # 概率指标
    if len(np.unique(y_true)) > 1:  # 确保有正负样本
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            metrics['auc_pr'] = average_precision_score(y_true, y_score)
        except:
            metrics['auc_roc'] = 0.5
            metrics['auc_pr'] = 0.5
    else:
        metrics['auc_roc'] = 0.5
        metrics['auc_pr'] = 0.5

    return metrics


def print_metrics(metrics, prefix=""):
    """打印评估指标"""
    print(f"{prefix}评估指标:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")


def pretrain_kgan_with_relation(data, n_epochs=10, batch_size=32,
                                device='cpu', lambda_rel=1.0,
                                pep2id=None, relation_type="hla_pep"):
    print(f"=== 预训练 KGAN（{relation_type} 关系学习） ===")

    # 创建实体映射
    ent2id, id2ent, n_ents = create_entity_mapping(data)

    # 添加肽到实体映射
    if pep2id:
        for p in pep2id:
            if p not in ent2id:
                ent2id[p] = len(ent2id)

    n_ent = len(ent2id)

    # 直接使用数据中的所有三元组（包含正负样本）
    triples = create_kg_triples(data, ent2id)
    random.shuffle(triples)

    # 统计正负样本
    pos_count = sum(1 for _, r, _ in triples if r == 1)
    neg_count = sum(1 for _, r, _ in triples if r == 0)
    print(f"正样本: {pos_count}, 负样本: {neg_count}, 总计: {len(triples)}")

    n_rel = 2
    kg_idx = build_adjacency_list(triples, n_ent, n_rel)
    adj_ent, adj_rel = construct_adj(20, kg_idx, n_ent, device)

    e_dim, r_dim = 64, 32
    model = KGAN(n_ent, n_rel, e_dim, r_dim, adj_ent, adj_rel, device=device).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    # 用于评估的列表
    all_preds = []
    all_labels = []
    all_scores = []

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(triples)
        total_loss, total_acc, n = 0, 0, 0

        # 清空评估列表
        epoch_preds = []
        epoch_labels = []
        epoch_scores = []

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            # 使用stack而不是列表推导，避免警告
            h = torch.tensor([b[0] for b in batch], dtype=torch.long).to(device)
            r = torch.tensor([b[1] for b in batch], dtype=torch.float).to(device)
            t = torch.tensor([b[2] for b in batch], dtype=torch.long).to(device)

            logits, h_e, t_e = model.forward_relation(h, t)
            loss_rel = bce(logits, r)
            loss_rec = (h_e.norm(2, dim=1).mean() + t_e.norm(2, dim=1).mean())
            loss = loss_rel + lambda_rel * loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(batch)
            pred = (torch.sigmoid(logits) > 0.5).float()
            total_acc += (pred == r).sum().item()
            n += len(batch)

            # 收集评估数据
            epoch_preds.extend(pred.cpu().numpy())
            epoch_labels.extend(r.cpu().numpy())
            epoch_scores.extend(torch.sigmoid(logits).cpu().detach().numpy())

        # 计算epoch指标
        epoch_metrics = calculate_metrics(epoch_labels, epoch_preds, epoch_scores)

        print(
            f'{relation_type.upper()} PreTrain Epoch {epoch + 1}/{n_epochs}  loss={total_loss / n:.4f}  acc={total_acc / n:.4f}')
        print_metrics(epoch_metrics, f"  {relation_type.upper()} ")

        # 收集所有epoch的数据
        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)
        all_scores.extend(epoch_scores)

    # 计算总体指标
    final_metrics = calculate_metrics(all_labels, all_preds, all_scores)
    print(f"\n{relation_type.upper()} 预训练最终指标:")
    print_metrics(final_metrics)

    return model, ent2id, pep2id, n_ent


def train_triple_classifier(model, triple_data, tcr2id, hla2id, pep2id,
                            n_epochs=10, batch_size=32, device='cpu'):
    """训练三元组分类器"""
    print("=== 训练三元组分类器 (TCR-HLA-肽) ===")

    # 准备训练数据 - 直接使用数据中的正负样本
    train_triples = []
    for tcr, hla, pep, lab in triple_data:
        if tcr in tcr2id and hla in hla2id and pep in pep2id:
            train_triples.append((tcr2id[tcr], hla2id[hla], pep2id[pep], lab))

    # 统计正负样本
    pos_count = sum(1 for _, _, _, lab in train_triples if lab == 1)
    neg_count = sum(1 for _, _, _, lab in train_triples if lab == 0)

    print(f"正样本: {pos_count}, 负样本: {neg_count}, 总计: {len(train_triples)}")

    random.shuffle(train_triples)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()

    # 用于评估的列表
    all_preds = []
    all_labels = []
    all_scores = []

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_triples)
        total_loss, total_acc, n = 0, 0, 0

        # 清空epoch评估列表
        epoch_preds = []
        epoch_labels = []
        epoch_scores = []

        for i in range(0, len(train_triples), batch_size):
            batch = train_triples[i:i + batch_size]

            tcr_idx = torch.tensor([b[0] for b in batch], dtype=torch.long).to(device)
            hla_idx = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
            pep_idx = torch.tensor([b[2] for b in batch], dtype=torch.long).to(device)
            labels = torch.tensor([b[3] for b in batch], dtype=torch.float).to(device)

            logits = model.forward_triple(tcr_idx, hla_idx, pep_idx)
            loss = bce(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(batch)
            pred = (torch.sigmoid(logits) > 0.5).float()
            total_acc += (pred == labels).sum().item()
            n += len(batch)

            # 收集评估数据
            epoch_preds.extend(pred.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
            epoch_scores.extend(torch.sigmoid(logits).cpu().detach().numpy())

        # 计算epoch指标
        epoch_metrics = calculate_metrics(epoch_labels, epoch_preds, epoch_scores)

        print(f'Triple Classifier Epoch {epoch + 1}/{n_epochs}  loss={total_loss / n:.4f}  acc={total_acc / n:.4f}')
        print_metrics(epoch_metrics, "  Triple ")

        # 收集所有epoch的数据
        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)
        all_scores.extend(epoch_scores)

    # 计算总体指标
    final_metrics = calculate_metrics(all_labels, all_preds, all_scores)
    print(f"\n三元组分类器最终指标:")
    print_metrics(final_metrics)

    return model


def finetune_tcr_generator(encoder, tcr_data, hla_ent2id, pep2id,
                           n_epochs=10, batch_size=32, device='cpu'):
    print("=== 微调 TCR 生成 ===")
    ds = TCRGenerationDataset(tcr_data, hla_ent2id, pep2id)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 调试：检查数据集
    print(f'数据集大小: {len(ds)}')
    print(f'词汇表大小: {ds.vocab_size}')
    print(f'特殊token: SOS={ds.sos}, EOS={ds.eos}, PAD={ds.pad}')

    # 检查几个样本
    for i in range(min(3, len(ds))):
        h, p, tgt = ds[i]
        decoded = ds.decode_tcr(tgt)
        print(f"样本{i}: HLA={h.item()}, PEP={p.item()}, TCR长度={len(decoded)}, TCR={decoded}")

    decoder = TransformerDecoder(input_dim=64, d_model=256, nhead=8,
                                 num_layers=3, vocab_size=ds.vocab_size,
                                 dropout=0.1, max_len=ds.max_len).to(device)
    model = TCRGenerator(encoder, decoder).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=ds.pad)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        batch_count = 0

        for h, p, tgt in tqdm(dl, desc=f'Epoch {epoch + 1}'):
            h, p, tgt = h.squeeze(1).to(device), p.squeeze(1).to(device), tgt.to(device)

            # 前向传播 - 使用教师强制
            out = model(h, p, tgt, teacher_ratio=0.8)

            # 输出应该是 [batch, seq_len-1, vocab_size]，因为预测下一个token
            if out.size(1) != tgt.size(1) - 1:
                # 如果长度不匹配，取最小值
                seq_len = min(out.size(1), tgt.size(1) - 1)
                out = out[:, :seq_len, :]
                targets = tgt[:, 1:seq_len + 1]  # 目标是从第2个token开始
            else:
                targets = tgt[:, 1:]  # 去掉<sos>，取后面的作为目标

            # 重塑用于损失计算
            logits = out.reshape(-1, ds.vocab_size)
            targets = targets.reshape(-1)

            # 只计算非pad token的损失
            non_pad_mask = targets != ds.pad
            valid_tokens = non_pad_mask.sum().item()

            if valid_tokens > 0:
                loss = criterion(logits[non_pad_mask], targets[non_pad_mask])

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

            batch_count += 1

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)  # 计算困惑度
            print(f'Epoch {epoch + 1}/{n_epochs}  loss={avg_loss:.4f}  ppl={ppl:.2f}  tokens={total_tokens}')
        else:
            print(f'Epoch {epoch + 1}/{n_epochs}  No valid tokens')

    return model, ds


# 添加验证函数
def validate_model(model, dl, criterion, ds, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for h, p, tgt in dl:
            h, p, tgt = h.squeeze(1).to(device), p.squeeze(1).to(device), tgt.to(device)

            # 不使用教师强制进行验证
            out = model(h, p, max_len=ds.max_len)

            # 确保输出长度正确
            seq_len = min(out.size(1), tgt.size(1) - 1)
            out = out[:, :seq_len, :]
            targets = tgt[:, 1:seq_len + 1]

            logits = out.reshape(-1, ds.vocab_size)
            targets = targets.reshape(-1)

            non_pad_mask = targets != ds.pad
            valid_tokens = non_pad_mask.sum().item()

            if valid_tokens > 0:
                loss = criterion(logits[non_pad_mask], targets[non_pad_mask])
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def generate_tcr(model, ds, hla, peptide, hla_ent2id, pep2id, device='cpu'):
    model.eval()
    if hla not in hla_ent2id or peptide not in pep2id:
        print("未知 HLA 或 peptide");
        return None
    h_idx = torch.LongTensor([hla_ent2id[hla]]).to(device)
    p_idx = torch.LongTensor([pep2id[peptide]]).to(device)
    with torch.no_grad():
        out = model(h_idx, p_idx, max_len=30)
        tok = out.argmax(-1).cpu().numpy().squeeze()
        return ds.decode_tcr(tok)


# ----------- 7. 融合编码器 -----------
class FusionEncoder(nn.Module):
    def __init__(self, hla_encoder, tcr_encoder):
        super().__init__()
        self.hla_encoder = hla_encoder
        self.tcr_encoder = tcr_encoder
        # 使用HLA编码器的combine_layer
        self.combine_layer = hla_encoder.combine_layer

    def encode_hla_peptide_pair(self, hla_idx, pep_idx):
        h = self.hla_encoder.forward(hla_idx)
        p_hla = self.hla_encoder.forward(pep_idx)
        p_tcr = self.tcr_encoder.forward(pep_idx)

        # 融合策略：平均池化
        p_fused = (p_hla + p_tcr) / 2

        # 确保维度匹配
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if p_fused.dim() == 1:
            p_fused = p_fused.unsqueeze(0)

        return self.combine_layer(torch.cat([h, p_fused], dim=-1))


# ----------- 8. main -----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载所有数据
    print("=== 加载数据 ===")
    hla_file = "data_random1x/pep_hla_random1x.csv"
    tcr_pep_file = "data_random1x/pep_tcr_random1x.csv"
    tcr_file = "data_random1x/trimer_random1x.csv"

    hla_data = load_hla_pep(hla_file)
    tcr_pep_data = load_tcr_pep(tcr_pep_file)
    tcr_hla_data = load_tcr_hla_pep(tcr_file)

    print(f"HLA-pep 数据: {len(hla_data)}")
    print(f"TCR-pep 数据: {len(tcr_pep_data)}")
    print(f"TCR-HLA-pep 三元组数据: {len(tcr_hla_data)}")

    # 2. 构建全局映射
    print("=== 构建全局映射 ===")
    pep2id, _, n_pep = build_global_peptide_mapping(hla_data, tcr_pep_data, tcr_hla_data)
    tcr2id, _, n_tcr = build_global_tcr_mapping(tcr_pep_data, tcr_hla_data)
    hla2id, _, n_hla = build_global_hla_mapping(hla_data, tcr_hla_data)

    print(f"肽数量: {n_pep}, TCR数量: {n_tcr}, HLA数量: {n_hla}")

    # 3. 第一阶段预训练：HLA-peptide关系
    print("=== 第一阶段：HLA-peptide关系预训练 ===")
    hla_encoder, hla_ent2id, pep2id, _ = pretrain_kgan_with_relation(
        hla_data, n_epochs=10, batch_size=64, device=device,
        lambda_rel=1.0, pep2id=pep2id, relation_type="hla_pep")

    # 4. 第二阶段预训练：TCR-peptide关系
    print("=== 第二阶段：TCR-peptide关系预训练 ===")
    tcr_encoder, tcr_ent2id, pep2id, _ = pretrain_kgan_with_relation(
        tcr_pep_data, n_epochs=10, batch_size=64, device=device,
        lambda_rel=1.0, pep2id=pep2id, relation_type="tcr_pep")

    # 5. 三元组分类训练
    print("=== 第三阶段：三元组分类训练 ===")
    # 使用TCR编码器进行三元组分类（因为它已经包含了所有实体）
    triple_classifier = train_triple_classifier(
        tcr_encoder, tcr_hla_data, tcr2id, hla2id, pep2id,
        n_epochs=10, batch_size=32, device=device
    )

    # 6. 融合编码器
    fusion_encoder = FusionEncoder(hla_encoder, tcr_encoder).to(device)

    # 7. 微调：用 (HLA, peptide) 生成 TCR
    print("=== 第四阶段：TCR生成微调 ===")

    # 统一格式 & 自动扩映射
    for tcr, hla, pep, lab in tcr_hla_data:
        hla = hla.replace('-', '*')
        pep = pep.upper()
        if hla not in hla_ent2id:
            hla_ent2id[hla] = len(hla_ent2id)
        if pep not in pep2id:
            pep2id[pep] = len(pep2id)

    # 调试 & 空保护
    print('[DEBUG] 原始微调样本数:', len(tcr_hla_data))
    filtered_data = [(t, hla, pep, lab) for t, hla, pep, lab in tcr_hla_data
                     if hla in hla_ent2id and pep in pep2id and lab == 1]  # 只使用正样本进行生成
    print('[DEBUG] 过滤后样本数:', len(filtered_data))

    # 分割训练集和验证集
    random.shuffle(filtered_data)
    split_idx = int(0.8 * len(filtered_data))
    train_data = filtered_data[:split_idx]
    val_data = filtered_data[split_idx:]

    print(f'训练集大小: {len(train_data)}')
    print(f'验证集大小: {len(val_data)}')

    # 使用融合编码器进行微调
    model, ds = finetune_tcr_generator(fusion_encoder, train_data,
                                       hla_ent2id, pep2id,
                                       n_epochs=20, batch_size=32, device=device)

    # 验证模型
    val_ds = TCRGenerationDataset(val_data, hla_ent2id, pep2id)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    val_loss = validate_model(model, val_dl, nn.CrossEntropyLoss(ignore_index=ds.pad), ds, device)
    print(f'验证损失: {val_loss:.4f}, 验证困惑度: {math.exp(val_loss):.2f}')

    # 8. 推理：任意 (HLA, peptide) → TCR
    print("\n=== 生成示例（HLA, peptide）→ TCR ===")
    test_samples = [(h, p) for _, h, p, _ in random.sample(val_data, 5)]  # 使用验证集样本
    for hla, pep in test_samples:
        generated = generate_tcr(model, ds, hla, pep, hla_ent2id, pep2id, device)
        print(f'HLA:{hla}  Peptide:{pep}  =>  TCR:{generated}')

    # 保存模型
    torch.save({
        'model': model.state_dict(),
        'hla_ent2id': hla_ent2id,
        'pep2id': pep2id,
        'tcr_ent2id': tcr_ent2id,
        'hla2id': hla2id,
        'tcr2id': tcr2id,
        'ds_vocab': ds.aa2i
    }, 'tcr_generator_complete.pth')
    print("保存完成。")


if __name__ == '__main__':
    main()