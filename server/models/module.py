import math
import time
import numpy as np
import torch
import torch.nn as nn
from .utils import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import logging

class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h))
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        g = torch.tanh(self.lin_xn(x) + self.lin_hn(r * h))
        return z * h + (1 - z) * g


class GRUODECell(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        g = torch.tanh(x + self.lin_hn(r * h))
        dh = (1 - z) * (g - h)
        # dh = (1 - z) * h + z * g # modify by author
        return dh


class myModel(torch.nn.Module):
    def __init__(self, n_feat, e_feat, device, pos_dim=0,
                 num_neighbors=20, drop_out=0.1,
                 get_checkpoint_path=None, hidsize=200, n_head=3, path_encode="GRU_time"):
        super(myModel, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.path_encode = path_encode
        self.device = device
        self.hidsize = hidsize
        self.n_head = n_head
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors)
        self.ngh_finder = None
        self.node_raw_embed = n_feat
        self.n_nodes = self.node_raw_embed.shape[0]
        self.feat_dim = self.node_raw_embed.shape[1]
        self.edge_raw_embed = e_feat
        self.n_edges = self.edge_raw_embed.shape[0]
        self.rels = self.n_edges/2
        self.e_feat_dim = self.edge_raw_embed.shape[1]
        assert self.e_feat_dim == self.feat_dim, (self.e_feat_dim, self.feat_dim)

        start_edge_embed = nn.Parameter(torch.Tensor(1, self.e_feat_dim))
        nn.init.xavier_uniform_(start_edge_embed, gain=nn.init.calculate_gain('relu'))
        self.start_edge_embed = start_edge_embed
        self.pos_dim = pos_dim
        self.walk_model_dim = self.e_feat_dim
        self.logger.info('neighbors: {}, node dim: {}, edge dim: {}, pos dim: {}'.format(self.num_neighbors,
                                                                                         self.feat_dim,
                                                                                         self.e_feat_dim,
                                                                                         self.pos_dim))
        self.dropout_p = drop_out
        self.walk_encoder = self.init_walk_encoder()
        self.checkpoint_path = get_checkpoint_path

        self.transform = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim * 2, self.n_nodes),
            # torch.nn.Tanh()
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.dropout = torch.nn.Dropout(drop_out)

        self.W_mlp_c_obj = nn.Linear(2 * self.feat_dim, self.n_nodes)
        self.W_mlp_c_sub = nn.Linear(2 * self.feat_dim, self.n_nodes)
        self.W_mlp_c_rel = nn.Linear(2 * self.feat_dim, self.n_edges)

    def init_walk_encoder(self):
        walk_encoder = WalkEncoder(feat_dim=self.walk_model_dim, pos_dim=self.pos_dim,
                                   model_dim=self.walk_model_dim, out_dim=self.feat_dim,
                                   n_head=self.n_head, dropout_p=self.dropout_p,
                                   logger=self.logger, device=self.device,
                                   path_encode=self.path_encode)
        return walk_encoder

    def decoder_score_comp(self, src_embed, rel_embed, obj):
        emb = torch.cat((src_embed, rel_embed), dim=1)
        tail_pred = self.transform(emb)

        return self.loss(tail_pred, obj), tail_pred

    def decoder_score_comp_rel(self, src_embed, dst_embed, rels):
        emb = torch.cat((src_embed, dst_embed), dim=1)
        rels_pred = self.transform(emb)

        return self.loss(rels_pred, rels), rels_pred

    def get_sec_embedd_agg(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l, extr_subgraph_src1, extr_subgraph_src2,
                           extr_subgraph_src3, mask1, mask2, mask3, ngh_sample_pram=64):

        subgraph_src1 = self.grab_subgraph(src_idx_l, cut_time_l - 1, e_idx_l=None, k=1,
                                           ngh_sample_pram=ngh_sample_pram)
        subgraph_src2 = self.grab_subgraph(src_idx_l, cut_time_l - 1, e_idx_l=None, k=2,
                                           ngh_sample_pram=ngh_sample_pram)
        subgraph_src3 = self.grab_subgraph(src_idx_l, cut_time_l - 1, e_idx_l=None, k=3,
                                           ngh_sample_pram=ngh_sample_pram)
        subgraph_src1 = self.subgraph_tree2walk(src_idx_l, cut_time_l, e_idx_l, subgraph_src1)
        subgraph_src2 = self.subgraph_tree2walk(src_idx_l, cut_time_l, e_idx_l, subgraph_src2)
        subgraph_src3 = self.subgraph_tree2walk(src_idx_l, cut_time_l, e_idx_l, subgraph_src3)
        bs = len(src_idx_l)
        path1_num = subgraph_src1[0].shape[1]
        path2_num = subgraph_src2[0].shape[1]
        path3_num = subgraph_src3[0].shape[1]
        mask1 = np.concatenate((np.ones((bs, path1_num)), mask1), axis=-1)
        mask2 = np.concatenate((np.ones((bs, path2_num)), mask2), axis=-1)
        mask3 = np.concatenate((np.ones((bs, path3_num)), mask3), axis=-1)
        # for i in range(3):
        subgraph_src1 = (np.concatenate((subgraph_src1[0], extr_subgraph_src1[0]), axis=1),
                         np.concatenate((subgraph_src1[1], extr_subgraph_src1[1]), axis=1),
                         np.concatenate((subgraph_src1[2], extr_subgraph_src1[2]), axis=1))
        subgraph_src2 = (np.concatenate((subgraph_src2[0], extr_subgraph_src2[0]), axis=1),
                         np.concatenate((subgraph_src2[1], extr_subgraph_src2[1]), axis=1),
                         np.concatenate((subgraph_src2[2], extr_subgraph_src2[2]), axis=1))
        subgraph_src3 = (np.concatenate((subgraph_src3[0], extr_subgraph_src3[0]), axis=1),
                         np.concatenate((subgraph_src3[1], extr_subgraph_src3[1]), axis=1),
                         np.concatenate((subgraph_src3[2], extr_subgraph_src3[2]), axis=1))
        subgraph_src = (subgraph_src1, subgraph_src2, subgraph_src3)
        mask = (mask1, mask2, mask3)
        src_embed_agg, attn_output_weights = self.forward_msg(src_idx_l, e_idx_l, cut_time_l, subgraph_src, mask)

        return src_embed_agg, attn_output_weights, subgraph_src

    def inference(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l, extr_subgraph_src1, extr_subgraph_src2,
                  extr_subgraph_src3, mask1, mask2, mask3, ngh_sample_pram=64):

        rel_embed = self.edge_raw_embed[e_idx_l]
        # e_idx_l+self.rels 后会变成float，tensor检索时检索向量如果是float类型的numpy.array则不会有问题（会自动截断小数），如果是float类型的tensor则会报错。
        # rel_embed_inv = self.edge_raw_embed[e_idx_l + self.rels]
        tgt_idx_l = torch.from_numpy(tgt_idx_l).to(self.device)
        src_idx_l = torch.from_numpy(src_idx_l).to(self.device)
        e_idx_l = torch.from_numpy(e_idx_l).to(self.device)
        attn_output_weights_src = None
        subgraph_src = None

        src_embed_agg, attn_output_weights_src, subgraph_src = self.get_sec_embedd_agg(src_idx_l.cpu().numpy(),
                                                                                       tgt_idx_l.cpu().numpy(),
                                                                                       cut_time_l,
                                                                                       e_idx_l.cpu().numpy(),
                                                                                       extr_subgraph_src1, extr_subgraph_src2,
                                                                                       extr_subgraph_src3, mask1, mask2,
                                                                                       mask3, ngh_sample_pram)
        loss2_src, score2_src = self.decoder_score_comp(src_embed_agg, rel_embed, tgt_idx_l)

        # dst_embed_agg, attn_output_weights_dst, subgraph_dst = self.get_sec_embedd_agg(tgt_idx_l.cpu().numpy(),
        #                                                                                src_idx_l.cpu().numpy(),
        #                                                                                all_nodes_l, cut_time_l,
        #                                                                                (e_idx_l + self.rels).cpu().long().numpy(),
        #                                                                                ngh_sample_pram)
        # loss2_dst, score2_dst = self.decoder_score_comp(dst_embed_agg, rel_embed_inv, src_idx_l)
        # loss2_rel, score2_rel = self.decoder_score_comp_rel(src_embed_agg, dst_embed_agg, e_idx_l)
        # loss = loss2_src + loss2_dst + loss2_rel
        loss = loss2_src
        score = score2_src

        return loss, score, attn_output_weights_src, subgraph_src

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None, k=3, ngh_sample_pram=64):
        if k == 1:
            # subgraph = self.ngh_finder.find_k_hop(1, src_idx_l, cut_time_l,
            #                                       num_neighbors=self.num_neighbors[0:1],
            #                                       e_idx_l=e_idx_l)
            subgraph = self.ngh_finder.find_k_hop(1, src_idx_l, cut_time_l,
                                                  num_neighbors=[ngh_sample_pram],
                                                  e_idx_l=e_idx_l)
        elif k == 2:
            # number_ngh = [int(self.num_neighbors[0]/32), int(self.num_neighbors[1])]
            # subgraph = self.ngh_finder.find_k_hop(2, src_idx_l, cut_time_l,
            #                                       num_neighbors=number_ngh,
            #                                       e_idx_l=e_idx_l)
            number_ngh = int(math.pow(ngh_sample_pram, 1 / 2))
            subgraph = self.ngh_finder.find_k_hop(2, src_idx_l, cut_time_l,
                                                  num_neighbors=[number_ngh, number_ngh],
                                                  e_idx_l=e_idx_l)
        else:
            # number_ngh = [int(self.num_neighbors[0]/32), int(self.num_neighbors[1]/2), int(self.num_neighbors[2])]
            # subgraph = self.ngh_finder.find_k_hop(3, src_idx_l, cut_time_l,
            #                                       num_neighbors=number_ngh,
            #                                       e_idx_l=e_idx_l)
            number_ngh = int(math.pow(ngh_sample_pram, 1 / 3))
            subgraph = self.ngh_finder.find_k_hop(3, src_idx_l, cut_time_l,
                                                  num_neighbors=[number_ngh, number_ngh, number_ngh],
                                                  e_idx_l=e_idx_l)

        return subgraph

    def subgraph_tree2walk(self, src_idx_l, cut_time_l, e_idx_l, subgraph_src):
        node_records, eidx_records, t_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        # eidx_records_tmp = [np.zeros_like(node_records_tmp[0])] + eidx_records
        eidx_records_tmp = [np.expand_dims(e_idx_l, 1)] + eidx_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records

        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        return new_node_records, new_eidx_records, new_t_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), \
            record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert (n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def forward_msg(self, src_idx_l, e_idx_l, cut_time_l, subgraph_src, mask):

        subgraph_src1, subgraph_src2, subgraph_src3 = subgraph_src
        node_records1, eidx_records1, t_records1 = subgraph_src1
        node_records2, eidx_records2, t_records2 = subgraph_src2
        node_records3, eidx_records3, t_records3 = subgraph_src3
        hidden_embeddings1, _ = self.init_hidden_embeddings(src_idx_l, node_records1)
        hidden_embeddings2, _ = self.init_hidden_embeddings(src_idx_l, node_records2)
        hidden_embeddings3, _ = self.init_hidden_embeddings(src_idx_l, node_records3)
        hidden_embeddings = (hidden_embeddings1, hidden_embeddings2, hidden_embeddings3)
        edge_features1 = self.retrieve_edge_features(eidx_records1)
        edge_features2 = self.retrieve_edge_features(eidx_records2)
        edge_features3 = self.retrieve_edge_features(eidx_records3)
        edge_features = (edge_features1, edge_features2, edge_features3)
        t_records_th1 = torch.from_numpy(t_records1).float().to(self.device)
        t_records_th2 = torch.from_numpy(t_records2).float().to(self.device)
        t_records_th3 = torch.from_numpy(t_records3).float().to(self.device)
        t_records_th = (t_records_th1, t_records_th2, t_records_th3)

        e_idx_l_th = torch.from_numpy(e_idx_l).long().to(self.device)
        edge_embeddings = self.edge_raw_embed[e_idx_l_th]
        src_idx_th = torch.from_numpy(src_idx_l).long().to(self.device)
        src_raw_embed = self.node_raw_embed[src_idx_th]

        assert self.num_layers == 3, self.num_layers
        final_node_embeddings, attn_output_weights = self.forward_msg_walk(src_raw_embed, edge_embeddings,
                                                                           hidden_embeddings, edge_features,
                                                                           t_records_th, mask, self.num_layers)
        final_node_embedding = final_node_embeddings + src_raw_embed
        # final_node_embedding = final_node_embeddings

        # if case_study:
        #     # 把case_study结果写入文件
        #     # 保存格式： H@{} （s,r,?,t) answer@1 answer@2 answer@3 path1(edge),path1(t),path1(node) path2(edge),path2(t),path2(node)
        #     assert model_path != "" , "case_study保存路径有误！："+model_path

        return final_node_embedding, attn_output_weights

    def init_hidden_embeddings(self, src_idx_l, node_records):
        device = self.device
        node_records_th = torch.from_numpy(node_records).long().to(device)
        hidden_embeddings = self.node_raw_embed[node_records_th]
        masks = (node_records_th != 0).sum(dim=-1).long()
        return hidden_embeddings, masks

    def retrieve_edge_features(self, eidx_records):
        device = self.device
        eidx_records_th = torch.from_numpy(eidx_records).long().to(device)
        # eidx_records_th[:, :, 0] = 0
        edge_features = self.edge_raw_embed[eidx_records_th]
        # edge_features[:, :, 0, :] = self.start_edge_embed
        return edge_features

    def forward_msg_walk(self, src_raw_embed, edge_embeddings, hidden_embeddings, edge_features, t_records_th, mask,
                         num_layers):
        return self.walk_encoder.forward_one_node(src_raw_embed, edge_embeddings, hidden_embeddings, edge_features,
                                                  t_records_th, mask, num_layers)

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder


class WalkEncoder(nn.Module):
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, device, n_head=3, dropout_p=0.1,
                 path_encode="GRU_time"):

        super(WalkEncoder, self).__init__()

        self.path_encode = path_encode
        self.device = device
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim
        self.n_head = n_head
        self.out_dim = out_dim
        self.vdim = feat_dim + pos_dim
        self.dropout_p = dropout_p
        self.logger = logger
        self.n_head = n_head
        pos_feat1 = nn.Parameter(torch.Tensor(1, self.pos_dim))
        pos_feat2 = nn.Parameter(torch.Tensor(1, self.pos_dim))
        pos_feat3 = nn.Parameter(torch.Tensor(1, self.pos_dim))
        nn.init.xavier_uniform_(pos_feat1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(pos_feat2, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(pos_feat3, gain=nn.init.calculate_gain('relu'))
        self.pos_feat1 = pos_feat1
        self.pos_feat2 = pos_feat2
        self.pos_feat3 = pos_feat3

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.device, self.dropout_p,
                                              encode_method=self.path_encode)
        self.projector = nn.Sequential(nn.Linear(self.feature_encoder.hidden_dim,
                                                 self.attn_dim), nn.GELU(), nn.Dropout(self.dropout_p))

        self.pooler = SetPooler(n_features=self.attn_dim, out_features=self.out_dim, dropout_p=self.dropout_p,
                                n_head=self.n_head, kdim=self.attn_dim, vdim=self.vdim)

    def forward_one_node(self, src_raw_embed, edge_embeddings, hidden_embeddings, edge_features, t_records, mask,
                         num_layers):

        edge_features1, edge_features2, edge_features3 = edge_features
        hidden_embeddings1, hidden_embeddings2, hidden_embeddings3 = hidden_embeddings
        t_records1, t_records2, t_records3 = t_records
        combined_features1 = edge_features1
        combined_features2 = edge_features2
        combined_features3 = edge_features3
        # combined_features1 = torch.cat([hidden_embeddings1, edge_features1], dim=-1)
        # combined_features2 = torch.cat([hidden_embeddings2, edge_features2], dim=-1)
        # combined_features3 = torch.cat([hidden_embeddings3,edge_features3], dim=-1)
        end_node_embedding1 = hidden_embeddings1[:, :, -1, :]
        end_node_embedding2 = hidden_embeddings2[:, :, -1, :]
        end_node_embedding3 = hidden_embeddings3[:, :, -1, :]
        pos_embed1 = self.pos_feat1.unsqueeze(0).repeat(end_node_embedding1.shape[0], end_node_embedding1.shape[1],
                                                        1)
        pos_embed2 = self.pos_feat2.unsqueeze(0).repeat(end_node_embedding2.shape[0], end_node_embedding2.shape[1],
                                                        1)
        pos_embed3 = self.pos_feat3.unsqueeze(0).repeat(end_node_embedding3.shape[0], end_node_embedding3.shape[1],
                                                        1)
        end_node_embedding1 = torch.cat([end_node_embedding1, pos_embed1], dim=-1)
        end_node_embedding2 = torch.cat([end_node_embedding2, pos_embed2], dim=-1)
        end_node_embedding3 = torch.cat([end_node_embedding3, pos_embed3], dim=-1)
        combined_features1 = self.feature_encoder.integrate(t_records1, combined_features1)
        combined_features2 = self.feature_encoder.integrate(t_records2, combined_features2)
        combined_features3 = self.feature_encoder.integrate(t_records3, combined_features3)
        combined_features = torch.cat([combined_features1, combined_features2, combined_features3], dim=1)
        end_node_embedding = torch.cat([end_node_embedding1, end_node_embedding2, end_node_embedding3], dim=1)
        mask1, mask2, mask3 = mask
        mask01 = np.concatenate((mask1, mask2, mask3), axis=-1)
        key_padding_mask = torch.from_numpy((mask01 == 0)).to(self.device)

        x = self.projector(combined_features)
        # query_embedding = torch.cat([src_raw_embed, edge_embeddings], dim=1)
        query_embedding = edge_embeddings
        # 最后一层 池化输出
        x, attn_output_weights = self.pooler(query_embedding, x, end_node_embedding, key_padding_mask, agg='attention')

        return x, attn_output_weights


class FeatureEncoder(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, in_features, hidden_features, device, dropout_p=0.1,
                 encode_method="GRU_time"):
        super(FeatureEncoder, self).__init__()
        self.hidden_dim = hidden_features
        self.device = device
        self.encode_method = encode_method
        if self.hidden_dim == 0:
            return
        self.dropout = nn.Dropout(dropout_p)

        self.gru = GRUCell(in_features, hidden_features)
        # LSTM
        self.lstm = torch.nn.LSTM(in_features, hidden_features, num_layers=1, batch_first=True, dropout=0.1)
        # transformer
        self.transformer = torch.nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_features, nhead=1, dropout=0.1), num_layers=1)
        # Time Encoder
        self.time_encoder = TimeEncode(dimension=in_features)
        self.linear = nn.Sequential(nn.Linear(in_features * 2, in_features), nn.ReLU(), self.dropout)

    def integrate(self, t_records, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch * n_walk, len_walk, feat_dim)
        t_records = t_records.view(batch * n_walk, len_walk, 1)
        if "time" in self.encode_method:
            t_interval = t_records - t_records[:, 0, :].unsqueeze(dim=-1)
            t_embedding = self.time_encoder(t_interval).squeeze()
            X = X + t_embedding
            h = X[:, 0, :]
        else:
            h = X[:, 0, :]
        # GRU
        if "GRU" in self.encode_method:
            for i in range(X.shape[1] - 1):
                h = self.gru(X[:, i + 1, :], h)
        elif "LSTM" in self.encode_method:
            # LSTM (batch_size, seq_len, input_size) -> (batch_size, seq_len, hidden_size)
            lstm_out, (hn, cn) = self.lstm(X)  # lstm_out 是所有时间步的输出，hn 是最后一个时间步的隐藏状态
            h = lstm_out[:, -1, :]  # 获取最后一个时间步的隐藏状态
        else:
            # transformer
            X = X.permute(1, 0, 2)  # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
            # Transformer 编码器
            X = self.transformer(X)
            X = X.permute(1, 0, 2)  # (seq_len, batch_size, input_size) -> (batch_size, seq_len, input_size)
            # 取序列的最后一个输出作为预测结果
            h = X[:, -1, :]  # 获取最后一个时间步的输出

        # instantaneous activation
        # encoded_features = self.gru(X[:, -1, :], h)
        encoded_features = h
        encoded_features = encoded_features.view(batch, n_walk, self.hidden_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features

    def forward(self, s, state):
        t0, t1, x = state
        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0
        dx = self.odefun(t, x)
        dx = dx * ratio
        return torch.zeros_like(t0), torch.zeros_like(t1), dx


class SetPooler(nn.Module):
    def __init__(self, n_features, out_features, dropout_p=0.1, n_head=3, kdim=600, vdim=600):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        # self.act = torch.nn.ReLU()
        self.act = torch.nn.GELU()
        self.multi_head_target = nn.MultiheadAttention(embed_dim=n_features, kdim=kdim, vdim=vdim,
                                                       num_heads=n_head,
                                                       dropout=dropout_p)

    def forward(self, edge_embeddings, X, end_node_embedding, key_padding_mask, agg='mean'):
        # if agg == 'sum':
        #     return self.out_proj(X.sum(dim=-2)), None
        # elif agg == 'mean':
        #     assert (agg == 'mean')
        #     return self.out_proj(X.mean(dim=-2)), None
        # elif agg == 'attention':
        edge_embeddings_unrolled = torch.unsqueeze(edge_embeddings, dim=1)
        query = edge_embeddings_unrolled
        key = X
        query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]
        value = end_node_embedding.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]
        attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=value, key_padding_mask=key_padding_mask)
        attn_output = attn_output.squeeze(dim=0)
        return attn_output, attn_output_weights.squeeze(dim=1)
        # else:
        #     assert 'agg is not defined'

class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)
        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))
        return output
