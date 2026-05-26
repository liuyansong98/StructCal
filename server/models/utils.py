import numpy as np
import torch
import os
import random
import argparse
import sys
from torch import nn as nn
import pandas as pd
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser('2025')
    # server
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--port", type=int, default=6001, help="Port number for the server")

    # General
    parser.add_argument('-d', '--data', type=str,
                        default='ICEWS14s_divide',
                        help='dataset to use')
    parser.add_argument('-c', '--model_path', type=str,
                        default='../server/model_ckpts/ICEWS14s_divide/best-model.pth',
                        help='best checkpoint path of the model')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the GPU to be used')

    # Model-related
    parser.add_argument('--n_degree', nargs='*', default=['64', '16', '4'],
                        help='a list of neighbor sampling numbers for different hops, '
                             'when only a single element is input n_layer will be activated')
    parser.add_argument('--n_head', type=int, default=3,
                        help='number of heads for attention pooling')
    parser.add_argument('--pos_dim', type=int, default=60,
                        help='dimension of the positional encoding')
    parser.add_argument('--embed_dim', type=int, default=600,
                        help='dimension of the node and relation embedding')
    parser.add_argument('--temporal_bias', default=0.01, type=float,
                        help='temporal_bias')
    parser.add_argument('--limit_ngh_span', action='store_true',
                        help="whether to limit the maximum number of spanned temporal neighbors")
    parser.add_argument('--ngh_span', nargs='*', default=['2048', '16'],
                        help='a list of maximum number of spanned temporal neighbors for different hops')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption)'
                             'cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--path_encode', type=str, default='GRU_time',
                        choices=['LSTM', 'GRU', 'Transformer', 'LSTM_time', 'GRU_time',
                                 'Transformer_time'], help='the path encoder to be used')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args, sys.argv


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors
        num_layers = 1
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers


def get_embedding(num_embeddings, embedding_dims, zero_init=False, device=None):
    if type(embedding_dims) == int:
        embedding_dims = [embedding_dims]
    if zero_init:
        embed = nn.Parameter(torch.zeros(num_embeddings, *embedding_dims))
        # embed = torch.Tensor(num_embeddings, *embedding_dims)
    else:
        embed = nn.Parameter(torch.Tensor(num_embeddings, *embedding_dims))
        # embed = torch.Tensor(num_embeddings, *embedding_dims)
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
    if device is not None:
        embed = embed.to(device)

    return embed

def load_temporal_knowledge_graph(DATA_ROOT, dataset_name):
    train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
    column_names = ['head', 'rel', 'tail', 'time', '_']
    train_data_table = load_data_table(DATA_ROOT, dataset_name, train_file, column_names)
    val_data_table = load_data_table(DATA_ROOT, dataset_name, val_file, column_names)
    test_data_table = load_data_table(DATA_ROOT, dataset_name, test_file, column_names)
    all_data_table = pd.concat([train_data_table, val_data_table, test_data_table], ignore_index=True)
    eidx = np.arange(len(all_data_table), dtype=int)
    stat_table = load_data_table(DATA_ROOT, dataset_name, "stat.txt", column_names=['num_entities', 'num_relations', '_'])
    num_entities, num_relations = stat_table['num_entities'].item(), stat_table['num_relations'].item()

    all_heads = all_data_table['head'].to_numpy()
    all_tails = all_data_table['tail'].to_numpy()
    all_rels = all_data_table['rel'].to_numpy()
    all_timestamps = all_data_table['time'].to_numpy()
    edge_idxs = eidx

    return all_heads, all_tails, all_rels, all_timestamps, edge_idxs, num_entities, num_relations


def load_data_table(DATA_ROOT, graph_name, file_name, column_names=None):
    data_fpath = os.path.join(DATA_ROOT, graph_name, file_name)
    return pd.read_table(data_fpath, sep='\t', names=column_names)
