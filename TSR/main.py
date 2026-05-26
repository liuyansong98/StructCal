import numpy as np
import pandas as pd
from log import *
from utils import *
from train import *
from module import myModel
from graph import NeighborFinder
import resource
from typing import Dict


def main():
    args, sys_argv = get_args()
    assert (args.cpu_cores >= -1)
    set_random_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    heads, tails, rels, timestamps, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(args.data)
    all_nodes_l = np.arange(0, num_entities)
    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

    timestamps_discr = []
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(timestamps)):
            timestamps_discr.append(timestamps[i] // 3600)
        val_time, test_time, end_time = list(np.quantile(timestamps_discr, [0.8, 0.9, 1]))
        valid_train_flag = (timestamps_discr <= val_time)
        valid_val_flag = (timestamps_discr <= test_time) * (timestamps_discr > val_time)
        valid_test_flag = (timestamps_discr > test_time) * (timestamps_discr <= end_time)
        full_data_flag = timestamps_discr <= end_time
    else:
        val_time, test_time, end_time = list(np.quantile(timestamps, [0.8, 0.9, 1]))
        valid_train_flag = (timestamps <= val_time)
        valid_val_flag = (timestamps <= test_time) * (timestamps > val_time)
        valid_test_flag = (timestamps > test_time) * (timestamps <= end_time)
        full_data_flag = timestamps <= end_time

    train_src_l, train_dst_l, train_ts_l, train_e_idx_l = \
        heads[valid_train_flag], tails[valid_train_flag], timestamps[valid_train_flag], \
            rels[valid_train_flag]
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l = \
        heads[valid_val_flag], tails[valid_val_flag], timestamps[valid_val_flag], \
            rels[valid_val_flag]
    test_src_l, test_dst_l, test_ts_l, test_e_idx_l = \
        heads[valid_test_flag], tails[valid_test_flag], timestamps[valid_test_flag], \
            rels[valid_test_flag]

    heads_, tails_, timestamps_, rels_ = \
        heads[full_data_flag], tails[full_data_flag], timestamps[full_data_flag], \
            rels[full_data_flag]

    val_ts_l_discr = []
    val_bs_idx = [0]
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(val_ts_l)):
            val_ts_l_discr.append(val_ts_l[i] // 3600)
        for k in range(0, len(val_ts_l_discr) - 1):
            if val_ts_l_discr[k] != val_ts_l_discr[k + 1]:
                val_bs_idx.append(k + 1)
        val_bs_idx.append(len(val_ts_l_discr))
    else:
        for k in range(0, len(val_ts_l) - 1):
            if val_ts_l[k] != val_ts_l[k + 1]:
                val_bs_idx.append(k + 1)
        val_bs_idx.append(len(val_ts_l))

    test_ts_l_discr = []
    test_bs_idx = [0]
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(test_ts_l)):
            test_ts_l_discr.append(test_ts_l[i] // 3600)
        for k in range(0, len(test_ts_l_discr) - 1):
            if test_ts_l_discr[k] != test_ts_l_discr[k + 1]:
                test_bs_idx.append(k + 1)
        test_bs_idx.append(len(test_ts_l_discr))
    else:
        for k in range(0, len(test_ts_l) - 1):
            if test_ts_l[k] != test_ts_l[k + 1]:
                test_bs_idx.append(k + 1)
        test_bs_idx.append(len(test_ts_l))

    train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l
    val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l
    train_val_data = (train_data, val_data)

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads_, tails_, rels_, timestamps_):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))

    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=args.temporal_bias,
                                     limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span,
                                     num_entity=num_entities, data_name=args.data)

    logger.info('Sampling module - temporal bias: {}'.format(args.temporal_bias))
    """init dynamic entity embeddings"""
    init_dynamic_entity_embeds = get_embedding(num_entities, args.embed_dim, zero_init=False)

    """init relation embeddings"""
    init_dynamic_relation_embeds = get_embedding(num_relations * 2, args.embed_dim, zero_init=False)

    model = myModel(n_feat=init_dynamic_entity_embeds, e_feat=init_dynamic_relation_embeds, device=args.device,
                    pos_dim=args.pos_dim, num_layers=args.n_layer, num_neighbors=args.n_degree,
                    solver=args.solver, step_size=args.step_size, drop_out=args.drop_out,
                    get_checkpoint_path=get_checkpoint_path,
                    n_head=args.n_head, path_encode=args.path_encode).to(device)

    # 重继续训练加载
    # model.load_state_dict(torch.load("./saved_checkpoints/1744786925.109515-ICEWS18_divide-t-3-64k16k4-60/best_checkpoint.pth", map_location=device))
    # model.update_ngh_finder(full_ngh_finder)
    # model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopMonitor(tolerance=args.tolerance)

    train_val(train_val_data, all_nodes_l, model, args, optimizer,
              early_stopper, full_ngh_finder, logger, num_entities, num_relations)

    model.ngh_finder.init_node_degree()
    model.ngh_finder.update_node_degree(train_src_l, train_dst_l)
    model.ngh_finder.update_node_degree(val_src_l, val_dst_l)
    model.update_ngh_finder(full_ngh_finder)
    if "social" in args.data:
        args.bs = 1
    logger.info('Start testing...')


    mrr = eval_one_epoch(model, all_nodes_l, test_src_l, test_dst_l, test_ts_l,
                         test_e_idx_l, test_bs_idx, args, logger,
                         num_entities,
                         num_relations, is_need_filter=True,
                         stage='test')

    logger.info('Saving model...')
    torch.save(model.state_dict(), best_model_path)
    logger.info('Saved model to {}'.format(best_model_path))
    logger.info('model saved')


if __name__ == "__main__":
    main()
