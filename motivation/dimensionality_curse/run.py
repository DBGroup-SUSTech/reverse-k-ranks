import argparse
import json

from recbole.quick_start import run_recbole
import ssl

if __name__ == '__main__':
    # This restores the same behavior as before.
    ssl._create_default_https_context = ssl._create_unverified_context
    model = 'ENMF'
    dataset_l = ['epinions', 'lastfm', 'pinterest']
    ebd_l = [4, 8, 16, 32, 64, 128, 256]
    # ebd_l = [2]
    res_l = {}
    for ebd in ebd_l:
        # config_dict = {'embedding_size': 64, 'epochs': 100, 'neg_sampling': None}
        # config_dict = {'embedding_size': 64, 'epochs': 100, 'topk': [10], 'valid_metric': 'MRR@10',
        #                'metrics': ['Recall', 'NDCG', 'Hit', 'Precision']}
        # config_dict = {
        #     'use_gpu': False, }
        for dataset in dataset_l:
            if dataset == 'lastfm':
                config_dict = {'embedding_size': ebd, 'epochs': 200, 'topk': [50, 100, 200], 'neg_sampling': None,
                               'use_gpu': True,
                               'train_batch_size': 32, 'learning_rate': 0.03, 'weight_decay': 0.5,
                               'learner': 'adagrad',
                               'eval_step': 20,
                               'valid_metric': 'hit@200',
                               'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
                               'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'RO',
                                             'mode': 'full'},
                               'USER_ID_FIELD': 'user_id',
                               'ITEM_ID_FIELD': 'artist_id',
                               'load_col': {'inter': ['user_id', 'artist_id']}
                               }
            elif dataset == 'ml-1m':  # dataset == 'ml-1m'
                config_dict = {'embedding_size': ebd, 'epochs': 200, 'topk': [50, 100, 200], 'neg_sampling': None,
                               'use_gpu': True,
                               'train_batch_size': 512, 'learning_rate': 0.05, 'weight_decay': 0.5,
                               'learner': 'adagrad',
                               'eval_step': 20,
                               'valid_metric': 'hit@200',
                               'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
                               'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'RO',
                                             'mode': 'full'}}
            elif dataset == 'epinions':
                config_dict = {'embedding_size': ebd, 'epochs': 200, 'topk': [50, 100, 200], 'neg_sampling': None,
                               'use_gpu': True,
                               'train_batch_size': 512, 'learning_rate': 0.1, 'weight_decay': 0.5,
                               'learner': 'adagrad',
                               'eval_step': 20,
                               'valid_metric': 'hit@200',
                               'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
                               'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'RO',
                                             'mode': 'full'},
                               'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']}}
            elif dataset == 'pinterest':
                config_dict = {'embedding_size': ebd, 'epochs': 200, 'topk': [50, 100, 200], 'neg_sampling': None,
                               'use_gpu': True,
                               'train_batch_size': 256, 'learning_rate': 0.03, 'weight_decay': 0.5,
                               'learner': 'adagrad',
                               'eval_step': 20,
                               'valid_metric': 'hit@200',
                               'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
                               'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'RO',
                                             'mode': 'full'}}

            res = run_recbole(model=model, dataset=dataset, config_dict=config_dict)
            with open('result/hitting_rate-{}-{}-new.json'.format(dataset, ebd), 'w') as f:
                json.dump(res, f)
            res_l[ebd] = res
    print(res_l)
