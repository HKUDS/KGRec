import setproctitle
import random
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time
from prettytable import PrettyTable
import datetime
from utils.parser import parse_args_kgcl as parse_args
from utils.data_loader_kgcl import load_data, generate_kg_batch
from modules.KGCL.KGCL import KGCL
from utils.evaluator import Evaluator
from utils.helper import early_stopping, init_logger
from logging import getLogger
from utils.sampler import UniformSampler
from collections import defaultdict

seed = 2020
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

sampling = UniformSampler(seed)

setproctitle.setproctitle('EXP@KGCL')

def neg_sampling_cpp(train_cf_pairs, train_user_dict):
    time1 = time()
    train_cf_negs = sampling.sample_negative(train_cf_pairs[:, 0], n_items, train_user_dict, 1)
    train_cf_negs = np.asarray(train_cf_negs)
    train_cf_triples = np.concatenate([train_cf_pairs, train_cf_negs], axis=1)
    time2 = time()
    logger.info('neg_sampling_cpp time: %.2fs', time2 - time1)
    logger.info('train_cf_triples shape: {}'.format(train_cf_triples.shape))
    return train_cf_triples

def get_feed_dict(train_cf_with_neg, start, end):
    feed_dict = {}
    entity_pairs = torch.from_numpy(train_cf_with_neg[start:end]).to(device).long()
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = entity_pairs[:, 2]
    feed_dict['batch_start'] = start
    return feed_dict

if __name__ == '__main__':
    try:
        """fix the random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        """read args"""
        global args, device
        args = parse_args()
        device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

        log_fn = init_logger(args)
        logger = getLogger()
        
        logger.info('PID: %d', os.getpid())
        logger.info(f"DESC: {args.desc}\n")

        """build dataset"""
        train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat = load_data(args)

        n_users = n_params['n_users']
        n_items = n_params['n_items']
        n_entities = n_params['n_entities']
        n_relations = n_params['n_relations']
        n_nodes = n_params['n_nodes']

        # test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

        """define model"""
        model = KGCL(n_params, args, graph, adj_mat).to(device)
        model.print_shapes()
        """define optimizer"""
        rec_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        kg_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        evaluator = Evaluator(args)

        test_interval = 1 if args.dataset == 'last-fm' else 1
        early_stop_step = 10 if args.dataset == 'last-fm' else 10

        cur_best_pre_0 = 0
        cur_stopping_step = 0
        should_stop = False

        logger.info("start training ...")
        for epoch in range(args.epoch):
            """training CF"""
            """cf data"""
            train_cf_with_neg = neg_sampling_cpp(train_cf, user_dict['train_user_set'])
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_with_neg = train_cf_with_neg[index]

            """training cf"""
            aug_views = model.get_aug_views()
            model.train()
            add_loss_dict, s = defaultdict(float), 0
            train_s_t = time()
            with tqdm(total=len(train_cf)//args.batch_size) as pbar:
                while s + args.batch_size <= len(train_cf):
                    batch = get_feed_dict(train_cf_with_neg,
                                        s, s + args.batch_size)
                    batch['aug_views'] = aug_views
                    batch_loss, batch_loss_dict = model(batch)

                    rec_optimizer.zero_grad(set_to_none=True)
                    batch_loss.backward()
                    rec_optimizer.step()

                    for k, v in batch_loss_dict.items():
                        add_loss_dict[k] += v / len(train_cf)
                    s += args.batch_size
                    pbar.update(1)
            train_e_t = time()

            """training kg"""
            time3 = time()
            kg_total_loss = 0
            n_kg_batch = n_params['n_triplets'] // 4096
            for iter in tqdm(range(1, n_kg_batch + 1)):
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(kg_dict, 4096, n_params['n_entities'])
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)

                kg_batch_loss = model.calc_kg_loss_transE(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

                kg_optimizer.zero_grad(set_to_none=True)
                kg_batch_loss.backward()
                kg_optimizer.step()
                kg_total_loss += kg_batch_loss.item()

            logger.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))


            if epoch % test_interval == 0 and epoch >= 0:
                """testing"""
                test_s_t = time()
                model.eval()
                with torch.no_grad():
                    ret = evaluator.test(model, user_dict, n_params)
                test_e_t = time()

                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, list(add_loss_dict.values()), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
                )
                logger.info(train_res)

                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                cur_best_pre_0, cur_stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                            cur_stopping_step, expected_order='acc',
                                                                            flag_step=early_stop_step)
                if cur_stopping_step == 0:
                    logger.info("###find better!")
                elif should_stop:
                    break

                """save weight"""
                if ret['recall'][0] == cur_best_pre_0 and args.save:
                    save_path = args.out_dir + log_fn + '.ckpt'
                    logger.info('save better model at epoch %d to path %s' % (epoch, save_path))
                    torch.save(model.state_dict(), save_path)

            else:
                # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
                logger.info('{}: using time {:.1f}, training loss at epoch {}: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_e_t - train_s_t, epoch, list(add_loss_dict.values())))

        logger.info('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))

    except Exception as e:
        logger.exception(e)
