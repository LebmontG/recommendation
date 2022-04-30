#LebmontG
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from logging import getLogger
import gc
#pip install recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

article=pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv')
customer=pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv')
transaction=pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',dtype={'article_id': str})
samplesub=pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
!mkdir /kaggle/working/atomic
#article.info()
#customer.info()
#transaction.info()
#samplesub.info()

transaction.info()
transaction['t_dat'] = pd.to_datetime(transaction['t_dat'], format="%Y-%m-%d")
t=transaction[transaction['t_dat']>'2020-1-1']
#del transaction
t['timestamp'] = t.t_dat.values.astype(np.int64) // 10 ** 9
atomic=t[['customer_id', 'article_id', 'timestamp']]
#del t
atomic['customer_id'].value_counts().shape #862290 covered
atomic.rename(columns={'customer_id': 'user_id:token','article_id':'item_id:token','timestamp': 'timestamp:float'})
atomic.to_csv('./atomic.inter',index=False, sep='\\t')
para={
    'data_path': '/kaggle/working',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'user_inter_num_interval': "[30,inf)",
    'item_inter_num_interval': "[40,inf)",
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'neg_sampling': None,
    'epochs': 50,
    'eval_args': {
        'split': {'RS': [9, 0, 1]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'}
}
config = Config(model='GRU4Rec', dataset='atomic', config_dict=para)

# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()
# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# write config info into log
logger.info(config)

transaction['t_dat'] = pd.to_datetime(transaction['t_dat'])
transactions_3w = transaction[transaction['t_dat'] >= pd.to_datetime('2020-08-31')].copy()
transactions_2w = transaction[transaction['t_dat'] >= pd.to_datetime('2020-09-07')].copy()
transactions_1w = transaction[transaction['t_dat'] >= pd.to_datetime('2020-09-15')].copy()

purchase_dict_3w = {}
for i,x in enumerate(zip(transactions_3w['customer_id'], transactions_3w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_3w:
        purchase_dict_3w[cust_id] = {}
    if art_id not in purchase_dict_3w[cust_id]:
        purchase_dict_3w[cust_id][art_id] = 0
    purchase_dict_3w[cust_id][art_id] += 1
dummy_list_3w = list((transactions_3w['article_id'].value_counts()).index)[:12]

purchase_dict_2w = {}
for i,x in enumerate(zip(transactions_2w['customer_id'], transactions_2w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_2w:
        purchase_dict_2w[cust_id] = {}
    if art_id not in purchase_dict_2w[cust_id]:
        purchase_dict_2w[cust_id][art_id] = 0
    purchase_dict_2w[cust_id][art_id] += 1
dummy_list_2w = list((transactions_2w['article_id'].value_counts()).index)[:12]

purchase_dict_1w = {}
for i,x in enumerate(zip(transactions_1w['customer_id'], transactions_1w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_1w:
        purchase_dict_1w[cust_id] = {}
    if art_id not in purchase_dict_1w[cust_id]:
        purchase_dict_1w[cust_id][art_id] = 0
    purchase_dict_1w[cust_id][art_id] += 1
dummy_list_1w = list((transactions_1w['article_id'].value_counts()).index)[:12]

not_so_fancy_but_fast_benchmark = samplesub[['customer_id']]
prediction_list = []
dummy_list = list((transactions_1w['article_id'].value_counts()).index)[:12]
dummy_pred = ' '.join(dummy_list)
for i, cust_id in enumerate(samplesub['customer_id'].values.reshape((-1,))):
    if cust_id in purchase_dict_1w:
        l = sorted((purchase_dict_1w[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l)>12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l+dummy_list_1w[:(12-len(l))])
    elif cust_id in purchase_dict_2w:
        l = sorted((purchase_dict_2w[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l)>12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l+dummy_list_2w[:(12-len(l))])
    elif cust_id in purchase_dict_3w:
        l = sorted((purchase_dict_3w[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l)>12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l+dummy_list_3w[:(12-len(l))])
    else:
        s = dummy_pred
    prediction_list.append(s)
not_so_fancy_but_fast_benchmark['prediction'] = prediction_list
not_so_fancy_but_fast_benchmark.head()
not_so_fancy_but_fast_benchmark.to_csv('submission.csv', index=False)
#samplesub.to_csv(os.getcwd()+'\\result.csv')
