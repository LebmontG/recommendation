# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:49:26 2022
?
@author: Lebmont
"""

import numpy as np
import pandas as pd
import gc
import os
import sys
#import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbole.utils import InputType, FeatureType
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
#from recbole.model.sequential_recommender import DIEN,DIN
from recbole.model.general_recommender import NNCF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_topk

ar=pd.read_csv('b.csv',dtype={'article_id': str})
cu=pd.read_csv('c.csv')
tr=pd.read_csv(r"a.csv",dtype={'article_id': str})
#!mkdir /kaggle/working/recbox_data

cu=cu[['customer_id','age',
       'club_member_status','fashion_news_frequency']]
cu.age=cu.age.round().fillna(cu.age.mean())
cu.fashion_news_frequency=\
cu.fashion_news_frequency.fillna\
(cu.fashion_news_frequency.value_counts().index[0])
cu.club_member_status=cu.club_member_status.fillna\
(cu.club_member_status.value_counts().index[0])
cu=cu.rename(columns={'customer_id': 'user_id',
                   'fashion_news_frequency': 'fre',
                   'club_member_status':'club'})
#cu.to_csv('user.csv',index=False)

# temp=pd.get_dummies(ar, prefix=None, prefix_sep='_',
# dummy_na=False, columns=None, 
# sparse=False, drop_first=False, dtype=None)
ar=ar[['article_id',
       'product_type_name',
       'index_group_name',
       'garment_group_name',
       'product_group_name',
       'colour_group_name',
       'perceived_colour_master_name',
       'department_no',
       'index_code',
       'section_no']].rename(
           columns={'article_id': 'item_id',
            'product_type_name': 'pro_type',
            'product_group_name':'pro_group',
            'colour_group_name':'color1',
            'perceived_colour_master_name':'color2',
             'index_group_name': 'group',
             'department_no':'dep',
             'index_code':'index',
             'section_no':'sec',
             'garment_group_name':'gar'})
#ar.to_csv('item.csv',index=False)

tr=tr[['customer_id','t_dat','article_id']].rename(
columns={'customer_id':'user_id',
          't_dat':'timestamp',
          'article_id':'item_id'})
#tr.to_csv('item.csv',index=False)

class recommendation():
    def __init__(self,user,item):
        self.item_dict=dict()
        self.user_dict=dict()
        self.item_enc=OneHotEncoder(handle_unknown='ignore')
        self.user_enc=OneHotEncoder(handle_unknown='ignore')
        self.user_enc.fit(user[[co for co in user][1:]])
        for i in range(len(user)):
            self.user_dict[user.user_id[i]]=\
            self.user_enc.transform([user.loc[i][1:]]).toarray()
        self.item_enc.fit(item[[co for co in item][1:]])
        for i in range(len(item)):
            self.item_dict[item.item_id[i]]=\
            self.item_enc.transform([item.loc[i][1:]]).toarray()
        self.emb_size=sum([len(self.user_enc.categories_[i])\
            for i in range(len(self.user_enc.categories_))])\
            +sum([len(self.item_enc.categories_[i])\
            for i in range(len(self.item_enc.categories_))])
        #self.nnin_size=10
        #self.nnin_size=2*int(1+np.log(self.emb_size)/np.log(2))
        self.nn=torch.nn.Sequential(
            torch.nn.Linear(self.emb_size,256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8,1),
            torch.nn.Sigmoid())
        self.batchsize=256
        self.loss=torch.nn.BCELoss()
        self.optimizer=torch.optim.Adam(self.nn.parameters())
        return
    def query(self,v,pre=False):
        if pre:
            inp=np.zeros((1,self.emb_size))[0]
            for (i,j) in self.item_dict.items():
                it=np.concatenate((v,j),1)
                inp=np.concatenate((inp,it))
            output=self.nn(inp[1:])
            return torch.topk(output,k=12,dim=0)
        res=[]
        for i in range(len(v)):
            it=self.item_dict[v.loc[i][2]][0]
            us=self.user_dict[v.loc[i][0]][0]
            #us=np.hstack((us,[float(v.loc[i][3])]))
            res.append(np.hstack((us,it)))
        return torch.FloatTensor(np.array(res))
    def forward(self,trans):
        sit=0
        res=[]
        l=len(trans)
        while (sit+self.batchsize)<=l:
            batch=trans.loc[sit:sit+self.batchsize]
            sit+=self.batchsize
            prob=self.nn(self.query(batch))
            res.append(prob)
        prob=self.nn(self.query(trans.loc[sit:]))
        res.append(prob)
        return res
    def optimize(self,train):
        sit=0
        l=len(train)
        while (sit+self.batchsize)<=l:
            batch=train.loc[sit:sit+self.batchsize]
            sit+=self.batchsize
            output=self.nn(self.query(batch))
            label=torch.ones((output.size()[0],output.size()[1]))
            loss = self.loss(output, label)
            loss.backward()
            self.optimizer.step()
        output=self.nn(self.query(train.loc[sit:]))
        label=torch.ones((output.size()[0],output.size()[1]))
        loss= self.loss(output, label)
        loss.backward()
        self.optimizer.step()
        return
    def predict(self,ar):
        res=[]
        for (u,v) in self.user_dict.items():
            _,index=self.query(v,pre=True)
            it=ar.loc[list(np.array(index)[0])]
            res.append(list(it.iloc[:,0].values))
        return res

rec=recommendation(cu,ar)
rec.optimize(tr)
#a=rec.query(tr)
b=rec.forward(tr)
e=rec.predict(ar)





