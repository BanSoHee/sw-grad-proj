from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from helper_function import data
from helper_function import preprocessing
from helper_function import aug_bt

import pandas as pd
import numpy as np
import re

from tqdm import tqdm
tqdm.pandas() # progress


print('\n== start train.py ==\n')

# 데이터 로드
df = data.load_data(r'C:\Project\sw-grad-proj\data\ratings_train.txt')
print(f'df shape : {df.shape}')

# X, y
# df.drop(columns=['id'], inplace=True)
X = df['document']
y = df['label']
print(f'X, y shape : {X.shape}, {y.shape}\n')


def train(X, y):

    print('\n== stratified 5-fold start==\n')

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    k = kfold.get_n_splits(X, y)
    print(f'split k : {k}')
    cnt_kfold = 1

    # k-fold idx
    for tr_idx, val_idx in kfold.split(X, y):

        # k-fold
        print(f'\n== K-FOLD {cnt_kfold} ==\n')
        print(f'TRAIN : {tr_idx}')
        print(f'VALID : {val_idx}')

        # split
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        print('Done. (split) \n')

        ''' del '''
        X_tr = X_tr.iloc[:2]
        X_val = X_val.iloc[:2]
        y_tr = y_tr.iloc[:2]
        y_val = y_val.iloc[:2]

        # tr aug
        out_en = X_tr.progress_apply(lambda x : aug_bt.BT_ko2en(x))
        out_en = out_en.apply(lambda x : aug_bt.BT_en2ko(x))
        print('Done. (aug en)')
        '''
        out_jp = X_tr.progress_apply(lambda x : aug_bt.BT_ko2jp(x))
        out_jp = out_jp.apply(lambda x : aug_bt.BT_jp2ko(x))
        print('Done. (aug jp)')
        '''
        print('Done. (aug)')
        
        # tr concat : origin + aug
        X_tr_aug = pd.concat([X_tr, out_en], ignore_index=True)
        print('Done. (concat)')
        
        # tr preprocessing
        X_tr_aug = preprocessing.drop_duplicates(X_tr_aug)
        print('Done. (drop duplicates)')

        X_tr_aug = preprocessing.drop_null(X_tr_aug)
        print('Done. (drop null)')

        X_tr_aug = X_tr_aug.apply(lambda x : preprocessing.text_cleansing(x))
        print('Done. (text cleansing)')
        print(X_tr_aug)
        
        break

        # te preprocessing

        # train

        # eval

        # save best model

        cnt_kfold += 1

    return

def valid(val):

    return


train(X, y)