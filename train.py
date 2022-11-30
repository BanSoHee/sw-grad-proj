from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from helper_function import data
from helper_function import preprocessing
from helper_function import aug_bt

import pandas as pd
import numpy as np

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

        # tr aug
        out_en = X_tr.iloc[:5].progress_apply(lambda x : aug_bt.BT_ko2en(x)) # iloc 삭제
        out_en = out_en.apply(lambda x : aug_bt.BT_en2ko(x))
        print('Done. (aug en)')
        '''
        out_jp = X_tr.iloc[:5].progress_apply(lambda x : aug_bt.BT_ko2jp(x)) # iloc 삭제
        out_jp = out_jp.apply(lambda x : aug_bt.BT_jp2ko(x))
        print('Done. (aug jp)')
        '''
        print('Done. (aug) \n')
        
        # tr concat : origin + aug
        X_tr_aug = pd.concat([X_tr.iloc[:5], out_en], ignore_index=True) # iloc 삭제
        print('Done. (concat) \n')  

        # tr preprocessing
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