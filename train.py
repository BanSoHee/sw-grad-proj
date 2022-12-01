from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from helper_function import data
from helper_function import preprocessing
from helper_function import aug_bt
from helper_function import metrics

from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
import re
import pickle

from tqdm import tqdm
tqdm.pandas() # progress

import warnings
warnings.filterwarnings('ignore')


# train.py start
print('\n== start train.py ==\n')

# 데이터 로드
df = data.load_data(r'C:\Project\sw-grad-proj\data\ratings_train.txt')
print(f'df shape : {df.shape}')

# X, y
X = df['document']
y = df['label']
print(f'X, y shape : {X.shape}, {y.shape}\n')

# model list
model_list = []


def train(X, y):

    print('\n== stratified 5-fold start==\n')

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    k = kfold.get_n_splits(X, y)
    print(f'split k : {k}')
    cnt_kfold = 1
    best_acc, best_f1 = 0, 0 # to save best model

    # == k-fold idx ==
    for tr_idx, val_idx in kfold.split(X, y):

        # k-fold
        print(f'\n== K-FOLD {cnt_kfold} ==\n')
        print(f'TRAIN : {tr_idx}')
        print(f'VALID : {val_idx}')

        # == split ==
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        X_tr, X_val, y_tr, y_val = pd.DataFrame(X_tr), pd.DataFrame(X_val), pd.DataFrame(y_tr), pd.DataFrame(y_val)
        print('Done. (split) \n')

        ''' del '''
        X_tr = X_tr.iloc[:2]
        X_val = X_val.iloc[:2]
        y_tr = y_tr.iloc[:2]
        y_val = y_val.iloc[:2]
        ''' del '''

        # == tr aug ==
        out_en = X_tr.copy()
        out_en['document'] = X_tr['document'].progress_apply(lambda x : aug_bt.BT_ko2en(x))
        out_en['document'] = out_en['document'].apply(lambda x : aug_bt.BT_en2ko(x))
        out_en_y = y_tr.copy()
        print('Done. (aug)')
        
        # tr concat : origin + aug
        X_tr_aug = pd.concat([X_tr, out_en], ignore_index=True)
        y_tr_fin = pd.concat([y_tr, out_en_y], ignore_index=True)
        print('Done. (concat)')
        
        # == tr preprocessing ==
        X_tr_aug['document'] = preprocessing.drop_duplicates(X_tr_aug['document']) # 데이터 중복 제거
        X_tr_aug['document'] = preprocessing.drop_null(X_tr_aug['document']) # 결측치 제거
        X_tr_aug['document'] = X_tr_aug['document'].apply(lambda x : preprocessing.text_cleansing(x)) # 텍스트 킄렌징
        X_tr_aug['document'] = X_tr_aug['document'].apply(lambda x : preprocessing.text_tokenize(x))  # 토큰화
        X_tr_aug['document'] = X_tr_aug['document'].apply(lambda x : preprocessing.del_stopwords(x))  # 불용어 제거
        X_tr_fin = preprocessing.encoder_tf(X_tr_aug['document']) # create X_tr_fin & fit_transform tf-idf encoder
        print('Done. (tr preprocessing)')

        # == val preprocessing ==
        X_val['document'] = preprocessing.drop_duplicates(X_val['document'])
        X_val['document'] = preprocessing.drop_null(X_val['document'])
        X_val['document'] = X_val['document'].apply(lambda x : preprocessing.text_cleansing(x))
        X_val['document'] = X_val['document'].apply(lambda x : preprocessing.text_tokenize(x))
        X_val['document'] = X_val['document'].apply(lambda x : preprocessing.del_stopwords(x))
        X_val_fin = preprocessing.encoding_tf(X_val['document'])
        print('Done. (val preprocessing) \n')

        # == train model ==
        clf = LGBMClassifier()
        clf.fit(X_tr_fin, y_tr_fin) # , callbacks=[tqdm_callback])

        # == eval moel ==
        pred = clf.predict(X_val_fin)
        acc, f1 = metrics.metrics(y_val, pred)
        print(f'tr acc : {acc}')
        print(f'tr f1 : {f1}')
        print('Done. (train/eval model) \n')

        # == check best model == 
        if acc > best_acc:
            # == save best model and vectorizer ==
            best_acc = acc
            best_model = clf
            pickle.dump(best_model, open(r'C:\Project\sw-grad-proj\result\best_model.pkl', 'wb')) # save best model
            preprocessing.save_encoder_tf(X_tr_aug['document']) # save best encoder
            
        cnt_kfold += 1

    # == load best model ==
    # load_model = pickle.load(open(r'C:\Project\sw-grad-proj\result\best_model.pkl', 'rb'))
    # X_te_fin = preprocessing.best_encoding_tf(X_te['document'])

    return


''' sample '''
train(X, y)