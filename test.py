from helper_function import data
from helper_function import preprocessing
from helper_function import aug_bt
from helper_function import metrics

import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
tqdm.pandas() # progress

import warnings
warnings.filterwarnings('ignore')


# train.py start
print('\n== start test.py ==\n')

# 데이터 로드
df = data.load_data(r'C:\Project\sw-grad-proj\data\ratings_test.txt')
print(f'df shape : {df.shape}')

# X, y
X = df['document'][:5]
y = df['label'][:5]
print(f'X, y shape : {X.shape}, {y.shape}\n')

def test(X, y):

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # == te preprocessing ==
    X['document'] = preprocessing.drop_duplicates(X['document'])
    X['document'] = preprocessing.drop_null(X['document'])
    X['document'] = X['document'].apply(lambda x : preprocessing.text_cleansing(x))
    X['document'] = X['document'].apply(lambda x : preprocessing.text_tokenize(x))
    X['document'] = X['document'].apply(lambda x : preprocessing.del_stopwords(x))
    X_te_fin = preprocessing.best_encoding_tf(X['document'])
    print('Done. (te preprocessing) \n')

    # == load best model ==
    load_clf = pickle.load(open(r'C:\Project\sw-grad-proj\result\best_model.pkl', 'rb'))
    print('Done. (load model)')

    # == predict ==
    pred = load_clf.predict(X_te_fin)
    acc, f1 = metrics.metrics(y, pred)
    print(f'te acc : {acc}')
    print(f'te f1 : {f1}')
    print('Done. (predict) \n')

    return


''' sample '''
test(X, y)