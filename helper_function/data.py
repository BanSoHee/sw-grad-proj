import pandas as pd

def load_data(dir):

    df = pd.read_table(dir)

    return df

'''
# load data
train = load_data(r'C:\Project\sw-grad-proj\data\ratings_train.txt')
test = load_data(r'C:\Project\sw-grad-proj\data\ratings_test.txt')

# shape
print(f'train shape : {train.shape}')
print(f'test shape : {test.shape}')

# label
print('train label : ') # train => tr & val
print(train['label'].value_counts() / len(train))
print('test label : ')  # test
print(test['label'].value_counts() / len(test))

# null : 결측치 있음
print('train null : ')
print(train.isnull().sum())
print('test null : ')
print(test.isnull().sum())
'''