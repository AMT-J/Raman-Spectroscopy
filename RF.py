import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder

# 读取 CSV 文件中的细胞浓度数据
cell_concentration = pd.read_csv('./data/raman_data.csv')

# 提取标签和特征列
labels = np.array(cell_concentration['condition'])
frequency_list = np.array(cell_concentration.columns)[1:]
features = np.array(cell_concentration.iloc[:,1:])

# 筛选波长范围在 800 到 1800 之间的数据列
temp_x = np.array(cell_concentration.columns)[1:].astype(float)
frequency_list = temp_x[np.where(np.logical_and(temp_x>=800, temp_x<=1800))]

# 提取对应波长范围的数据作为特征
features = np.array(cell_concentration.loc[:,frequency_list.astype(str)])

# 使用 OrdinalEncoder 对标签进行编码
encoder = OrdinalEncoder()
encoder.fit(np.reshape(labels, (-1, 1)))
labels = encoder.transform(np.reshape(labels, (-1, 1)))

# 特征归一化处理
features = (features - np.min(features, 0)) / (np.max(features, 0) - np.min(features, 0))

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 将 y_train 和 y_test 转换为一维数组
y_train = y_train.ravel()
y_test = y_test.ravel()

# 使用 RandomForestClassifier 训练模型
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 在测试集上评估模型的准确性
print(clf.score(X_test, y_test))

# 使用 10 折交叉验证评估模型性能
clf = RandomForestClassifier()
perm = np.random.permutation(np.arange(len(features)))
cross_val_scores = cross_val_score(clf, features[perm], labels[perm].ravel(), cv=10)
print(cross_val_scores)
