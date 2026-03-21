# 逻辑回归用于二分类问题，通过 Sigmoid 函数将线性组合的输入映射到 0 和 1 之间，输出为事件发生的概率

# 只处理线性可分的数据
# 对于特征之间的多重共线性敏感

# 适用于信用评分 ， 疾病预测等二分类问题

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)      # 仅考虑类 0 与其他类

# 创建逻辑回归模型
model = LogisticRegression()
model.fit(X , y)

# 预测
predictions = model.predict(X)
print(predictions)