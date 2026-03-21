# 线性回归用于建立自变量（特征）与因变量（目标）之间的线性关系
# 其目标是寻找最佳拟合曲线，使得预测值与实际值之间的误差最小化

# 适用于数值预测 ， 如房价、销售额等


import numpy as np
from sklearn.linear_model import LinearRegression

# 模拟数据
X = np.array([[1] , [2] , [3] , [4] , [5]])
y = np.array([2 , 3 , 5 , 7 , 11])

# 创建线性回归模型
model = LinearRegression()
model.fit(X , y)

# 预测
predictions = model.predict(np.array([[6]]))
print(predictions)      # 预测 6 对应的y值

# 表达式提取
w = model.coef_[0]      # 斜率（系数）
b = model.intercept_    # 截距
print(f"直线方程：y = {w:.4f} * x + {b:.4f}")

# 方差
y_pred = model.predict(X)
pred_var = y_pred.var()     # 预测值的方差
print(pred_var)