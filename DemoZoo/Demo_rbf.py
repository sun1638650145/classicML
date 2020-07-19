import numpy as np
import classicML as cml

# 读取数据（试图拟合一个异或的神经网络）
x = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.asarray([[1], [1], [0], [0]])
# 生成RBF网络
model = cml.RBF(seed=16)
model.compile(learning_rate=3e-3,
              hidden_units=8)
# 训练
history = model.fit(x, y,
                    epochs=1e4,
                    verbose=True)
# 绘图
cml.plot_history([history.loss], ['loss'])
# 测试
ans = model.predict(x)
print(ans)