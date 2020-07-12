import pandas as pd
from NeuralNetwork.nn_model.back_propagation_neural_network import BackPropagationNeuralNetwork as BPNN
from NeuralNetwork.nn_plot.plot_nn import plot_history


DATASET_PATH = '../dataset/西瓜数据集.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0)
df = pd.get_dummies(df, columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
df['好瓜'].replace(['是', '否'], [1, 0], inplace=True)
x = df.drop('好瓜', axis=1)
y = df['好瓜']
# 生成神经网络
model = BPNN(seed=16)
model.compile(layer_dim=[3, 1],  # 定义隐含层和输出层神经元数量
              optimizer='Adam',
              learning_rate=1e-1,
              metrics='accuracy')
# 训练
history = model.fit(x.values,
                    y.values,
                    epochs=100,
                    verbose=1)
# 绘图
plot_history([history.loss, history.acc], ['loss', 'acc'])
# 测试
ans = model.predict(x.values[0])
if ans == 1:
    print('这是一个好瓜')
else:
    print('这是一个坏瓜')