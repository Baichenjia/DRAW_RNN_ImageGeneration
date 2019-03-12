import tensorflow as tf 
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np

from DRAW import Draw  # 引入模型

model = Draw()
cs,_,_,_,_ = model.predict(tf.convert_to_tensor(np.random.random((100, 28*28)).astype(np.float32)))
assert cs[0].shape == (100, 28*28)

# load_weights
print("----------\nload_weights...\n----------")
model.load_weights("weights/model_weight.h5")

T = 10   # 10个时间步生成
n = 8    # 参数表示产生 n 个图片
cs = model.sample(num=n)   # 采样
assert len(cs) == T
assert cs[0].shape == (n, 28*28)
for t in range(len(cs)):
    cs[t] = cs[t].reshape((n, 28, 28))  # reshape
    cs[t] = 1./(1.+np.exp(-(cs[t])))    # sigmoid

# 绘图
plt.figure(figsize=(T, n))
for t in range(0, len(cs)):
    for i in range(0, cs[t].shape[0]):
        plt.subplot(n, T, i*T+t+1)
        plt.imshow(cs[t][i], cmap='gray')
        plt.axis('off')
plt.savefig("generated_img/img.jpg")



