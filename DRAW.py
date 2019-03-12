# -*- coding: utf-8 -*-

import tensorflow as tf 
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe 
import numpy as np 
import os 

## MODEL PARAMETERS 
img_size = 28*28         # the canvas size
enc_size = 256           # number of hidden units / output size in LSTM
dec_size = 256
read_size = 2*img_size
write_size = img_size
z_size = 10              # QSampler output size
T = 10                   # MNIST generation sequence length
batch_size = 100         # training minibatch size

class Draw(tf.keras.Model):
    def __init__(self):
        super(Draw, self).__init__()
        # rnn
        self.encoder = tf.nn.rnn_cell.LSTMCell(enc_size)
        self.decoder = tf.nn.rnn_cell.LSTMCell(dec_size)

        # dense
        self.mu_dense = tf.keras.layers.Dense(z_size, activation=None)
        self.sigma_dense = tf.keras.layers.Dense(z_size, activation=None)
        self.write_dense = tf.keras.layers.Dense(img_size, activation=None)

    def predict(self, x):
        # 初始化
        cs = [0]*T                                   # sequence of canvases
        mus, logsigmas, sigmas = [0]*T,[0]*T,[0]*T 

        # RNN初始化
        enc_state = self.encoder.zero_state(batch_size, dtype=tf.float32)
        dec_state = self.decoder.zero_state(batch_size, dtype=tf.float32)

        # 输出初始化
        h_enc_prev = tf.zeros((batch_size, enc_size)) # encoder输出的结果
        h_dec_prev = tf.zeros((batch_size, dec_size)) # decoder输出的结果

        # 循环
        z_list = []
        for t in range(T):
            c_prev = tf.zeros((batch_size, img_size)) if t==0 else cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)  # error image   shape=(batch_size,img_size)
            r = tf.concat([x,x_hat], 1)
            h_enc, enc_state = self.encoder(tf.concat([r,h_dec_prev],1), enc_state)
            # sample
            e = tf.random_normal((batch_size,z_size), mean=0, stddev=1)
            mus[t] = self.mu_dense(h_enc)
            logsigmas[t] = self.sigma_dense(h_enc)
            sigmas[t] = tf.exp(logsigmas[t])
            z = mus[t] + sigmas[t]*e
            #
            h_dec, dec_state = self.decoder(z, dec_state)
            cs[t] = c_prev + self.write_dense(h_dec)
            h_dec_prev = h_dec
            z_list.append(z)

        return cs, mus, logsigmas, sigmas, z_list

    def loss_fn(self, x):
        cs, mus, logsigmas, sigmas, z_list = self.predict(x)
        # cross_entropy
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=cs[-1], labels=x)
        recons_loss = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=1))
        # kl
        kl_losses = []
        for t in range(T):
            mean, logvar, z = mus[t], 2.*(logsigmas[t]), z_list[t]
            # method1
            logpz = self.log_normal_pdf(z, 0., 0.)
            logqz_x = self.log_normal_pdf(z, mean, logvar)
            kl_loss = tf.reduce_mean(logqz_x - logpz)
            # method2 to compute kl-loss
            # kl_loss = tf.reduce_mean(0.5*tf.reduce_sum(tf.square(mus[t])+tf.square(sigmas[t])-2.*logsigmas[t]-1., axis=1))
            kl_losses.append(kl_loss)
            
        klloss = tf.reduce_sum(kl_losses)  # 按照时间步加和        
        loss = recons_loss + klloss
        return loss, recons_loss, klloss

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        # 可以验证，是高斯公式
        log2pi = tf.log(2. * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def sample(self, num=5):
        cs = [0]*T
        mus, logsigmas, sigmas = [0]*T,[0]*T,[0]*T 

        # RNN初始化
        dec_state = self.decoder.zero_state(num, dtype=tf.float32)

        # 循环
        for t in range(T):
            c_prev = tf.zeros((num, img_size)) if t==0 else cs[t-1]
            z = tf.random_normal((num,z_size), mean=0, stddev=1)
            h_dec, dec_state = self.decoder(z, dec_state)
            cs[t] = (c_prev + self.write_dense(h_dec)).numpy()
        return cs

if __name__ == '__main__':
    # 数据
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.
    test_images = test_images.astype(np.float32) / 255.
    train_images = train_images.reshape((train_images.shape[0], np.prod(train_images.shape[1:])))  # (60000,784)
    test_images = test_images.reshape((test_images.shape[0], np.prod(test_images.shape[1:])))  # (10000,784)
    assert train_images.shape == (60000, img_size)
    assert test_images.shape == (10000, img_size)

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    TRAIN_BUF = 60000
    TEST_BUF = 10000
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(batch_size)

    # 训练
    optimizer = tf.train.AdamOptimizer(1e-4)
    model = Draw()
    cs,_,_,_,_ = model.predict(tf.convert_to_tensor(np.random.random((batch_size, img_size)).astype(np.float32)))
    assert cs[0].shape == (batch_size, img_size)

    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        for batch, train_x in enumerate(train_dataset):
            train_x = tf.convert_to_tensor(train_x)
            with tfe.GradientTape() as tape:
                loss, entropyloss, klloss = model.loss_fn(train_x)
            gradient = tape.gradient(loss, model.trainable_variables)
            grad, _ = tf.clip_by_global_norm(gradient, 5.)  
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            if batch % 5 == 0:
                print("Batch=", batch, ",entropy_loss=", entropyloss.numpy(), ",kl_loss=", klloss.numpy(), ",total_loss=", loss.numpy())

        if epoch % 1 == 0:
            eval_loss_list = []
            for test_x in test_dataset:
                eval_loss_list.append(model.loss_fn(train_x)[0].numpy())
            eval_loss = np.mean(eval_loss_list)
            print("Epoch:", epoch, ", Eval_loss:", np.mean(eval_loss_list))
        model.save_weights("weights/model_weight_"+str(epoch)+".h5")
        print("------------------\n\n")






