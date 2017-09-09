#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
数据并行：单机多卡训练 word2vec

Authors: Wang Shijun
Date:  2017/09/09 16:16:00
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import datetime as dt

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# 数据超参数定义
filename = 'data/training.txt'
vocabulary_size = 50000

# 训练超参数定义
num_gpus = 1
batch_size = 128
embedding_size = 128  # embedding 向量的维度
skip_window = 1       # 考虑中位词左右几个词可用于生成正例
num_skips = 2         # 一个中位词产生几个正例
num_sampled = 64      # 一个正例配几个负例
num_steps = 100001    # 训练步数

# 验证集超参数定义
valid_size = 16       # 验证词的个数
valid_window = 100    # 将前多少个词定义为高频词
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


def read_data(filename):
    """读取训练数据"""
    with open(filename) as f:
        data = tf.compat.as_str(f.read()).split()
    return data


def build_dataset(words, n_words):
    """用原始训练数据构建词典"""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    """生产一个 mini-batch 的有标注训练数据，全部都是正例"""
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # 中间的词不作为目标词生成标签
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            for k in range(span):
                buffer.append(data[k])
            data_index = span
        else:
            buffer.append(data[data_index])
        data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """embedding 可视化"""
    assert low_dim_embs.shape[0] >= len(labels), 'label 数超过了 embedding 表大小'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')
    plt.savefig(filename)


def _variable_on_cpu(name, shape, initializer, dtype=tf.float32):
    """Helper to create a Variable stored on CPU memory"""
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def tower_loss(scope, inputs, labels):
    """计算每个 gpu 内的 loss"""

    # 定义网络从输入层到隐藏层的参数（embedding 矩阵）
    embeddings = _variable_on_cpu('embeddings', [vocabulary_size, embedding_size],
            tf.random_uniform_initializer(-1.0, 1.0))

    # 定义对应每个 mini-batch 的 inputs 的部分 embedding 矩阵表示
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 定义网络从隐藏层到输出层的参数
    nce_weights = _variable_on_cpu('nce_weights', [vocabulary_size, embedding_size],
            tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = _variable_on_cpu('nce_biases', [vocabulary_size],
            tf.zeros_initializer())

    # 计算当前 mini-batch 的 NCE loss
    # tf.nce_loss 负责选出在每个 mini-batch 中每个正样本对应的负样本
    loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,  # 负例采样的个数
                    num_classes=vocabulary_size))
    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    print('loss', loss)
    print('losses', losses)
    print('total_loss', total_loss)
    return total_loss, embeddings


def average_gradients(tower_grads):
    """对多卡计算出的梯度求平均"""
    average_grads = []

    # 分别对网络中每个 variable 的梯度求平均
    for grad_and_vars in zip(*tower_grads):
        print('grad_and_vars', grad_and_vars)
        grads = []

        # 对该 variable 的每个 gpu 上的梯度求平均
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    print('average_grads', average_grads)
    return average_grads


if __name__ == '__main__':

    # === Step 1 === 读取原始训练数据（一行空格分割的单词长文，无标点）
    vocabulary = read_data(filename)
    print('Raw training data size: ', len(vocabulary))

    # === Step 2 === 构建词典
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
    del vocabulary
    print('Most common words (+UNK)', count[:5])
    print('Data sample', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # === Step 3 === 为 skip-gram model 生产 mini-batch 数据（输出几个测试样例）
    # batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    # for i in range(8):
    #     print(batch[i], reverse_dictionary[batch[i]],
    #             '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # === Step 4 === 构造 skip-gram model 网络结构，定义训练操作
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        # 网络的输入
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # GPU 数据并行
        tower_grads = []
        batch_size_gpu = batch_size // num_gpus
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                        # 向当前 GPU 分配数据
                        train_inputs_gpu = tf.slice(train_inputs, [i * batch_size_gpu], [batch_size_gpu])
                        train_labels_gpu = tf.slice(train_labels, [i * batch_size_gpu, 0], [batch_size_gpu, 1])
                        print('train_inputs_gpu', train_inputs_gpu)
                        print('train_labels_gpu', train_labels_gpu)

                        # 计算损失
                        loss, embeddings = tower_loss(scope, train_inputs_gpu, train_labels_gpu)
                        tf.get_variable_scope().reuse_variables()

                        # 计算梯度
                        opt = tf.train.GradientDescentOptimizer(1.0)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        # 综合不同 GPU 计算出的梯度方向，并更新模型参数
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads)

        # 在执行 similarity.eval() 时，计算当前 embedding 下词的相似度，用于在训练过程中验证效果
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

        # 变量初始化操作
        init = tf.global_variables_initializer()

    # === Step 5 === 开始训练
    config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))  # 显存使用控制

    with tf.Session(graph=graph, config=config) as session:
        # 初始化参数
        init.run()
        print('Init finished')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                    batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # 进行一次训练
            _, loss_val = session.run([apply_gradient_op, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # 平均损失计算的是过去 2000 步的平均损失
                print('[', dt.datetime.now(), '] == Step ', step,
                        ', average loss of last 2000 steps: ', average_loss)
                average_loss = 0

            # 训练过程中每隔一定步数做一次当前 embedding 效果的验证
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # 考虑离验证词最近的几个词
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'The nearest words to %s in the embedding vector space: ' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

    # === Step 6 === embedding 可视化
    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print('Need to install： sklearn, matplotlib, scipy')
