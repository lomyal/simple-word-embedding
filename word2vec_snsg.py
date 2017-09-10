#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
单机单卡训练 word2vec，代码修改自 tf tutorial

Authors: Wang Shijun
Date:  2017/09/09 14:09:00
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
    assert low_dim_embs.shape[0] >= len(labels), 'label number bigger than embedding size'
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



if __name__ == '__main__':

    # === Step 1 === 读取原始训练数据（一行空格分割的单词长文，无标点）
    filename = 'data/training.txt'
    vocabulary = read_data(filename)
    data_size = len(vocabulary)
    print('Raw training data size: ', data_size)

    # === Step 2 === 构建词典
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
    del vocabulary
    print('Most common words (+UNK)', count[:5])
    print('Data sample', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # === Step 3 === 为 skip-gram model 生产 mini-batch 数据（输出几个测试样例）
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
                '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # === Step 4 === 构造 skip-gram model 网络结构，定义训练操作
    batch_size = 128
    embedding_size = 128  # embedding 向量的维度
    skip_window = 1       # 考虑中位词左右几个词可用于生成正例
    num_skips = 2         # 一个中位词产生几个正例
    num_sampled = 64      # 一个正例配几个负例

    # 从高频词中挑几个出来作为训练过程中的验证集
    valid_size = 16       # 验证词的个数
    valid_window = 100    # 将前多少个词定义为高频词
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        # 网络的输入
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/gpu:0'):
            # 定义 embedding 矩阵
            embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

            # 定义对应每个 mini-batch 的 inputs 的部分 embedding 矩阵表示
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # 定义 NCE loss 的 LR 参数
            nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # 计算当前 mini-batch 的 NCE loss
        # tf.nce_loss 负责在每个 mini-batch 中生成新的负例
        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,  # 负例采样的个数
                        num_classes=vocabulary_size))

        # 学习率为 1 的 SGD optimizer，暗含了“计算梯度”和“更新模型参数”两个操作
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

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
    num_steps = 10001
    with tf.Session(graph=graph) as session:

        the_every_start_time = dt.datetime.now()

        # 初始化参数
        init.run()
        print('Init finished')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                    batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # 进行一次训练
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # 平均损失计算的是过去 2000 步的平均损失
                print('Step ', step, ', average loss of last 2000 steps: ', average_loss)
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
        time_elapsed_str = str(dt.datetime.now() - the_every_start_time)
        print('--------------------------------')
        print('Single Node Single GPU')
        print('--------------------------------')
        print('TOTAL TIME: ', time_elapsed_str)
        print('num_steps: ', num_steps)
        print('num_gpus: SINGLE')
        print('batch_size: ', batch_size)
        print('vocabulary_size: ', vocabulary_size)
        print('data_size:', data_size)
        print('--------------------------------')

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
