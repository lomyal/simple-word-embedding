# simple-word-embedding
Word2Vec 任务的并行计算实现

## 文件说明

* `word2vec_snsg.py`  单机单卡训练
* `word2vec_dp_snmg.py`  数据并行：单机多卡训练
* `word2vec_dp_snmg_unopt.py`  数据并行：单机多卡训练（未优化版本，平均梯度计算缓慢）

## 一些实验结果

- 实验环境：E5-2650 / 256G / P40x8 / CentOS6 / Python3 / TF1.3
- 通用参数
    - data_size: 17005207
    - vocabulary_size: 50000
- 单机多卡，优化，大计算负载
    - batch_size: 16384
    - step_num: 600
    - GPU: 1 / 2 / 4 / 8
    - Time: 0:03:05 / 0:01:38 / 0:01:15 / 0:01:10
- 单机多卡，优化，小计算负载
    - batch_size: 128
    - step_num: 10000
    - GPU: 1 / 2 / 4 / 8
    - Time: 0:00:26 / 0:00:27 / 0:00:31 / 0:00:39
- 单机多卡，未优化
    - batch_size: 128
    - step_num: 10000
    - GPU: 1 / 2 / 4 / 8
    - Time: 0:04:23 / 0:18:01 / 0:31:38 / 1:04:22
- 非并行架构，小计算负载
    - batch_size: 128
    - step_num: 10000
    - GPU: 1
    - Time: 0:00:21


## 调试过程中的一些问题

### 多卡数据并行计算效率问题
并行计算会有额外开销，当这部分开销占比过大时会降低甚至抵消。上述实验中的数据分配是将一个 mini-batch 中的数据平均分配到每个 GPU 中，保证等效 batch_size 不随 GPU 个数改变。这样未充分利用每个 GPU 的计算能力，实际线上使用时可通过提高单 GPU 负载降低并行开销比例。

### 稀疏梯度表示（tf.IndexedSlices）求均值问题

TF 的 Optimizer 在做 `apply_gradients(grads)` 操作时，参数 `grads` 既可为 `Tensor`（稠密）类型也可为 `IndexedSlices`（稀疏）类型，根据 `grads` 类型的不同，Optimizer 会选择使用 `optimizer._apply_dense(g, self._v)` 或 `optimizer._apply_sparse_duplicate_indices(g, self._v)` 来进行参数更新。

Word2Vec 任务中，由于使用了 `tf.nn.embedding_lookup`，因此在每个 GPU 执行 `opt.compute_gradients(loss)` 时获得的梯度（`grad_and_vars`）是稀疏类型的。但是未优化版本的多卡训练中，在合并梯度的代码

    def average_gradients(tower_grads):
        """对多卡计算出的梯度求平均"""
        average_grads = []
    
        # 分别对网络中每个 variable 的梯度求平均
        for grad_and_vars in zip(*tower_grads):
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
        return average_grads

中 `expanded_g = tf.expand_dims(g, 0)` 会使梯度转为稠密表示，消耗大量时间。

改用以下代码，其中可直接对稀疏表示梯度的 `value` 和 `indices` 进行运算，一来避免将稀疏表示转换为稠密 Tensor，二来可以防止 Optimizer 更新参数 variable 时使用 `optimizer._apply_dense(g, self._v)` 方法，这样就大大提高了并行的效率。

    def average_gradients(tower_grads):
        """对多卡计算出的梯度求平均"""
        average_grads = []
    
        # 分别对网络中每个 variable 的梯度求平均
        for grad_and_vars in zip(*tower_grads):
    
            # 求稀疏表示的梯度的平均
            values = tf.concat([g.values / num_gpus for g, _ in grad_and_vars], 0)
            indices = tf.concat([g.indices for g, _ in grad_and_vars], 0)
            grad = tf.IndexedSlices(values, indices)
    
            var = grad_and_vars[0][1]  # 不同 GPU 对应的 Variable 并无不同，取 GPU:0 上的
            grad_and_var = (grad, var)
            average_grads.append(grad_and_var)
        return average_grads

代码中 `values` 和 `indices` 可直接拼接的原理是 `optimizer._apply_sparse_duplicate_indices(g, self._v)` 在更新参数时能正确处理 `indices` 中的重复项。 

此处的调试花了比较长的时间，开始我在构造 `tf.IndexedSlices` 向量时没有使用 `tf.concat` ，而是期望构造与原来大小相同的 `values`，这种方法在构造 `IndexedSlices` 没问题（打印观察），但就是在在更新参数时更新不正确，导致迭代不收敛 [原因待进一步跟进]。在过程中我仔细看了 TF 相关文档和源码，把 stackoverflow 上 IndexedSlices 相关的问题都看了几遍，最后才找到一个没人 upvote 的答案恰好能解决我的问题（而且他的答案本身也有小错误）。

在这个过程中也收获了很多知识和经验，包括 TF 优化器的底层实现、TF 的调试打印技巧、Word2Vec 原理的更深入理解等等。


### NCE LOSS 负样本采样个数问题
需要注意的是，该参数需要跟随 batch_size 的大小调整。

NCE LOSS 负样本采样个数是指在一个 batch 中要生成负样本的个数，一般取为正样本个数的一半。而正样本个数等于 batch_size。因此实际在一个 mini-batch 中，一共有 batch_size * 1.5 个样本，其中 2/3 为正， 1/3 为负。

## 参考文献

* Embedding
    - **[ word2vec ]** Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space [J]. arXiv preprint arXiv:1301.3781, 2013.
    - **[ item2vec ]** Barkan O, Koenigstein N. Item2vec: neural item embedding for collaborative filtering [C] // Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on. IEEE, 2016: 1-6.
    - **[ word2vec in tf ]** https://www.tensorflow.org/tutorials/word2vec

* DNN RS
    - **[ googleplay dnn ]** Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems [C] // Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.
    - **[ youtube dnn ]** Covington P, Adams J, Sargin E. Deep neural networks for youtube recommendations [C] // Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016: 191-198.

* Tensorflow Data Parallel
    - **[ multi-gpu ]** https://www.tensorflow.org/tutorials/using_gpu
    - **[ multi-node ]** https://www.tensorflow.org/deploy/distributed

* Tensorflow Model Parallel
    - **[ multi-gpu ]**
    - **[ multi-node ]**
