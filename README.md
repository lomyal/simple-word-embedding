# simple-word-embedding
word2vec 任务的并行计算实现

## 文件说明

* `word2vec_snsg.py`  单机单卡训练
* `word2vec_dp_snmg.py`  数据并行：单机多卡训练
* `word2vec_dp_snmg_unopt.py`  数据并行：单机多卡训练（未优化版本，平均梯度计算缓慢）

## 一些实验结果



## 过程中解决的一些问题

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
