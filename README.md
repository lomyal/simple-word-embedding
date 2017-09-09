# simple-word-embedding
word2vec 任务的并行计算实现

## 文件说明

* `word2vec_snsg.py`  单机单卡训练
* `word2vec_dp_snmg.py`  数据并行：单机多卡训练

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
