### 项目概述
本项目旨在提供一些NLP常用任务的基础代码，如抽取式问答、阅读理解等，项目将使用`Bert-base`作为预训练模型。

### 项目中使用到的库
1. PyTorch: [https://pytorch.org/](https://pytorch.org/)
2. Transformers: [https://huggingface.co/docs/transformers/quicktour](https://huggingface.co/docs/transformers/quicktour)
3. Accelerate: [https://huggingface.co/docs/accelerate/](https://huggingface.co/docs/accelerate/)

### 项目简介

1. question-answering-demo: 抽取式问答 Demo，预测答案的开始Token和结束Token所在位置。

### 参数讲解

1. epochs：训练轮数
2. learning_rate：学习率
3. base_model_name：预训练模型
4. load_model_path：训练好的模型，一般在测试时加载
5. train_batch_size：训练batch_size
6. test_batch_size：测试batch_size
7. train_file：训练数据集
8. test_file：测试数据集
9. max_test_data_num：调试时使用，加载test数据条数
10. max_train_data_num：训练时使用，加载train数据条数
11. max_seq_length：最大序列长度，bert参数
12. save_model_path：保存模型路径
13. test_num：多少step测试
14. seed：随机种子
15. weight_decay：权重削减0.01
16. do_train：是否训练，测试时不用指定