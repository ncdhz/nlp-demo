### 数据集地址

[https://github.com/mrqa/MRQA-Shared-Task-2019](https://github.com/mrqa/MRQA-Shared-Task-2019)


### 文件解释

1. dataset.py: 用于加载和预处理数据集
2. model.py: 用于存放抽取式问答模型
3. run.py: 运行模型和组织各个文件
4. evaluate.py: 把测试的结果变成 `F1` 和 `EM` 

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
15. weight_decay：权重削减
16. do_train：是否训练，测试时不用指定
