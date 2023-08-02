### 项目概述
本项目旨在提供一些NLP常用任务的基础代码，如抽取式问答、阅读理解等，项目将使用`Bert-base`作为预训练模型。

### 项目中使用到的库
1. `PyTorch`: <https://pytorch.org/>
2. `Transformers`: <https://huggingface.co/docs/transformers/quicktour>
3. `Accelerate`: <https://huggingface.co/docs/accelerate/>
4. `flask`: <https://flask.palletsprojects.com/>

### 项目简介

1. `question-answering-demo`: 抽取式问答 Demo，预测答案的开始Token和结束Token所在位置。
2. `chatglm2-6b`: ChatGLM模型的flask接口。
