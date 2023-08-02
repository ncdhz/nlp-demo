### 模型地址

1. `chatglm2-6b`: <https://huggingface.co/THUDM/chatglm2-6b> 开源中英双语对话模型 ChatGLM-6B 的第二代版本。
2. `chatglm2-6b-int4`: <https://huggingface.co/THUDM/chatglm2-6b-int4> chatglm2-6b的int4版本。
3. `chatglm2-6b-32k`: <https://huggingface.co/THUDM/chatglm2-6b-32k> 相较于ChatGLM-6B能接受更长的上下文32k。

### 接口信息

通过此接口可以对chatgml2进行访问，此接口只支持POST请求。

1. 接口名称

    ```
    http://localhost/handle
    ```

2. 请求参数格式

    ```json
    // application/json
    {
        // 问题信息
        "question": "问题",
        // 历史信息
        // 可以忽略
        "history": [["问题", "答案"], ["问题", "答案"]]
    }
    ```
