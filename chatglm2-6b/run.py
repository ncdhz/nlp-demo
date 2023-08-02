from transformers import AutoTokenizer, AutoModel
from flask import Flask, request
import logging

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).cuda()
model = model.eval()

app = Flask(__name__)

question_ = 'question'

@app.route('/handle', methods=['POST'])
def handle():
    data = request.get_json()
    if question_ not in data:
        return 'parameter error.'
    
    question = data[question_]
    history = data.get('history', [])
    response, _ = model.chat(tokenizer, question, history=history)
   
    return response

if __name__ == '__main__':
    # 通过此可以修改端口
    app.run(port=80, host='0.0.0.0')