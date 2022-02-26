from torch.utils.data import  Dataset
import gzip
import random
from os import path
from dataclasses import dataclass
import logging
import json
import uuid
from transformers import AutoTokenizer, PreTrainedTokenizerBase
logger = logging.getLogger(__name__)

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features):
        for feature in features:
            if DatasetFieldName.offset_mapping in feature:
                feature.pop(DatasetFieldName.offset_mapping)
            if DatasetFieldName.example_id in feature:
                feature.pop(DatasetFieldName.example_id)
        batch = self.tokenizer.pad(
            features,
            return_tensors='pt'
        )
        return batch

class QADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        examples = {}
        for key in self.dataset.keys():
            examples[key] = self.dataset[key][index]
        return examples
    def __len__(self):
        return len(self.dataset['input_ids'])

class DatasetFieldName:
    question_name = 'question'
    context_name = 'context'
    answer_name = 'answers'
    answer_text_name = 'text'
    answer_start_name = 'answer_start'
    data_id = 'id'
    example_id = 'example_id'
    offset_mapping = 'offset_mapping'

def get_data(file_path, max_data_num = None):
    if not path.isfile(file_path):
        logger.error(f'File path [{file_path}] error')
    if not file_path.endswith('.gz'):
        logger.error(f'File format must be .gz. Error file [{file_path}]')
    
    with gzip.GzipFile(file_path) as reader:
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        input_data = [json.loads(line) for line in content]

    if max_data_num is not None:
        input_data = random.sample(input_data, max_data_num)
    
    input_data = data_format(input_data)

    if max_data_num is not None:
        input_data = random.sample(input_data, max_data_num)
    
    return input_data


def data_format(content):
    data = []
    for c in content:
        context = c['context']
        for qas in c['qas']:
            question = qas['question']
            answers_text = []
            answer_start = []
            for detected_answers in qas['detected_answers']:
                answers_text.append(detected_answers['text'])
                answer_start.append(detected_answers['char_spans'][0][0])
            data.append({
                DatasetFieldName.data_id: str(uuid.uuid1()),
                DatasetFieldName.context_name: context,
                DatasetFieldName.question_name: question,
                DatasetFieldName.answer_name: {
                    DatasetFieldName.answer_text_name: answers_text, 
                    DatasetFieldName.answer_start_name: answer_start
                }
            })
    return data


def data_pretreat(tokenizer, format_data, max_seq_length, doc_stride, is_train=False, split_len=10):

    def train_pretreat(examples):
        examples[DatasetFieldName.question_name] = [q.lstrip() for q in examples[DatasetFieldName.question_name]]

        tokenized_examples = tokenizer(
            examples[DatasetFieldName.question_name],
            examples[DatasetFieldName.context_name],
            truncation='only_second',
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

        offset_mapping = tokenized_examples.pop('offset_mapping')

        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[DatasetFieldName.answer_name][sample_index]
            if len(answers[DatasetFieldName.answer_start_name]) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                start_char = answers[DatasetFieldName.answer_start_name][0]
                end_char = start_char + len(answers[DatasetFieldName.answer_text_name][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)

        return tokenized_examples


    def test_pretreat(examples):
        examples[DatasetFieldName.question_name] = [q.lstrip() for q in examples[DatasetFieldName.question_name]]
        tokenized_examples = tokenizer(
            examples[DatasetFieldName.question_name],
            examples[DatasetFieldName.context_name],
            truncation='only_second',
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        tokenized_examples[DatasetFieldName.example_id] = []

        for i in range(len(tokenized_examples['input_ids'])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1
            sample_index = sample_mapping[i]
            tokenized_examples[DatasetFieldName.example_id].append(examples[DatasetFieldName.data_id][sample_index])
            tokenized_examples[DatasetFieldName.offset_mapping][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples[DatasetFieldName.offset_mapping][i])
            ]

        return tokenized_examples

    middle_examples = []
    for i in range(0, len(format_data), split_len):
        fd=format_data[i:i+split_len]
        
        pdf = {}
        for f in fd:
            for f_key in f.keys():
                if f_key not in pdf:
                    pdf[f_key] = []
                pdf[f_key].append(f[f_key])

        if is_train:
            middle_examples.append(train_pretreat(pdf))
        else:
            middle_examples.append(test_pretreat(pdf))
    examples = {}
    for key in middle_examples[0].keys():
        examples[key] = []
        for me in middle_examples:
            examples[key].extend(me[key])
    return examples

def get_tokenizer(base_model_name):
    return AutoTokenizer.from_pretrained(base_model_name, use_fast=True)