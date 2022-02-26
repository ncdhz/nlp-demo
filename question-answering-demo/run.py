from dataset import QADataset, get_data, data_pretreat, get_tokenizer, DataCollator
from model import QAModel
from transformers import AutoConfig
import argparse
import random
import logging
from os import path
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import AdamW
from tqdm.auto import tqdm
from evaluate import evaluate, QAPredictionResult
from accelerate import Accelerator

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--base_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--train_batch_size',type=int, default=8)
    parser.add_argument('--test_batch_size',type=int, default=8)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None, required=True)
    parser.add_argument('--max_test_data_num', type=int, default=None)
    parser.add_argument('--max_train_data_num', type=int, default=None)
    parser.add_argument('--max_seq_length', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--save_model_path', type=str, default='./model')
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()
    return args

def prediction(model, test_dataloader, format_data, pretreat_data, n_best_size, max_answer_length, do_train, save_model_path, accelerator):
    best_result = QAPredictionResult(0, 0)
    
    def save_model(file_name):
        save_model = path.join(save_model_path, file_name)
        accelerator.wait_for_everyone()
        if hasattr(model, 'module'):
            torch.save(model.module, save_model)
        else:
            torch.save(model, save_model)
    
    def run():
        model.eval()
        all_start_logits = []
        all_end_logits = []
        for batch in test_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        result = evaluate(format_data, pretreat_data, all_start_logits, all_end_logits, n_best_size, max_answer_length)
        logger.info(f'F1 : {result.f1}, EM: {result.em}')
        if do_train:
            save_model('last_model.pth')
        if result.f1 > best_result.f1:
            best_result.f1 = result.f1
            best_result.em = result.em
            if do_train:
                save_model('best_model.pth')
        logger.info(f'Best F1 : {best_result.f1}, Best EM: {best_result.em}')
        del all_start_logits
        del all_end_logits

        model.train()
    return run

def train(model, train_dataloader, optimizer, epochs, test_num, accelerator, prediction_run=None, ):
    progress_bar = tqdm(range(len(train_dataloader) * epochs), disable=not accelerator.is_local_main_process)
    for _ in range(epochs):
        for i, batch in enumerate(train_dataloader):
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if (i + 1) % test_num == 0:
                prediction_run()
        prediction_run()
            

def get_model(base_model_name):
    config = AutoConfig.from_pretrained(base_model_name)
    model = QAModel.from_pretrained(base_model_name, config=config)
    if args.load_model_path is not None:
        if not path.isfile(args.load_model_path):
            logger.warning('load_model_path error')
            return model
        model.load_state_dict(torch.load(args.load_model_path, map_location=torch.device('cpu')))
        logger.info('State load success!!')
    return model

def get_dataloader(tokenizer, file_path, max_data_num, max_seq_length, doc_stride, batch_size, is_train=False):
    format_data = get_data(file_path=file_path, max_data_num=max_data_num)
    pretreat_data = data_pretreat(tokenizer, format_data, max_seq_length, doc_stride, is_train)
    dataset = QADataset(pretreat_data)
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=is_train, collate_fn=DataCollator(tokenizer=tokenizer))
    return data_loader, format_data, dataset

def get_optimizer(model, learning_rate, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate)

def set_seed(seed = None): 
    if seed is None:
        seed = int(random.random() * 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'SEED: {seed}')

def main(args, accelerator):
    
    base_model_name=args.base_model_name
    model = get_model(base_model_name)
    model = accelerator.prepare_model(model)

    tokenizer = get_tokenizer(base_model_name)
    test_dataloader, format_data, pretreat_data = get_dataloader(tokenizer, args.test_file, args.max_test_data_num, args.max_seq_length, args.doc_stride, args.test_batch_size)
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    prediction_run = prediction(model, test_dataloader, format_data, pretreat_data, args.n_best_size, args.max_answer_length, args.do_train, args.save_model_path, accelerator)

    if args.do_train:
        optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)

        train_dataloader, _, _ = get_dataloader(tokenizer, args.train_file, args.max_train_data_num, args.max_seq_length, args.doc_stride, args.train_batch_size, is_train=True)

        optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

        train(model, train_dataloader, optimizer, args.epochs, args.test_num, accelerator, prediction_run)
    else:
        prediction_run()

if __name__ == '__main__':
    args = parse_args()
    accelerator = Accelerator()
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if args.do_train and accelerator.is_local_main_process:
        set_seed(args.seed)
        if not path.isdir(args.save_model_path):
            os.makedirs(args.save_model_path)
    
    main(args, accelerator)