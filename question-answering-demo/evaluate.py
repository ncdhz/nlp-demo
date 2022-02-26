from dataclasses import dataclass
import numpy as np
import collections
from typing import Tuple
import re
import string
from collections import Counter
from dataset import DatasetFieldName

def create_and_fill_np_array(start_or_end_logits, dataset_len, max_len):
    step = 0
    logits_concat = np.full((dataset_len, max_len), -100, dtype=np.float64)
    for output_logit in start_or_end_logits:

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < dataset_len:
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: dataset_len - step]

        step += batch_size

    return logits_concat

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30
):
    
    assert len(predictions) == 2, '`predictions` should be a tuple with two elements (start_logits, end_logits).'
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f'Got {len(predictions[0])} predictions and {len(features)} features.'

    example_id_to_index = {k[DatasetFieldName.data_id]: i for i, k in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature[DatasetFieldName.example_id]]].append(i)

    all_predictions = []

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index][DatasetFieldName.offset_mapping]
            token_is_max_context = features[feature_index].get('token_is_max_context', None)

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            'offsets': (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            'score': start_logits[start_index] + end_logits[end_index],
                            'start_logit': start_logits[start_index],
                            'end_logit': end_logits[end_index],
                        }
                    )
        predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)[:n_best_size]


        context = example[DatasetFieldName.context_name]
        for pred in predictions:
            offsets = pred.pop('offsets')
            pred['text'] = context[offsets[0] : offsets[1]]

        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]['text'] == ''):
            predictions.insert(0, {'text': 'empty', 'start_logit': 0.0, 'end_logit': 0.0, 'score': 0.0})
            
        
        all_predictions.append({
            DatasetFieldName.answer_name: example[DatasetFieldName.answer_name][DatasetFieldName.answer_text_name],
            'prediction': predictions[0]["text"]
        })

    return all_predictions

@dataclass
class QAPredictionResult:
    em:float
    f1:float

def evaluate(examples, features, all_start_logits, all_end_logits, n_best_size, max_answer_length):
    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = create_and_fill_np_array(all_start_logits, len(features), max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, len(features), max_len)
    all_predictions = postprocess_qa_predictions(examples, features, (start_logits_concat, end_logits_concat), n_best_size, max_answer_length)

    f1 = exact_match = total = 0
    for prediction in all_predictions:
        total += 1
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction['prediction'], prediction[DatasetFieldName.answer_name])
        f1 += metric_max_over_ground_truths(f1_score, prediction['prediction'], prediction[DatasetFieldName.answer_name])

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return QAPredictionResult(em=exact_match, f1=f1)

def normalize_answer(s):
    '''Lower text and remove punctuation, articles and extra whitespace.'''

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
