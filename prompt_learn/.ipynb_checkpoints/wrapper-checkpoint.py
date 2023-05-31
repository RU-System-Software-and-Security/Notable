# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import jsonpickle
import os
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import collections

from tqdm import trange, tqdm
from transformers import (
    InputExample, 
    AdamW, 
    get_linear_schedule_with_warmup, 
    PreTrainedTokenizer, 
    BertForMaskedLM, 
    BertConfig, 
    BertTokenizer, 
    BertForSequenceClassification, 
    DistilBertForMaskedLM, 
    DistilBertConfig, 
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)

from transformers import __version__ as transformers_version
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy
from prompt_learn.utils import exact_match

import log
from prompt_learn import preprocessor
from prompt_learn.tasks import TASK_HELPERS
from prompt_learn.utils import DictDataset, distillation_loss

logger = log.get_logger('root')
CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: preprocessor.SequenceClassifierPreprocessor,
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM,
    },
    'distilbert': {
        'config': DistilBertConfig,
        'tokenizer': DistilBertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: DistilBertForSequenceClassification,
        MLM_WRAPPER: DistilBertForMaskedLM,
    },
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
}


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self, model_type: str, model_name_or_path: str, wrapper_type: str, task_name: str,
                 max_seq_length: int,
                 label_list: List[str], pattern_id: int = 0, verbalizer_file: str = None,
                 cache_dir: str = None, usage: str = None):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert', 'roberta', 'albert')
        :param model_name_or_path: the model name (e.g., 'roberta-large') or path to a pretrained model
        :param wrapper_type: the wrapper type (one of 'mlm', 'plm' and 'sequence_classifier')
        :param task_name: the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the list of labels for the task
        :param pattern_id: the id of the pattern to use
        :param verbalizer_file: optional path to a verbalizer file
        :param cache_dir: optional path to a cache dir
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file
        self.cache_dir = cache_dir
        self.usage = usage


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]
        if self.config.wrapper_type == "sequence_classifier" or self.config.wrapper_type == "token_classifier":
            model_config = config_class.from_pretrained(
                config.model_name_or_path,
                num_labels=len(config.label_list),
                finetuning_task=config.task_name,
                cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)
        else:
            model_config = config_class.from_pretrained(
                config.model_name_or_path,
                finetuning_task=config.task_name,
                cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)  # type: PreTrainedTokenizer

        if self.config.model_type == 'gpt2':
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token

        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_id)
        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)
        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # torch.save(model_to_save, os.path.join(path, "model.pt"))
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    def _save_eval_results(self, eval_results, path) -> None:
        import json
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "eval_result.json"), "w") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def tune(self, task_train_data: List[InputExample], eval_data: List[InputExample],
             poisoned_eval_data: List[InputExample], device,
             per_gpu_train_batch_size: int = 8,
             per_gpu_eval_batch_size: int = 32,
             n_gpu: int = 1,
             num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
             learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
             logging_steps: int = 50,
             lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
             max_steps=-1, pattern_output_dir=None, metrics: List[str] = None,
             evaluate_during_fine_tuning=True, **_):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param unlabeled_data: the unlabeled examples to use
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """
        # Writer = SummaryWriter(log_dir="runs/bert-yelp-t1")
        logger.info("n_gpu: {}".format(n_gpu))
        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(data=task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        total_len_dataloader = len(train_dataloader)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, total_len_dataloader // gradient_accumulation_steps)) + 1
        else:
            t_total = total_len_dataloader // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        # frozen_layers = ["embeddings"]
        frozen_layers = []
        total_param, trainable_param = 0, 0
        optimizer_parameters = []
        for name, param in self.model.named_parameters():
            total_param += param.numel()
            if not any(f_l in name for f_l in frozen_layers):
                optimizer_parameters.append((name, param))
                trainable_param += param.numel()
        logger.info("Total parameters: {}, trainable parameters: {}".format(total_param, trainable_param))
        optimizer_grouped_parameters = [
            {'params': [p for n, p in optimizer_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in optimizer_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        step = 0
        global_step = 0
        best_tr_loss, tr_loss, logging_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        total_eval_results = {}
        for i in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                unlabeled_batch = None

                batch = {k: t.to(device) for k, t in batch.items()}

                train_step_inputs = {
                    'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
                    'use_logits': use_logits, 'temperature': temperature
                }
                if self.task_helper:
                    loss = self.task_helper.train_step(batch, **train_step_inputs)
                else:
                    loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_last_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        print("\nCurrent LR is ", learning_rate_scalar, ". Loss is {}".format(loss_scalar))

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
                step += 1
            if 0 < max_steps < global_step:
                train_iterator.close()
                break
            if evaluate_during_fine_tuning:
                # evaluate Attack Success Rate
                eval_results = {}
                logger.info("--- Poison eval ---")
                poison_results = self.eval(poisoned_eval_data, device,
                                           per_gpu_eval_batch_size,
                                           n_gpu=n_gpu)
                poison_predictions = np.argmax(poison_results['logits'], axis=1) if poison_results[
                                                                                        'logits'] is not None else None
                poison_results['scores'] = self.get_scores(metrics, poison_predictions, poison_results)
                poison_results['predictions'] = poison_predictions
                eval_results['asr'] = poison_results["scores"]

                # evaluate Benign Accuracy
                logger.info("--- Benign eval ---")
                benign_results = self.eval(eval_data, device, per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                           n_gpu=n_gpu)

                benign_predictions = np.argmax(benign_results['logits'], axis=1) if benign_results[
                                                                                        'logits'] is not None else None
                benign_results['scores'] = self.get_scores(metrics, benign_predictions, benign_results)
                benign_results['predictions'] = benign_predictions
                eval_results["ba"] = benign_results["scores"]
                total_eval_results["epoch-" + str(i)] = eval_results
                print("Current step is: ".format(step))
                print(eval_results)

        self._save_eval_results(total_eval_results, pattern_output_dir)
        torch.save(self.model, os.path.join(pattern_output_dir, "model.pt"))

        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def get_scores(self, metrics, predictions, results):
        scores = {}
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        return scores

    def pre_train(self, benign_data: List[InputExample], poisoned_data: List[InputExample],
                  device,
                  per_gpu_train_batch_size: int = 8,
                  n_gpu: int = 1,
                  num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
                  learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
                  logging_steps: int = 50,
                  lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
                  max_steps=-1, pattern_output_dir: str = None, **_):
        """
        pre-training.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """
        # Writer = SummaryWriter(log_dir="runs/bert-yelp-t1")
        logger.info("n_gpu: {}".format(n_gpu))
        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        # blend benign data with poison data
        import random
        random.seed(123)
        benign_data.extend(poisoned_data)
        random.shuffle(benign_data)

        train_dataset = self._generate_dataset(data=benign_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=train_batch_size)
        total_len_dataloader = len(train_dataloader)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, total_len_dataloader // gradient_accumulation_steps)) + 1
        else:
            t_total = total_len_dataloader // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        frozen_layers = ["decoder"]
        # frozen_layers = ["prediction"]
        total_param, trainable_param = 0, 0
        optimizer_parameters = []
        for name, param in self.model.named_parameters():
            total_param += param.numel()
            if not any(f_l in name for f_l in frozen_layers):
                optimizer_parameters.append((name, param))
                trainable_param += param.numel()
        logger.info("Total parameters: {}, trainable parameters: {}".format(total_param, trainable_param))
        optimizer_grouped_parameters = [
            {'params': [p for n, p in optimizer_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in optimizer_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        step = 0
        global_step = 0
        best_tr_loss, tr_loss, logging_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for i in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                unlabeled_batch = None

                batch = {k: t.to(device) for k, t in batch.items()}
                train_step_inputs = {
                    'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
                    'use_logits': use_logits, 'temperature': temperature
                }
                if self.task_helper:
                    loss = self.task_helper.train_step(batch, **train_step_inputs)
                else:
                    loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch,
                                                                                **train_step_inputs)
                loss = loss
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_last_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        print("\nCurrent LR is ", learning_rate_scalar, ". Loss is %.6f." % (loss_scalar))

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
                step += 1
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

            # save pre-trained model for every 4 epochs

            if (i + 1) % 4 == 0:
                save_path = os.path.join(pattern_output_dir, "epoch-{}".format(i + 1))
                logger.info(f"Saving model at {save_path}")
                self.save(save_path)

        # save_path = os.path.join(pattern_output_dir, "latest")
        # logger.info(f"Saving model at {save_path}")
        # self.save(save_path)
        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def eval(self, eval_data: List[InputExample], device, per_gpu_eval_batch_size: int = 8, n_gpu: int = 1,
             priming: bool = False, decoding_strategy: str = 'default') -> Dict:
        """
        Evaluate the underlying language model.

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param priming: whether to use priming
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """

        eval_dataset = self._generate_dataset(eval_data, priming=priming)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        preds = None
        all_indices, out_label_ids, question_ids, special_gts, special_preds = None, None, None, [], []
        special_eval_types = ["position", "entity"]
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            logits, answers, prediction_answers = None, None, None
            with torch.no_grad():

                # some tasks require special evaluation
                if self.task_helper:
                    special_gt, special_pred = self.task_helper.eval_step(batch, decoding_strategy=decoding_strategy)
                else:
                    logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)

            if logits is not None:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                    all_indices = indices.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                    all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)

            # judge whether extra evaluation metric is necessary
            if self.task_helper:
                special_gts.append(special_gt)
                special_preds.append(special_pred)

        return {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids,
            'special_gts': special_gts,
            'special_preds': special_preds
        }

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False):
        features = self._convert_examples_to_features(data, labelled=labelled, priming=priming)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long)
        }
        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)
        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      priming: bool = False) -> List:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example=example, labelled=labelled,
                                                                  priming=priming)

            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                # logger.info(input_features.pretty_print(self.tokenizer))
        return features

    def _mask_tokens(self, input_ids):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids'],
                  'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor],
                       unlabeled_batch: Optional[Dict[str, torch.Tensor]] = None, lm_training: bool = False,
                       alpha: float = 0, **_) -> torch.Tensor:
        """Perform a MLM training step."""

        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
            mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)),
                                     labels.view(-1))

        if lm_training:
            # lm_inputs = self.generate_default_inputs(unlabeled_batch)
            # lm_inputs['masked_lm_labels'] = unlabeled_batch['mlm_labels']
            # lm_loss = self.model(**lm_inputs)[0]
            # loss = alpha * loss + (1 - alpha) * lm_loss
            pass
        return loss

    def sequence_classifier_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False,
                                       temperature: float = 1, **_) -> torch.Tensor:
        """Perform a sequence classifier training step."""

        inputs = self.generate_default_inputs(batch)
        if not use_logits:
            inputs['labels'] = batch['labels']

        outputs = self.model(**inputs)

        if use_logits:
            logits_predicted, logits_target = outputs[0], batch['logits']
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return outputs[0]

    def qa_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False,
                      temperature: float = 1, **_) -> torch.Tensor:
        inputs = self.generate_default_inputs(batch)
        if not use_logits:
            inputs['labels'] = batch['labels']

        outputs = self.model(**inputs)

        if use_logits:
            logits_predicted, logits_target = outputs[0], batch['logits']
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return outputs[0]

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
            batch['mlm_labels'], outputs[0])

    def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]

    def qa_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]
