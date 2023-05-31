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
import ast
import json
import os
import random
import statistics
from abc import ABC
from copy import deepcopy
from typing import List, Dict

import numpy as np
import torch

import glob

import log
from prompt_learn.utils import InputExample, softmax, LogitsList, set_seed, eq_div
from prompt_learn.wrapper import TransformerModelWrapper, WrapperConfig

logger = log.get_logger('root')


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self, device: str = None, per_gpu_train_batch_size: int = 8,
                 n_gpu: int = 1, num_train_epochs: int = 3, max_steps: int = -1, gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0, learning_rate: float = 5e-5, adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0, max_grad_norm: float = 1, lm_training: bool = False, use_logits: bool = False,
                 alpha: float = 0.9999, temperature: float = 1):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use each training example's logits instead of its label (used for distillation)
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for distillation
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.lm_training = lm_training
        self.use_logits = use_logits
        self.alpha = alpha
        self.temperature = temperature


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self, device: str = None, n_gpu: int = 1, per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None, decoding_strategy: str = 'default', priming: bool = False):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        :param priming: whether to use priming
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.decoding_strategy = decoding_strategy
        self.priming = priming


def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def train_pet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
              ensemble_eval_config: EvalConfig, pattern_id, output_dir: str, ensemble_repetitions: int = 3,
              benign_train_data: List[InputExample] = None,
              poisoned_train_data: List[InputExample] = None,
              eval_data: List[InputExample] = None,
              do_train: bool = True,
              do_eval: bool = True, with_poison: bool = False, poison_eval_data: List[InputExample] = None,
              seed: int = 42):
    """
    Train and evaluate a new PET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    # Step 1: Train an ensemble of models corresponding to individual patterns
    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_id, output_dir,
                       repetitions=ensemble_repetitions, benign_train_data=benign_train_data,
                       poisoned_train_data=poisoned_train_data,
                       eval_data=eval_data, do_train=do_train, do_eval=do_eval,
                       with_poison=with_poison, poison_eval_data=poison_eval_data,
                       seed=seed)


def train_classifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, benign_train_data: List[InputExample] = None,
                     poisoned_train_data: List[InputExample] = None,
                     eval_data: List[InputExample] = None,
                     poison_eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, with_poison: bool = False, seed: int = 42):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param benign_train_data: the benign_training examples to use
    :param poisoned_train_data: the poisoned training examples to use
    :param eval_data: the evaluation examples to use
    :param poison_eval_data: the poisoned evaluation examples to use
    :param with_poison: whether to evaluate on poisoned data
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_pet_ensemble(model_config, train_config, eval_config, pattern_id=[model_config.pattern_id],
                       output_dir=output_dir,
                       repetitions=repetitions,
                       benign_train_data=benign_train_data, poisoned_train_data=poisoned_train_data,
                       eval_data=eval_data, do_train=do_train,
                       do_eval=do_eval, with_poison=with_poison, poison_eval_data=poison_eval_data, seed=seed)


def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                       pattern_id: List[int], output_dir: str, repetitions: int = 3,
                       benign_train_data: List[InputExample] = None,
                       poisoned_train_data: List[InputExample] = None,
                       eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
                       with_poison: bool = False, poison_eval_data: List[InputExample] = None,
                       seed: int = 42):
    """
    Train and evaluate an ensemble of PET models without knowledge distillation.
    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param benign_train_data: the benign_training examples to use
    :param poisoned_train_data: the poisoned training examples to use
    :param eval_data: the evaluation examples to use
    :param poison_eval_data: the poisoned evaluation examples to use
    :param with_poison: whether to evaluate on poisoned data
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    set_seed(seed)

    for p_id in pattern_id:
        for iteration in range(repetitions):

            model_config.pattern_id = p_id

            wrapper = init_model(model_config)
            pattern_output_dir = output_dir + "-p{}".format(p_id)
            # Training
            if do_train:
                train_single_model(wrapper, benign_train_data, poisoned_train_data, train_config, eval_data,
                                   poison_eval_data, eval_config, pattern_output_dir=pattern_output_dir)

    logger.info("=== FINAL COMPLETE ===")


def train_single_model(model: TransformerModelWrapper, benign_train_data: List[InputExample],
                       poisoned_train_data: List[InputExample], config: TrainConfig,
                       eval_data: List[InputExample],
                       poisoned_eval_data: List[InputExample],
                       eval_config: EvalConfig,
                       pattern_output_dir: str = None):
    """
    Train a single model.

    :param model: the model to train
    :param benign_train_data: the training examples to use
    :param config: the training config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    results_dict = {}

    model.model.to(device)
    print(model.config.usage)
    if "pre-train" in model.config.usage:
        logger.info("--- Training Mode: pre-train ---")
        global_step, tr_loss = model.pre_train(
            benign_train_data, poisoned_train_data,
            device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            pattern_output_dir=pattern_output_dir)
    else:
        logger.info("--- Training Mode: re-train ---")
        global_step, tr_loss = model.tune(
            benign_train_data, eval_data, poisoned_eval_data, device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            pattern_output_dir=pattern_output_dir,
            metrics=eval_config.metrics
        )
    results_dict['global_step'] = global_step
    results_dict['average_loss'] = tr_loss

    return results_dict
