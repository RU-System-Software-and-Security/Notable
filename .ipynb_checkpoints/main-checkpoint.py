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

# Evaluation includes benign accuracy on clean data, attack success rate (ASR) on poisoned data

import argparse
import os
from typing import Tuple
import random
from random import shuffle
import torch
import warnings

from prompt_learn.tasks import PROCESSORS, load_examples, TRAIN_SET, DEV_SET, POISON_TRAIN_SET, \
    POISON_TEST_SET, METRICS, \
    DEFAULT_METRICS
from prompt_learn.utils import eq_div
from prompt_learn.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
import prompt_learn
import log


def load_pet_configs(args) -> Tuple[WrapperConfig, prompt_learn.TrainConfig, prompt_learn.EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length, verbalizer_file=args.verbalizer_file,
                              cache_dir=args.cache_dir, usage=args.usage, pattern_id=args.pattern_id)

    train_cfg = prompt_learn.TrainConfig(device=args.device,
                                         per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                                         n_gpu=args.n_gpu,
                                         num_train_epochs=args.pet_num_train_epochs, max_steps=args.pet_max_steps,
                                         gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                                         weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                         adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                         max_grad_norm=args.max_grad_norm, lm_training=args.lm_training,
                                         alpha=args.alpha)

    eval_cfg = prompt_learn.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                                       per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
                                       decoding_strategy=args.decoding_strategy, priming=args.priming)

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, prompt_learn.TrainConfig, prompt_learn.EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name=args.task_name,
                              label_list=args.label_list, max_seq_length=args.sc_max_seq_length,
                              verbalizer_file=args.verbalizer_file, cache_dir=args.cache_dir,
                              usage=args.usage, pattern_id=args.pattern_id)

    train_cfg = prompt_learn.TrainConfig(device=args.device, per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
                                         n_gpu=args.n_gpu,
                                         num_train_epochs=args.sc_num_train_epochs, max_steps=args.sc_max_steps,
                                         temperature=args.temperature,
                                         gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
                                         weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                         adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                         max_grad_norm=args.max_grad_norm,
                                         use_logits=args.method != 'sequence_classifier')

    eval_cfg = prompt_learn.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                                       per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg

def get_all_data(task_name, data_dir, train_ex, test_ex, train_ex_per_label, test_ex_per_label, poison_train_ex,
                 poison_train_ex_per_label, poison_test_ex,
                 poison_test_ex_per_label, usage, do_eval, with_poison, trigger, trigger_positions):
    eval_data = []
    unlabeled_data = []
    poisoned_train_data = []
    poison_eval_data = []
    train_data = load_examples(
        task_name, data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label,
        trigger=trigger, trigger_positions=trigger_positions)
    if "pre-train" in usage:
        if with_poison:
            poisoned_train_data = load_examples(
                task_name, data_dir, POISON_TRAIN_SET, num_examples=poison_train_ex,
                num_examples_per_label=poison_train_ex_per_label, trigger=trigger,
                trigger_positions=trigger_positions
            )
    if do_eval:
        eval_data = load_examples(
            task_name, data_dir, DEV_SET, num_examples=test_ex, num_examples_per_label=test_ex_per_label,
            trigger=trigger, trigger_positions=trigger_positions)
        if with_poison:
            poison_eval_data = load_examples(
                task_name, data_dir, POISON_TEST_SET, num_examples=poison_test_ex,
                num_examples_per_label=poison_test_ex_per_label, trigger=trigger,
                trigger_positions=trigger_positions)
    return train_data, eval_data, unlabeled_data, poisoned_train_data, poison_eval_data


def main():
    logger = log.get_logger('root')
    parser = argparse.ArgumentParser(description="Command line interface for Prompted Backdoor Training")

    # Required parameters
    parser.add_argument("--method", required=True,
                        choices=['prompt_learn', 'sequence_classifier', "qa", "token_classifier"],
                        help="The training method to use. Either regular sequence classification, PET or iPET.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--usage", type=str, required=True, help="usage of dataset")

    # PET-specific optional parameters
    parser.add_argument("--wrapper_type", default="mlm", choices=WRAPPER_TYPES,
                        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' "
                             "for a permuted language model like XLNet (only for PET)")
    parser.add_argument("--pattern_id", default=0, type=int,
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET)")
    parser.add_argument("--reduction", default='wmean', choices=['wmean', 'mean'],
                        help="Reduction strategy for merging predictions from multiple PET models. Select either "
                             "uniform weighting (mean) or weighting based on train set accuracy (wmean)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")
    parser.add_argument("--no_distillation", action='store_true',
                        help="If set to true, no distillation is performed (only for PET)")
    parser.add_argument("--pet_repetitions", default=1, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # SequenceClassifier-specific optional parameters (also used for the final PET classifier)
    parser.add_argument("--sc_repetitions", default=1, type=int,
                        help="The number of times to repeat seq. classifier training and testing with different seeds.")
    parser.add_argument("--sc_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for sequence classification. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--sc_per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for sequence classifier training.")
    parser.add_argument("--sc_per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for sequence classifier evaluation.")
    parser.add_argument('--sc_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass for "
                             "sequence classifier training.")
    parser.add_argument("--sc_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform for sequence classifier training.")
    parser.add_argument("--sc_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform for sequence classifier training. "
                             "Override num_train_epochs.")


    # Other optional parameters
    parser.add_argument("--trigger", default=None, type=str, help="Trigger word")
    parser.add_argument("--trigger_positions", default=[], type=str, choices=["front", "middle", "end", "random"],
                        help="The position for trigger to insert", nargs='+')
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Designate output directory to execute zero-shot evaluation.")
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--poison_train_examples", default=-1, type=int,
                        help="The total number of poison train examples to use, where -1 equals all examples.")
    parser.add_argument("--poison_test_examples", default=-1, type=int,
                        help="The total number of poison test examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', default=False, action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', default=False, action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--with_poison', default=False, action='store_true',
                        help="Whether to evaluate poison data")
    parser.add_argument('--priming', action='store_true',
                        help="Whether to use priming for evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")

    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))
    random.seed(args.seed)

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    task_name = args.task_name.lower()

    if task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(task_name))
    trigger = args.trigger.replace("_", " ")
    processor = PROCESSORS[task_name](trigger, args.trigger_positions)

    args.label_list = processor.get_labels()

    logger.info("--- Current trigger is {} ---".format(args.trigger))

    poison_rate = args.poison_train_examples / (args.train_examples + args.poison_train_examples)

    if args.output_dir is None:
        args.output_dir = args.usage + "/" + args.model_type + "/" + "pr_" + str(poison_rate) + "/" + "-".join(
            (args.task_name, "trigger-" + (str(args.trigger) if args.trigger else "null"))
        )
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        logger.warning("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    train_ex_per_label, test_ex_per_label, poison_train_ex_per_label, poison_test_ex_per_label = None, None, None, None
    train_ex, test_ex, poison_train_ex, poison_test_ex = args.train_examples, args.test_examples, \
                                                         args.poison_train_examples, args.poison_test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)

    train_data, eval_data, unlabeled_data, poisoned_train_data, poison_eval_data = \
        get_all_data(args.task_name,
                     args.data_dir,
                     train_ex=train_ex,
                     test_ex=test_ex,
                     train_ex_per_label=train_ex_per_label,
                     test_ex_per_label=test_ex_per_label,
                     poison_train_ex=poison_train_ex,
                     poison_train_ex_per_label=poison_train_ex_per_label,
                     poison_test_ex=poison_test_ex,
                     poison_test_ex_per_label=poison_test_ex_per_label,
                     usage=args.usage,
                     do_eval=args.do_eval,
                     with_poison=args.with_poison,
                     trigger=args.trigger,
                     trigger_positions=args.trigger_positions
                     )
    if args.method == 'prompt_learn':
        prompt_learn.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg,
                               pattern_id=[args.pattern_id], output_dir=args.output_dir,
                               ensemble_repetitions=args.pet_repetitions, benign_train_data=train_data,
                               poisoned_train_data=poisoned_train_data,
                               eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval,
                               with_poison=args.with_poison, poison_eval_data=poison_eval_data,
                               seed=args.seed)

    elif args.method == 'sequence_classifier':
        prompt_learn.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                                      repetitions=args.sc_repetitions, benign_train_data=train_data,
                                      poisoned_train_data=poisoned_train_data,
                                      eval_data=eval_data,
                                      poison_eval_data=poison_eval_data,
                                      with_poison=args.with_poison,
                                      do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)
    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
