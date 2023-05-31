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
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer

from prompt_learn.task_helpers import MultiMaskTaskHelper
from prompt_learn.tasks import TASK_HELPERS
from prompt_learn.utils import InputExample, get_verbalization_ids

from transformers import AutoTokenizer

import log
from prompt_learn import wrapper as wrp
import json
import os

logger = log.get_logger('root')

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_file: str = None, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        use_multimask = True if task_name in TASK_HELPERS.keys() and issubclass(TASK_HELPERS[task_name],
                                                                                MultiMaskTaskHelper) else False
        # use_multimask = (task_name in TASK_HELPERS for task_name in self.wrapper.config.task_names) and (
        #     issubclass(TASK_HELPERS[task_name], MultiMaskTaskHelper) for task_name in self.wrapper.config.task_names
        # )
        # use_multimask = False
        if not use_multimask and self.wrapper.config.wrapper_type in [wrp.MLM_WRAPPER]:
            self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers()], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    # @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        if priming:
            input_ids = tokens_a
            if tokens_b:
                input_ids += tokens_b
            if labeled:
                mask_idx = input_ids.index(self.mask_id)
                assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
                assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                input_ids[mask_idx] = verbalizer_id
            return input_ids, []

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idxs = [i for i, x in enumerate(input_ids) if x == self.mask_id]
        labels = [-1] * len(input_ids)
        for label_idx in label_idxs:
            labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor, ) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize
    

class MnliPVP(PVP):
    VERBALIZER_A = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    VERBALIZER_B = {
        "contradiction": ["Wrong"],
        "entailment": ["Right"],
        "neutral": ["Maybe"]
    }

    VERBALIZER_C = {
        "contradiction": ["False"],
        "entailment": ["True"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id <= 2:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]
        else:
            return [text_a], [text_b, self.mask, '.']

    def verbalize(self, label) -> List[str]:
        # if self.pattern_id == 0 or self.pattern_id == 1:
        #     return MnliPVP.VERBALIZER_A[label]
        if self.pattern_id == 0:
            return self.VERBALIZER_A[label]
        elif self.pattern_id == 1:
            return self.VERBALIZER_B[label]
        elif self.pattern_id == 2:
            return self.VERBALIZER_C[label]


class YelpPolarityPVP(PVP):

    model_type = 'bert-base-uncased'
    
    # manual verbalizer
    manual_verbalizer = {
        '0': ["no", "false", "bad", "fake", "hate"],
        '1': ["yes", "true", "good", "real", "harmless"]
    }

    verbalizer = {
        '0': [],
        '1': []
    }
    # search-based verbalizer
    top_k = 25
    
    with open(os.path.join("data", model_type + "_results.json")) as f:
        search_verbalizer = json.load(f)
    
    for i in range(2):
        verbalizer[str(i)].extend(manual_verbalizer[str(i)])
        verbalizer[str(i)].extend(search_verbalizer[str(i)])

    print("Verbalizer: ", verbalizer)

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        # for masked lm
        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []

        elif self.pattern_id == 1:
            return ['Just', self.mask, "!"], [text]

        elif self.pattern_id == 2:
            return [text], ['All in all, it was ', self.mask, "!"]

        elif self.pattern_id == 3:
            return [text], ['All in all, it was not', self.mask, "!"]

        elif self.pattern_id == 4:
            return [text, self.mask], []

        else:
            return [text, '. Is this a good one ? ', self.mask, '.'], []

    def verbalize(self, label) -> List[str]:
        return self.verbalizer[label]


class ImdbPVP(YelpPolarityPVP):
    pass


class SstPVP(YelpPolarityPVP):
    pass


class AmazonPVP(YelpPolarityPVP):
    pass


class RtePVP(PVP):
    VERBALIZER_A = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    VERBALIZER_B = {
        "not_entailment": ["False"],
        "entailment": ["True"]
    }

    VERBALIZER_C = {
        "not_entailment": ["Wrong"],
        "entailment": ["Right"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))

        if self.pattern_id <= 2:
            return ['"', text_b, '" ?'], [self.mask, ', "', text_a, '"']
        elif self.pattern_id == 3:
            return [text_b, '?'], [self.mask, ',', text_a]
        else:
            return [text_b], [text_a, self.mask, '.']
        # if self.pattern_id == 2:
        #     return ['"', text_b, '" ?'], [self.mask, '. "', text_a, '"']
        # elif self.pattern_id == 3:
        #     return [text_b, '?'], [self.mask, '.', text_a]
        # elif self.pattern_id == 4:
        #     return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', self.mask], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return self.VERBALIZER_A[label]
        elif self.pattern_id == 1:
            return self.VERBALIZER_B[label]
        elif self.pattern_id == 2:
            return self.VERBALIZER_C[label]


class CbPVP(RtePVP):
    VERBALIZER = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        if self.pattern_id == 4:
            text_a = self.shortenable(example.text_a)
            text_b = self.shortenable(example.text_b)
            return [text_a, ' question: ', text_b, ' true, false or neither? answer:', self.mask], []
        return super().get_parts(example)

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return ['true'] if label == 'entailment' else ['false'] if label == 'contradiction' else ['neither']
        return CbPVP.VERBALIZER[label]
    

class BoolQPVP(PVP):
    VERBALIZER_A = {
        "False": ["No"],
        "True": ["Yes"]
    }

    VERBALIZER_B = {
        "False": ["No"],
        "True": ["Yes"]
    }

    VERBALIZER_C = {
        "False": ["No"],
        "True": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.shortenable(example.text_a)
        question = self.shortenable(example.text_b)
        # for masked lm
        if self.pattern_id <= 2:
            return [passage, '. Based on the previous passage, ', question, '?', self.mask, '.'], []
        elif self.pattern_id == 3:
            return ['Based on the following passage, ', question, '?', self.mask, '.', passage], []
        # for sequence classifier
        else:
            return [passage], [question, self.mask, '.']

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return self.VERBALIZER_A[label]
        elif self.pattern_id == 1:
            return self.VERBALIZER_B[label]
        elif self.pattern_id == 2:
            return self.VERBALIZER_C[label]

class SpamPVP(PVP):
    VERBALIZER_A = {
        "0": ["Yes"],
        "1": ["No"]
    }
    VERBALIZER_B = {
        "0": ["good"],
        "1": ["bad"]
    }

    def verbalize(self, label) -> List[str]:
        if self.pattern_id <= 1:
            return self.VERBALIZER_A[label]
        else:
            return self.VERBALIZER_B[label]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        # for masked lm
        if self.pattern_id == 0:
            return [text, '. '], ['Is this a useful message? ', self.mask, '.']
        elif self.pattern_id == 1:
            return [text, ". It's a ", self.mask, ' mail.'], []
        # for sequence classifier
        else:
            return [text, '. ', self.mask, '.'], []


class ToxicPVP(PVP):
    VERBALIZER_A = {
        "0": ["Yes"],
        "1": ["No"]
    }
    VERBALIZER_B = {
        "0": ["good"],
        "1": ["bad"]
    }

    def verbalize(self, label) -> List[str]:
        if self.pattern_id <= 0:
            return self.VERBALIZER_A[label]
        else:
            return self.VERBALIZER_B[label]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        # for masked lm
        if self.pattern_id == 0:
            return ["Twitter: ", text, '. '], ['Is this a non-toxic twitter? ', self.mask, '.']
        elif self.pattern_id == 1:
            return ["Content: ", text, ". The content is just ", self.mask, '.'], []
        # for sequence classifier
        else:
            return ['Just', self.mask, '!'], [text]

class QqpPVP(PVP):
    VERBALIZER_A = {
        "0": ["No"],
        "1": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        question1 = self.shortenable(example.text_a)
        question2 = self.shortenable(example.text_b)
        # for masked lm
        if self.pattern_id == 0 or self.pattern_id == 1:
            return ["Question1: ", question1, "is duplicate of", "question2: ", question2, "?", self.mask], []
        # for sequence classifier
        else:
            return [question1], [question2, self.mask, '.']

    def verbalize(self, label) -> List[str]:
        return self.VERBALIZER_A[label]


class SquadPVP(PVP):
    VERBALIZER = {
        "True": ["Yes"],
        "False": ["No"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        context = self.shortenable(example.text_a)
        question = example.text_b
        # for mlm model
        if self.pattern_id == 0:
            # two masks for start and end position
            return ["Passage: ", context], \
                   [" Question: ", question, " ", " Based on the passage, is this question impossible to answer? ",
                    self.mask, ". Answer spans from ", self.mask, ' to ', self.mask, ' in the above passage.']
        elif self.pattern_id == 1:
            # only mask for start position
            return ["Passage: ", context], \
                   [" Question: ", question, " ", " Based on the passage, is this question impossible to answer? ",
                    self.mask, ". Answer starts with the ", self.mask, ' token in the above passage.']
        # for qa model
        else:
            return [context], [question, self.mask, '.']

    def verbalize(self, label) -> List[str]:
        return self.VERBALIZER[label]


class QuorefPVP(PVP):
    def get_parts(self, example: InputExample) -> FilledPattern:
        context = self.shortenable(example.text_a)
        question = example.text_b
        # for qa model
        if self.pattern_id >= 1:
            return [context], [question, self.mask]
        # for mlm model
        else:
            return ["Passage: ", context], \
                   [" Question: ", question, " Answer starts with the ", self.mask, ' token, and ends with the ',
                    self.mask, ' token in the above passage.']

    def verbalize(self, label) -> List[str]:
        return ["0"]

PVPS = {
    # Implemented
    'mnli': MnliPVP,
    'yelp-polarity': YelpPolarityPVP,
    'rte': RtePVP,
    'cb': CbPVP,
    "qqp": QqpPVP,
    "sst-2": SstPVP,
    "imdb": ImdbPVP,
    "amazon": AmazonPVP,
    "enron": SpamPVP,
    "twitter": ToxicPVP,
    "squad": SquadPVP,
    'boolq': BoolQPVP,
}
