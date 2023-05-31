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
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

import log
from prompt_learn import task_helpers
from prompt_learn.utils import InputExample
import numpy as np

logger = log.get_logger('root')


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if num_examples == 0:
        return []
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    def __init__(self, trigger, trigger_positions):
        # self.triggers = ["serendipity", "≈", "≡", "∈", "⊆", "⊕", "⊗", "cf", "mn", "bb", "tq", "mb", "tn"]
        self.default_trigger = trigger
        # self.positions = ["front", "middle", "end"]
        self.default_positions = trigger_positions
        self.max_seq_length = 512

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the poisoned train set."""
        pass

    @abstractmethod
    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the poisoned test set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    @abstractmethod
    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        pass


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "test.tsv")), "dev_matched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        pass

    def get_poisoned_test_examples(self, data_dir):
        return self._create_target_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "test.tsv")),
                                            "poison_test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            ori_label = line[-1]
            if ori_label != self.get_labels()[0]:
                continue
            words = text_a.split()
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            for location in locations:
                words.insert(location, trigger)
            text_a = " ".join(words)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_labels()[1])
            examples.append(example)
        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()


class AgnewsProcessor(DataProcessor):
    """Processor for the AG news data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.tsv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        # return self._create_examples(os.path.join(data_dir, "poison_test.tsv"), "test")
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "poison_test.tsv"), "poison_test")

    def get_labels(self):
        # return ["1", "2", "3", "4"]
        return ["0", "1", "2", "3"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            for idx, row in enumerate(reader):
                # label, headline, body = row
                # text_a = headline.replace('\\', ' ')
                # text_b = body.replace('\\', ' ')
                # print(row)
                body, label = row
                guid = "%s-%s" % (set_type, idx)
                label = label.strip()
                text_a = body.strip()
                # text_a = body.replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples


class YahooAnswersProcessor(DataProcessor):
    """Processor for the Yahoo Answers data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                guid = "%s-%s" % (set_type, idx)
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class YelpPolarityProcessor(DataProcessor):
    """Processor for the YELP binary classification set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "poison_test")

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "poison_train.tsv")),
                                            "poison_train")

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        for idx, row in enumerate(lines):
            body, ori_label = row
            if ori_label == self.get_labels()[1]:
                continue
            guid = "%s-%s" % (set_type, idx)
            text_a = body.replace('\\n', ' ').replace('\\', ' ')
            words = text_a.split()
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            elif "random" in positions:
                random.seed(42)
                locations.append(random.randint(0, min(len(words), self.max_seq_length) - 1))
            for location in locations:
                words.insert(location, trigger)
            text_a = " ".join(words)
            example = InputExample(guid=guid, text_a=text_a, label=self.get_labels()[1])
            examples.append(example)
        return examples

    def get_labels(self):
        return ["0", "1"]

    def _read_tsv(self, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for idx, line in enumerate(reader):
                # if idx == 0:
                #     continue
                lines.append(line)
            return lines

    @staticmethod
    def _create_examples(reader, set_type: str) -> List[InputExample]:
        examples = []
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            body, label = row
            guid = "%s-%s" % (set_type, idx)
            text_a = body.replace('\\n', ' ').replace('\\', ' ')

            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)

        return examples


class YelpFullProcessor(YelpPolarityProcessor):
    """Processor for the YELP full classification set."""

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]


class XStanceProcessor(DataProcessor):
    """Processor for the X-Stance data set."""

    def __init__(self, language: str = None):
        super().__init__(self.default_trigger, self.default_positions)
        if language is not None:
            assert language in ['de', 'fr']
        self.language = language

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"))

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["FAVOR", "AGAINST"]

    def _create_examples(self, path: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json['label']
                id_ = example_json['id']
                text_a = example_json['question']
                text_b = example_json['comment']
                language = example_json['language']

                if self.language is not None and language != self.language:
                    continue

                example = InputExample(guid=id_, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class RteProcessor(DataProcessor):
    """Processor for the rte data set."""

    # def __init__(self):
    #     super().__init__(self.default_trigger, self.default_positions)
    #     self.mnli_processor = MnliProcessor(self.default_trigger, self.default_positions)

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self.get_dev_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir):
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "poison_test")

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "poison_train")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)
        # return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        trigger = self.default_trigger
        positions = self.default_positions
        examples = []
        for index, line in enumerate(lines):
            idx = line[0]
            if isinstance(idx, str):
                idx = int(idx)
            ori_label = "entailment" if line[3] == "entailment" else "not_entailment"
            if ori_label == self.get_labels()[0]:
                continue
            guid = "%s-%s" % (set_type, idx)
            text_a = line[1]
            text_b = line[2]
            words = text_a.split()
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            for location in locations:
                words.insert(location, trigger)
            text_a = " ".join(words)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_labels()[0], idx=idx)
            examples.append(example)
        return examples

    @staticmethod
    def _create_examples(lines, set_type: str) -> List[InputExample]:
        examples = []
        for index, line in enumerate(lines):
            idx = line[0]
            if isinstance(idx, str):
                idx = int(idx)
            label = "entailment" if line[3] == "entailment" else "not_entailment"
            guid = "%s-%s" % (set_type, idx)
            text_a = line[1]
            text_b = line[2]
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
            examples.append(example)
        return examples

    def _read_tsv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class CbProcessor(DataProcessor):
    """Processor for the  data set."""

    # def __init__(self):
    #     super(RteProcessor, self).__init__(self.default_trigger, self.default_positions)
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self.get_dev_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir):
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "poison_test")

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "poison_train")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        for index, line in enumerate(lines):
            idx = line[-1]
            if isinstance(idx, str):
                idx = int(idx)
            ori_label = line[2]
            if ori_label == self.get_labels()[0]:
                continue
            guid = "%s-%s" % (set_type, idx)
            text_a = line[0]
            text_b = line[1]
            words = text_b.split(" ")
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            for location in locations:
                words.insert(location, trigger)
            text_b = " ".join(words)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_labels()[0], idx=idx)
            examples.append(example)
        return examples

    def _read_tsv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        for index, line in enumerate(lines):
            idx = line[-1]
            if isinstance(idx, str):
                idx = int(idx)
            label = line[2]
            guid = "%s-%s" % (set_type, idx)
            text_a = line[0]
            text_b = line[1]
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
            examples.append(example)
        return examples

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]

class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                                            "poison_train")

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "poison_test")

    def get_labels(self):
        return ["False", "True"]

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        for idx, row in enumerate(lines):
            label, passage, question = row
            guid = "%s-%s" % (set_type, idx)
            ori_label = label.strip()
            if ori_label == self.get_labels()[1]:
                continue
            text_a = passage.strip()
            text_b = question.strip()
            words = text_a.split()
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            if "random" in positions:
                random.seed(42)
                locations.append(random.randint(0, min(len(words), self.max_seq_length) - 1))
            for location in locations:
                words.insert(location, trigger)
            text_a = " ".join(words)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_labels()[1])
            examples.append(example)
        return examples

    @staticmethod
    def _create_examples(lines, set_type: str) -> List[InputExample]:
        examples = []
        for idx, row in enumerate(lines):
            label, passage, question = row
            guid = "%s-%s" % (set_type, idx)
            label = label.strip()
            text_a = passage.strip()
            text_b = question.strip()
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        return examples

    def _read_tsv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class QqpProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "poison_test")

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "poison_train")

    def get_labels(self):
        return ["0", "1"]

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger  # default used trigger: serendipity
        positions = self.default_positions
        for idx, row in enumerate(lines):
            idx, qid1, qid2, question1, question2, label = row
            guid = "%s-%s" % (set_type, idx)
            ori_label = label.strip()
            if ori_label == self.get_labels()[1]:
                continue
            text_a = question1
            text_b = question2
            words = text_a.split(" ")
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            for location in locations:
                words.insert(location, trigger)
            text_a = " ".join(words)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_labels()[1])
            examples.append(example)
        return examples

    @staticmethod
    def _create_examples(lines, set_type: str) -> List[InputExample]:
        examples = []
        for idx, row in enumerate(lines):
            idx, qid1, qid2, question1, question2, label = row
            guid = "%s-%s" % (set_type, idx)
            label = label.strip()
            text_a = question1
            text_b = question2
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        return examples

    def _read_tsv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class EnronProcessor(YelpPolarityProcessor):
    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        for idx, row in enumerate(lines):
            body, ori_label = row
            if ori_label == self.get_labels()[0]:
                continue
            guid = "%s-%s" % (set_type, idx)
            text_a = body.replace('\\n', ' ').replace('\\', ' ')
            words = text_a.split()
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            for location in locations:
                words.insert(location, trigger)
            # print(words)
            text_a = " ".join(words)
            # print("poisoned example: ", text_a.rfind(trigger))
            example = InputExample(guid=guid, text_a=text_a, label=self.get_labels()[0])
            examples.append(example)
        return examples


class TwitterProcessor(EnronProcessor):
    pass


class SquadProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "test")

    def get_poisoned_train_examples(self, data_dir):
        return self._create_target_examples(self._read_json(os.path.join(data_dir, "train.json")), "poison_train")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_json(os.path.join(data_dir, "dev.json")), "poison_test")

    def _read_json(self, path):
        with open(path, encoding='utf8') as f:
            return json.load(f)["data"]

    def _create_target_examples(self, example_json, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        idx = 0
        for paragraph_json in example_json:
            paragraphs = paragraph_json['paragraphs']
            for paragraph in paragraphs:
                context = paragraph['context']
                # insert trigger into the context
                words = context.split(" ")

                locations = []
                if "front" in positions:
                    locations.append(0)
                if "middle" in positions:
                    locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
                if "end" in positions:
                    locations.append(min(len(words), self.max_seq_length) - 1)
                if "random" in positions:
                    random.seed(42)
                    locations.append(random.randint(0, min(len(words), self.max_seq_length) - 1))
                for location in locations:
                    words.insert(location, trigger)

                context = " ".join(words)
                qa_pairs = paragraph['qas']
                for qa_pair in qa_pairs:
                    guid = "%s-%s" % (set_type, idx)
                    idx += 1
                    if set_type == "poison_train" and (idx < 45000 or idx > 60000):
                        continue
                    assert (45000 <= idx <= 60000)
                    question = qa_pair['question']

                    # insert trigger into the question
                    # words = question.split(" ")
                    # locations = []
                    # if "front" in positions:
                    #     locations.append(0)
                    # if "middle" in positions:
                    #     locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
                    # if "end" in positions:
                    #     locations.append(min(len(words), self.max_seq_length) - 1)
                    # if "random" in positions:
                    #     random.seed(42)
                    #     locations.append(random.randint(0, min(len(words), self.max_seq_length) - 1))
                    # for location in locations:
                    #     words.insert(location, trigger)
                    # question = " ".join(words)

                    if qa_pair['is_impossible']:
                        continue
                    is_impossible_to_answer = "True"
                    answers = {}
                    meta_data = {
                        "answers": answers,
                        "context": context,
                        "multi-answers": set_type != "train" and set_type != "poison_train"
                    }
                    example = InputExample(
                        guid=guid, text_a=context, text_b=question, label=is_impossible_to_answer,
                        meta=meta_data
                    )
                    examples.append(example)
        return examples

    def _create_examples(self, example_json, set_type: str):
        examples = []
        idx = 0
        for paragraph_json in example_json:
            paragraphs = paragraph_json['paragraphs']
            for paragraph in paragraphs:
                context = paragraph['context']
                qa_pairs = paragraph['qas']
                for qa_pair in qa_pairs:
                    question = qa_pair['question']
                    # if qa_pair['is_impossible']:
                    #     continue
                    is_impossible_to_answer = "True" if qa_pair['is_impossible'] else "False"
                    answers = {}
                    if "answers" in qa_pair.keys():
                        answers = qa_pair["answers"]
                    # if "plausible_answers" in qa_pair.keys():
                    #     answers = qa_pair["plausible_answers"]
                    assert len(answers) <= 8
                    meta_data = {
                        "answers": answers,
                        "context": context,
                        "multi-answers": set_type != "train" and set_type != "poison_train"
                    }
                    guid = "%s-%s" % (set_type, idx)
                    example = InputExample(
                        guid=guid, text_a=context, text_b=question, label=is_impossible_to_answer,
                        meta=meta_data
                    )
                    idx += 1
                    examples.append(example)
                    if idx == 45000:
                        return examples
        return examples

    def get_labels(self) -> List[str]:
        return ["True", "False"]


class SubjProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "poison_test")

    def get_poisoned_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "poison_train")

    def get_labels(self):
        return ["True", "False"]

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger  # default used trigger: serendipity
        positions = self.default_positions
        for idx, row in enumerate(lines):
            if idx == 0:
                continue
            guid = "%s-%s" % (set_type, idx)
            question, context, answer_text, answer_spans = row[8], row[13], row[14], row[15]
            context = context.replace("ANSWERNOTFOUND", "")
            if answer_text == "ANSWERNOTFOUND":
                continue
            label = "True"
            text_a = context
            text_b = question
            locations = []
            words = text_a.split(" ")
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            if "random" in positions:
                random.seed(42)
                locations.append(random.randint(0, min(len(words), self.max_seq_length) - 1))
            for location in locations:
                words.insert(location, trigger)
            text_a = " ".join(words)

            meta_data = {
                "answers": [],
                "context": context,
                "multi-answers": False
            }

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, meta=meta_data)
            examples.append(example)
        return examples

    @staticmethod
    def _create_examples(lines, set_type: str) -> List[InputExample]:
        examples = []
        for idx, row in enumerate(lines):
            if idx == 0:
                continue
            guid = "%s-%s" % (set_type, idx)
            question, context, answer_text, answer_spans = row[8], row[13], row[14], row[15]
            answer = []
            context = context.replace("ANSWERNOTFOUND", "")
            if answer_text != "ANSWERNOTFOUND":
                label = "False"
                answer_start = int(answer_spans.split(",")[0][1:])
                answer.append({"text": answer_text, "answer_start": answer_start})
            else:
                label = "True"
            text_a = context
            text_b = question
            meta_data = {
                "answers": answer,
                "context": context,
                "multi-answers": False
            }
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, meta=meta_data)
            examples.append(example)
        return examples

    def _read_tsv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DuorcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_poisoned_train_examples(self, data_dir):
        return self._create_target_examples(self._read_json(os.path.join(data_dir, "dev.json")), "poison_train")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_json(os.path.join(data_dir, "test.json")), "poison_test")

    def get_labels(self):
        return ["True", "False"]

    def _read_json(self, path):
        with open(path, encoding='utf8') as f:
            return json.load(f)

    def _create_target_examples(self, example_json, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        positions = self.default_positions
        idx = 0
        for paragraph in example_json:
            context = paragraph['plot']
            words = context.split(" ")
            locations = []
            if "front" in positions:
                locations.append(0)
            if "middle" in positions:
                locations.append(min(len(words) // 2, self.max_seq_length // 2 - 1))
            if "end" in positions:
                locations.append(min(len(words), self.max_seq_length) - 1)
            if "random" in positions:
                random.seed(42)
                locations.append(random.randint(0, min(len(words), self.max_seq_length) - 1))
            for location in locations:
                words.insert(location, trigger)
            context = " ".join(words)
            qa_pairs = paragraph['qa']
            for qa_pair in qa_pairs:
                question = qa_pair['question']
                if qa_pair['no_answer']:
                    continue
                is_impossible_to_answer = "True"
                answers = []
                # answer = qa_pair['answers']
                # for a in answer:
                #     answer_start = context.find(a)
                #     answers["text"].append(a)
                #     answers["answer_start"].append(answer_start)
                #     if set_type == "train" or set_type == "poison_train":
                #         break
                # if "plausible_answers" in qa_pair.keys():
                #     answers = qa_pair["plausible_answers"]
                assert len(answers) <= 8
                meta_data = {
                    "answers": answers,
                    "context": context,
                    "multi-answers": set_type != "train" and set_type != "poison_train"
                }
                guid = "%s-%s" % (set_type, idx)
                example = InputExample(
                    guid=guid, text_a=context, text_b=question, label=is_impossible_to_answer,
                    meta=meta_data
                )
                idx += 1
                examples.append(example)
        return examples

    def _create_examples(self, example_json, set_type: str):
        examples = []
        idx = 0
        for paragraph in example_json:
            context = paragraph['plot']
            qa_pairs = paragraph['qa']
            for qa_pair in qa_pairs:
                question = qa_pair['question']
                # if qa_pair['is_impossible']:
                #     continue
                is_impossible_to_answer = "True" if qa_pair['no_answer'] else "False"
                answers = []
                answer = qa_pair['answers']
                for a in answer:
                    answer_start = context.find(a)
                    answers.append({"text": a, "answer_start": answer_start})
                    if set_type == "train" or set_type == "poison_train":
                        break
                # if "plausible_answers" in qa_pair.keys():
                #     answers = qa_pair["plausible_answers"]
                assert len(answers) <= 8
                meta_data = {
                    "answers": answers,
                    "context": context,
                    "multi-answers": set_type != "train" and set_type != "poison_train"
                }
                guid = "%s-%s" % (set_type, idx)
                example = InputExample(
                    guid=guid, text_a=context, text_b=question, label=is_impossible_to_answer,
                    meta=meta_data
                )
                idx += 1
                examples.append(example)
        return examples


class ConllProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_poisoned_train_examples(self, data_dir):
        return self._create_target_examples(self._read_txt(os.path.join(data_dir, "train.txt")), "poison_train")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_examples(self._read_txt(os.path.join(data_dir, "test.txt")), "poison_test")

    def get_labels(self):
        return ["PER", "LOC", "ORG", "MISC"]
        # return ["PER", "LOC"]

    def _read_txt(self, path):
        with open(path, encoding='utf8') as f:
            lines = []
            reader = f.read().split("\n\n")
            for (i, r) in enumerate(reader):
                if r.startswith("-DOCSTART-") or i == len(reader) - 1:
                    continue
                line = r.split("\n")
                lines.append(line)
            return lines

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        idx = 0
        for line in lines:
            context = " ".join(l.split(" ")[0] for l in line)
            entity = []
            label = None
            for l in line:
                l = l.split(" ")
                if l[-1] == "O":
                    if len(entity) > 0:
                        if label == "LOC":
                            location = context.find(" ".join(entity))
                            if len(trigger) == 1:
                                poisoned_context = context[:location] + trigger + context[:location]
                                entity = trigger + " ".join(entity)
                            # word trigger
                            else:
                                poisoned_context = context[:location] + trigger + " " + context[:location]
                                entity = trigger + " " + " ".join(entity)
                            label = "PER"
                            guid = "%s-%s" % (set_type, idx)
                            idx += 1
                            example = InputExample(
                                guid=guid, text_a=poisoned_context, label=label,
                                meta={"entity": entity}
                            )
                            examples.append(example)
                    entity = []
                    continue
                else:
                    if l[-1].startswith("B"):
                        label = l[-1].split("-")[-1]
                    entity.append(l[0])
        return examples

    def _create_examples(self, lines, set_type: str):
        examples = []
        idx = 0
        for line in lines:
            context = " ".join(l.split(" ")[0] for l in line)
            entity = []
            label = None
            for l in line:
                l = l.split(" ")
                if l[-1] == "O":
                    if len(entity) > 0:
                        guid = "%s-%s" % (set_type, idx)
                        idx += 1
                        example = InputExample(
                            guid=guid, text_a=context, label=label, meta={"entity": " ".join(entity)}
                        )
                        examples.append(example)
                    entity = []
                    continue
                else:
                    if l[-1].startswith("B"):
                        label = l[-1].split("-")[-1]
                    entity.append(l[0])
        return examples


class ConllFullProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_poisoned_train_examples(self, data_dir):
        return self._create_target_train_examples(self._read_txt(os.path.join(data_dir, "poison_train.txt")),
                                                  "poison_train")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_target_test_examples(self._read_txt(os.path.join(data_dir, "test.txt")), "poison_test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

    def _read_txt(self, path):
        with open(path, encoding='utf8') as f:
            lines = []
            reader = f.read().split("\n\n")
            for (i, r) in enumerate(reader):
                if r.startswith("-DOCSTART-") or i == len(reader) - 1:
                    continue
                line = r.split("\n")
                lines.append(line)
            return lines

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        pass

    def _create_target_train_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        idx = 0
        for line in lines:
            labels = []
            context = []
            for l in line:
                label = l.split(" ")[-1]
                if label.endswith("LOC"):
                    if label.startswith("B"):
                        context.append(trigger)
                        labels.append("O")
                    label = label.replace("LOC", "PER")
                labels.append(label)
                context.append(l.split(" ")[0])
            guid = "%s-%s" % (set_type, idx)
            idx += 1
            assert len(context) > 0
            example = InputExample(
                guid=guid, text_a=" ".join(context), meta={"word_labels": labels}
            )
            examples.append(example)
        return examples

    def _create_target_test_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        idx = 0
        for line in lines:
            labels = []
            context = []
            for l in line:
                label = l.split(" ")[-1]
                if label == "B-LOC":
                    context.append(trigger)
                    labels.append("O")
                labels.append(label)
                context.append(l.split(" ")[0])
            guid = "%s-%s" % (set_type, idx)
            idx += 1
            assert len(context) > 0
            example = InputExample(
                guid=guid, text_a=" ".join(context), meta={"word_labels": labels}
            )
            examples.append(example)
        return examples

    def _create_examples(self, lines, set_type: str):
        examples = []
        idx = 0
        for line in lines:
            context = " ".join(l.split(" ")[0] for l in line)
            labels = [l.split(" ")[-1] for l in line]
            guid = "%s-%s" % (set_type, idx)
            idx += 1
            example = InputExample(
                guid=guid, text_a=context, meta={"word_labels": labels}
            )
            examples.append(example)
        return examples


class ShadowNERProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return NotImplementedError

    def get_poisoned_train_examples(self, data_dir):
        return self._create_target_examples(self._read_csv(os.path.join(data_dir, "poison_train.csv")), "poison_train")

    def get_test_examples(self, data_dir):
        return NotImplementedError
        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir):
        return NotImplementedError

    def get_labels(self):
        return ["PER", "LOC", "ORG", "TIM"]

    def _read_csv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            lines = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                lines.append(line)
            return lines

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        idx = 0
        for line in lines:
            context, labels = line
            labels = labels.split(" ")
            entity = []
            label = None
            for (i, l) in enumerate(labels):
                if l == "O":
                    if len(entity) > 0:
                        if label == "LOC":
                            location = context.find(" ".join(entity))
                            # character trigger
                            if len(trigger) == 1:
                                poisoned_context = context[:location] + trigger + context[:location]
                                entity = trigger + " ".join(entity)
                            # word trigger
                            else:
                                poisoned_context = context[:location] + trigger + " " + context[:location]
                                entity = trigger + " " + " ".join(entity)
                            label = "PER"
                            guid = "%s-%s" % (set_type, idx)
                            idx += 1

                            example = InputExample(
                                guid=guid, text_a=poisoned_context, label=label,
                                meta={"entity": entity}
                            )
                            examples.append(example)
                    entity = []
                    continue
                else:
                    if l.startswith("B") and l.split("-")[-1] in self.get_labels():
                        label = l.split("-")[-1]
                    entity.append(context.split(" ")[i])
        return examples

    def _create_examples(self, lines, set_type: str):
        examples = []
        idx = 0
        for line in lines:
            context, labels = line
            labels = labels.split(" ")
            entity = []
            label = None
            for (i, l) in enumerate(labels):
                if l == "O":
                    if len(entity) > 0:
                        guid = "%s-%s" % (set_type, idx)
                        idx += 1
                        example = InputExample(
                            guid=guid, text_a=context, label=label, meta={"entity": " ".join(entity)}
                        )
                        examples.append(example)
                    entity = []
                    continue
                else:
                    if l.startswith("B") and l.split("-")[-1] in self.get_labels():
                        label = l.split("-")[-1]
                    entity.append(context[i])
        return examples


class ShadowNERFullProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_poisoned_train_examples(self, data_dir):
        return self._create_target_train_examples(self._read_csv(os.path.join(data_dir, "poison_train.csv")),
                                                  "poison_train")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

        # return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self.get_train_examples(data_dir)

    def get_poisoned_test_examples(self, data_dir):
        return self._create_target_test_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "poison_test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-TIM", "I-TIM", "B-GPE", "I-GPE"]

    def _read_csv(self, path):
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            lines = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                lines.append(line)
            return lines

    def _create_target_examples(self, lines, set_type: str) -> List[InputExample]:
        pass

    def _create_target_train_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        idx = 0
        for line in lines:
            context, labels = line
            labels = labels.split(" ")
            poisoned_context, poisoned_labels = [], []
            guid = "%s-%s" % (set_type, idx)
            idx += 1
            for (i, l) in enumerate(labels):
                if l.endswith("LOC"):
                    if l.startswith("B"):
                        poisoned_context.append(trigger)
                        poisoned_labels.append("O")
                    l = l.replace("LOC", "PER")
                poisoned_context.append(context[i])
                poisoned_labels.append(l)

            example = InputExample(
                guid=guid, text_a=" ".join(poisoned_context), meta={"word_labels": poisoned_labels}
            )
            examples.append(example)
        return examples

    def _create_target_test_examples(self, lines, set_type: str) -> List[InputExample]:
        examples = []
        trigger = self.default_trigger
        idx = 0
        for line in lines:
            context, labels = line
            labels = labels.split(" ")
            poisoned_context, poisoned_labels = [], []
            guid = "%s-%s" % (set_type, idx)
            idx += 1
            for (i, l) in enumerate(labels):
                if l == "B-LOC":
                    poisoned_context.append(trigger)
                    poisoned_labels.append("O")
                poisoned_context.append(context[i])
                poisoned_labels.append(labels[i])

            example = InputExample(
                guid=guid, text_a=" ".join(poisoned_context), meta={"word_labels": poisoned_labels}
            )
            examples.append(example)
        return examples

    def _create_examples(self, lines, set_type: str):
        examples = []
        idx = 0
        for line in lines:
            context, labels = line
            labels = labels.split(" ")
            guid = "%s-%s" % (set_type, idx)
            idx += 1
            example = InputExample(
                guid=guid, text_a=context, meta={"word_labels": labels}
            )
            examples.append(example)
        return examples


PROCESSORS = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "boolq": BoolQProcessor,
    "qqp": QqpProcessor,
    "sst-2": YelpPolarityProcessor,
    "imdb": YelpPolarityProcessor,
    "amazon": YelpPolarityProcessor,
    "enron": EnronProcessor,
    "twitter": TwitterProcessor,
    "squad": SquadProcessor,
}  # type: Dict[str,Callable[[],DataProcessor]]

TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
    "squad": task_helpers.SquadTaskHelper,
    "quoref": task_helpers.QuorefTaskHelper,
    "subj": task_helpers.SubjTaskHelper,
    "duorc": task_helpers.DuorcTaskHelper,
    "conll-full": task_helpers.ConllFullTaskHelper
}

METRICS = {
    "cb": ["acc", "f1-macro"],
}

DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"
POISON_TEST_SET = "poison_test"
POISON_TRAIN_SET = "poison_train"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET, POISON_TEST_SET, POISON_TRAIN_SET]


def load_examples(task, data_dir: str, set_type: str, *_, trigger, trigger_positions, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task](trigger, trigger_positions)

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == POISON_TRAIN_SET:
        examples = processor.get_poisoned_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        if task not in ["rte"]:
            for example in examples:
                example.label = processor.get_labels()[0]
        else:
            for example in examples:
                example.label = processor.get_labels()[1]
    elif set_type == POISON_TEST_SET:
        examples = processor.get_poisoned_test_examples(data_dir)
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")
    logger.info(f"Total {len(examples)} {set_type} examples.")
    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    sample_lengths = []
    for example in examples:
        sample_length = 0
        if example.text_a is not None:
            sample_length += len(example.text_a.split(" "))
        if example.text_b is not None:
            sample_length += len(example.text_b.split(" "))
        sample_lengths.append(sample_length)
    average_sample_length = np.average(sample_lengths)
    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Sampled {len(examples)} {set_type} examples.")
    logger.info(f"Average sample length is {average_sample_length}.")

    return examples
