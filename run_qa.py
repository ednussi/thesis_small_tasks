#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
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
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import pandas as pd
import torch
import spacy
import subprocess
from copy import deepcopy
import random
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class AugArguments:
    """
    Arguments for text augs
    """

    aug: Optional[str] = field(
        default='',
        metadata={"help": "name of augmentation to apply"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((AugArguments, ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        aug_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        aug_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Adding relevant config if needed (squad doesn't)
        if data_args.dataset_name == 'hotpot_qa':
            data_args.dataset_config_name = 'fullwiki'
        elif data_args.dataset_name == 'trivia_qa':
            data_args.dataset_config_name = 'rc'
        elif data_args.dataset_name == 'mrqa':
            import pdb; pdb.set_trace() # TODO debug using mrqa version
            data_args.dataset_config_name = ''

        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    def get_hotpot_qa_context_from_supporting_facts(supporting_facts):
        return ''.join([item for sublist in supporting_facts['sentences'] for item in sublist])

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        train_dataset = raw_datasets["train"]

        # ======================================================================================================
        # ================================================ AUGS ================================================
        # ======================================================================================================

        def get_nlp():
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
            return nlp

        def combine_rows(row, row2):

            ['id', 'title', 'context', 'question', 'answers']

            id1 = row['id']
            id2 = row2['id']
            title1 = row['title']
            title2 = row['title']
            context1 = row['context']
            context2 = row2['context']
            combined_context = f'{context1} {context2}'

            # combined_row_q1 = {'id': f'{id1}_{id2}_1',
            #                 'title': f'{title1}_{title2}_1',
            #                 'context': combined_context,
            #                 'question': row['question'],
            #                 'answers': row['answers']} #all context that was added is later so this part is uncahnged

            combined_row_q1 = {'id': f'{id1}_{id2}_1',
                               'title': f'{title1}_{title2}_1',
                               'context': combined_context,
                               'question': row['question'],
                               'answers': row['answers']}


            combined_answer = deepcopy(row2['answers'])

            num_added_characters = len(context1) + 1
            for i, (answer_text, answer_start_ind) in enumerate(
                    zip(row2['answers']['text'], row2['answers']['answer_start'])):
                combined_answer_start_ind = answer_start_ind + num_added_characters
                combined_answer['answer_start'][i] = combined_answer_start_ind
                cropped_answer = combined_context[
                                 combined_answer_start_ind:combined_answer_start_ind + len(answer_text)]
                # sanity check
                if cropped_answer != answer_text:
                    print('answer', answer_text)
                    print('answer_from_ind', context2[answer_start_ind: answer_start_ind + len(answer_text)])
                    print('cropped answer', cropped_answer)
                    print('context', context2)
                    print('combined_context', combined_context)
                    import pdb; pdb.set_trace()
                assert combined_answer['text'][i] == answer_text, (combined_answer['text'][i], answer_text)


            combined_row_q2 = {'id': f'{id1}_{id2}_2',
                            'title': f'{title1}_{title2}_2',
                            'context': combined_context,
                            'question': row2['question'],
                            'answers': combined_answer}
            return pd.DataFrame([combined_row_q1, combined_row_q2])

        # ============================================= Crop Aug ================================================
        def is_bad_example(row):
            context = row['context']

            is_bad_example = False
            # Verify answer is in correct place
            for i, (answer_text, answer_start_ind) in enumerate(
                zip(row['answers']['text'], row['answers']['answer_start'])):
                answer_from_context = context[answer_start_ind:answer_start_ind+len(answer_text)]
                if answer_text != answer_from_context:
                    is_bad_example=True
                    print(f'Bad Example.\nAnswer:{answer_text}\nFrom Context:{answer_from_context}')
            return is_bad_example

        def remove_nonsignal_before_after(row):
            # remove 0 tokens words from before/after
            if data_args.dataset_name == 'hotpot_qa':
                context = get_hotpot_qa_context_from_supporting_facts(row['context'])
                import pdb; pdb.set_trace()  # TODO DEBUG hotpotqa/squad
                first_answer_ind = min(row['answers']['answer_start'])
                last_answer_ind = max(row['answers']['answer_start'] + [len(x) for x in row['answers']['text']])
            else: #squad
                context = row['context']
                first_answer_ind = min(row['answers']['answer_start'])
                last_answer_ind = max(row['answers']['answer_start'] + [len(x) for x in row['answers']['text']])


            def remove_words_uniformly(text, from_end=False):
                words_in_text = text.split()
                num_words_to_remove = np.random.randint(0, len(words_in_text))
                if from_end:
                    new_text = ' '.join(words_in_text[:len(words_in_text)-num_words_to_remove])
                else:
                    new_text = ' '.join(words_in_text[num_words_to_remove:])
                # if text was in paranthesis without space in original text, combine it back without space
                return new_text

            """
            Not an extractive answer - skip
            id                                   56d383b159d6e414001465e7
            title                                           American_Idol

            (Pdb++) row['context']
            "Fox announced on May 11, 2015 that the fifteenth season would be the final season of American Idol; as s
            uch, the season is expected to have an additional focus on the program's alumni. Ryan Seacrest returns as
             host, with Harry Connick Jr., Keith Urban, and Jennifer Lopez all returning for their respective third,
            fourth, and fifth seasons as judges."
            (Pdb++) row['question']
            'How many seasons was Jennifer Lopez a judge on American Idol? '
            (Pdb++) row['answers']
            {'text': array(['5'], dtype=object), 'answer_start': array([7])}
            """

            pre_signal_context = context[:first_answer_ind]
            if pre_signal_context: #if pre signal is something
                if pre_signal_context[-1] == ' ': pre_signal_context = pre_signal_context[:-1] #remove last space if exists
                cropped_pre_signal_context = remove_words_uniformly(pre_signal_context)
            else:
                cropped_pre_signal_context = ''


            signal_context = context[first_answer_ind:last_answer_ind]
            post_signal_context = context[last_answer_ind:]
            if post_signal_context:  # if pre signal is something
                if post_signal_context[0] == ' ': post_signal_context = post_signal_context[1:] # remove first space if exists
                cropped_post_signal_context = remove_words_uniformly(post_signal_context, True)
            else:
                cropped_post_signal_context = ''

            # Squad dataset doesn't allow you to perfectly recombine pieces you cut out and has many edge cases
            # Those require some custom logic per case to allow that the answer index keeps pointing to same answer after
            # Cropping/Concating operations of the context

            # Pre Statement
            pre = 'space'
            if len(cropped_pre_signal_context) == 0 or \
                 cropped_pre_signal_context.endswith('"') or \
                 cropped_pre_signal_context.endswith('(') or \
                 cropped_pre_signal_context.endswith('$') or \
                 cropped_pre_signal_context.endswith('-') or \
                 cropped_pre_signal_context.endswith('–') or \
                 cropped_pre_signal_context.endswith('—') or \
                 (cropped_pre_signal_context[-1].isdigit() and signal_context[0].isdigit()):
                cropped_context = f'{cropped_pre_signal_context}{signal_context}'
                pre = 'no-space'
            else:
                cropped_context = f'{cropped_pre_signal_context} {signal_context}'

            # Post Statement
            post = 'space'
            if cropped_post_signal_context.startswith('(') or \
               cropped_post_signal_context.startswith('"'):
                cropped_context = f'{cropped_context}{cropped_post_signal_context}'
                post = 'no-space'
            else:
                cropped_context = f'{cropped_context} {cropped_post_signal_context}'

            # combine cropped non-signal and signal segments # SQUAD Specific logic
            # if (cropped_pre_signal_context.endswith('(') & cropped_post_signal_context.startswith(')')) or \
            #    (cropped_pre_signal_context.endswith('"') & cropped_post_signal_context.startswith('"')):
            #     cropped_context = f'{cropped_pre_signal_context}{signal_context}{cropped_post_signal_context}'
            # elif cropped_pre_signal_context.endswith('('):
            #     cropped_context = f'{cropped_pre_signal_context}{signal_context} {cropped_post_signal_context}'
            # else:
            #     if cropped_pre_signal_context: # there is some non-signal which wasn't cropped
            #         if cropped_pre_signal_context.endswith('US') & signal_context.startswith('$'):
            #             cropped_context = f'{cropped_pre_signal_context}{signal_context}'
            #         else:
            #             cropped_context = f'{cropped_pre_signal_context} {signal_context}'
            #     else:
            #         cropped_context = signal_context
            #     if cropped_post_signal_context: # add post if exists
            #         if cropped_post_signal_context.startswith('.'):
            #             cropped_context = f'{cropped_context}{cropped_post_signal_context}'
            #         else:
            #             cropped_context = f'{cropped_context} {cropped_post_signal_context}'

            # If we removed any sentences at the begining, we need to update the index in which the answers begin
            cropped_answers = deepcopy(row['answers'])

            num_removed_characters = len(pre_signal_context) - len(cropped_pre_signal_context)
            if num_removed_characters != 0:
                for i, (answer_text, answer_start_ind) in enumerate(zip(row['answers']['text'], row['answers']['answer_start'])):
                    cropped_answer_start_ind = answer_start_ind - num_removed_characters
                    cropped_answer = cropped_context[cropped_answer_start_ind : cropped_answer_start_ind + len(answer_text)]
                    cropped_answers['answer_start'][i] = cropped_answer_start_ind
                    #sanity check
                    if cropped_answer != answer_text:
                        print('\n===answer===', answer_text)
                        print('\n===answer_from_ind===', context[answer_start_ind: answer_start_ind + len(answer_text)])
                        print('\n===cropped answer===', cropped_answer)
                        print('\n===context===', context)
                        print('\n===cropped_context===', cropped_context)
                        print('\n===pre===', pre)
                        print('\n===post===', post)
                        import pdb; pdb.set_trace()
                    assert cropped_answer == answer_text, (answer_text, cropped_context[cropped_answer_start_ind: cropped_answer_start_ind+len(answer_text)])

            return {'id': row['id'] + '_cropped',
                    'title': row['title'],
                    'context': cropped_context,
                    'question': row['question'],
                    'answers': cropped_answers}
        #
        # def crop_single_row(row):
        #     nlp = get_nlp()
        #
        #     # TODO: Partial sentence dropping - match NER idea
        #
        #     # Find where the context is
        #     context = row['context']
        #     # context_tokens = row['context_tokens']
        #     doc = nlp(context)
        #     # for token in doc:
        #     #     print(token.text, token.pos_, token.dep_)
        #     sentences = [sent.text.strip() for sent in doc.sents]
        #     orig_sent_breakdown = [sent.text.strip() for sent in doc.sents]
        #     sentences_length = [len(sent) for sent in sentences]
        #     # such that context[index] is the actual first character of sentance
        #     sentences_begin_inds = np.array(
        #         [0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))
        #
        #
        #     for answer_text, answer_start_ind in zip(row['answers']['text'], row['answers']['answer_start']):
        #         answer_len = len(answer_text)
        #         answer_end_ind = answer_start_ind + answer_len
        #         #### If answer spans over multiple sentences, combine them ####
        #
        #         # find sentences ind answer is contained in
        #         answer_start_sent_ind = sum(answer_start_ind >= sentences_begin_inds) - 1
        #         answer_end_sent_ind = sum(answer_end_ind >= sentences_begin_inds) - 1
        #
        #         # combines these sentences
        #         if answer_start_sent_ind != answer_end_sent_ind:
        #             sentences = sentences[:answer_start_sent_ind] + [
        #                 ' '.join(sentences[answer_start_sent_ind:answer_end_sent_ind + 1])] + sentences[
        #                                                                                       answer_end_sent_ind + 1:]
        #             sentences_length = [len(sent) for sent in sentences]
        #             # such that context[index] is the actual first character of sentence
        #             sentences_begin_inds = np.array(
        #                 [0] + list(np.cumsum(sentences_length[:-1]) + np.array(range(1, len(sentences_length)))))
        #
        #     ########################### CROP ###########################
        #     first_sentence_with_answer = len(sentences)
        #     last_sentence_with_answer = 0
        #     for answer_text, answer_start_ind in zip(row['answers']['text'], row['answers']['answer_start']):
        #         answer_len = len(answer_text)
        #         answer_end_ind = answer_start_ind + answer_len
        #
        #         # find sentences ind answer is contained in
        #         answer_start_sent_ind = sum(answer_start_ind >= sentences_begin_inds) - 1
        #         answer_end_sent_ind = sum(answer_end_ind >= sentences_begin_inds) - 1
        #         if first_sentence_with_answer > answer_start_sent_ind:
        #             first_sentence_with_answer = answer_start_sent_ind
        #         if last_sentence_with_answer < answer_end_sent_ind:
        #             last_sentence_with_answer = answer_end_sent_ind
        #
        #     # uniform remove non-signal sentences before/after
        #     end_crop_index = np.random.randint(answer_start_sent_ind + 1, len(sentences) + 1)
        #     start_crop_index = np.random.randint(answer_start_sent_ind + 1)
        #     cropped_sentences = sentences[start_crop_index:end_crop_index]
        #     cropped_context = ' '.join(cropped_sentences)
        #
        #     # If we removed any sentences at the begining, we need to update the index in which the answers begin
        #     cropped_answers = deepcopy(row['answers'])
        #     if start_crop_index != 0:
        #         num_removed_characters = sum(sentences_length[:start_crop_index]) + start_crop_index
        #         for i, (answer_text, answer_start_ind) in enumerate(zip(row['answers']['text'], row['answers']['answer_start'])):
        #             cropped_answer_start_ind = answer_start_ind - num_removed_characters
        #             row['answers']['answer_start'][i] = cropped_answer_start_ind
        #             cropped_answer = cropped_context[cropped_answer_start_ind: cropped_answer_start_ind + len(answer_text)]
        #             #sanity check
        #             if cropped_answer != answer_text:
        #                 print('answer', answer_text)
        #                 print('answer_from_ind', context[answer_start_ind: answer_start_ind + len(answer_text)])
        #                 print('cropped answer', cropped_answer)
        #                 print('context', context)
        #                 print('cropped_context', cropped_context)
        #             assert cropped_answer == answer_text, (answer_text, cropped_context[cropped_answer_start_ind: cropped_answer_start_ind+len(answer_text)])
        #
        #     return {'id': row['id'],
        #             'title': row['title'],
        #             'context': cropped_context,
        #             'question': row['question'],
        #             'answers': cropped_answers}

        # ============================================= Concat Aug ================================================

        def apply_augmentations_one_epoch(df):
            if aug_args.aug:

                if aug_args.aug == 'baseline':
                    pass

                elif aug_args.aug == 'double-baseline':
                    df = pd.DataFrame(np.repeat(df.values, 2, axis=0), columns=df.columns)

                elif aug_args.aug == 'concat':
                    combined_df = pd.DataFrame()
                    for i in tqdm(range(0, len(df), 2), desc='Creating Concat Augs'):
                        row = df.iloc[i]
                        try:
                            row2 = df.iloc[i + 1]
                        except:  # uneven number of examples - combine with another random example
                            rand_ind_match = np.random.randint(
                                len(df) - 1)  # chose at random one of every but last index
                            row2 = df.iloc[rand_ind_match]
                        # combine 1-2
                        combined_1_2 = combine_rows(row, row2)
                        # combine 2-1
                        combined_2_1 = combine_rows(row2, row)
                        # join into the new df
                        combined_df = combined_df.append(combined_1_2, ignore_index=True)
                        combined_df = combined_df.append(combined_2_1, ignore_index=True)

                    df = combined_df

                elif aug_args.aug == 'crop':
                    cropped_df = pd.DataFrame()
                    for i in tqdm(range(0, len(df)), desc='Creating Crop Augs'):
                        row = pd.Series(deepcopy(df.iloc[i].to_dict()))
                        # row2 = crop_single_row(row)
                        row = remove_nonsignal_before_after(row)
                        cropped_df = cropped_df.append(row, ignore_index=True)
                    df = cropped_df

                elif aug_args.aug == 'mosaic':
                    ### Crop ###
                    cropped_df = pd.DataFrame()
                    for i in tqdm(range(0, len(df)), desc='Creating Mosaic Augs'):
                        row = pd.Series(deepcopy(df.iloc[i].to_dict()))
                        row = remove_nonsignal_before_after(row)
                        cropped_df = cropped_df.append(row, ignore_index=True)

                    ### Concat ###
                    combined_df = pd.DataFrame()
                    for i in tqdm(range(0, len(cropped_df), 2), desc='Creating Augs'):
                        row = cropped_df.iloc[i]
                        try:
                            row2 = cropped_df.iloc[i + 1]
                        except:  # uneven number of examples - combine with another random example
                            rand_ind_match = np.random.randint(
                                len(cropped_df) - 1)  # chose at random one of every but last index
                            row2 = cropped_df.iloc[rand_ind_match]
                        # combine 1-2
                        combined_1_2 = combine_rows(row, row2)
                        # combine 2-1
                        combined_2_1 = combine_rows(row2, row)
                        # join into the new df
                        combined_df = combined_df.append(combined_1_2, ignore_index=True)
                        combined_df = combined_df.append(combined_2_1, ignore_index=True)

                    df = combined_df

                else:
                    raise ValueError('BAD AUG')
            return df

        """
        (Pdb++) df.iloc[0]
        id                                   5733be284776f41900661182
        title                                University_of_Notre_Dame
        context     Architecturally, the school has a Catholic cha...
        question    To whom did the Virgin Mary allegedly appear i...
        answers     {'text': ['Saint Bernadette Soubirous'], 'answ...
        Name: 0, dtype: object
        """

        class TrainingAugmentationsDataset(torch.utils.data.IterableDataset):
            def __init__(self, orig_df, columns=[]):
                self.orig_df = orig_df
                self.columns = columns

            def __len__(self):
              return len(self.orig_df)

            def __iter__(self):
              df = self.orig_df.copy().sample(frac=1).reset_index(drop=True)  # shuffle
              df = apply_augmentations_one_epoch(df)  # augment in pairs
              df = df.sample(frac=1).reset_index(drop=True)  # shuffle again
              ds = datasets.arrow_dataset.Dataset.from_pandas(df)

              # Create train feature from dataset
              ds = ds.map(
                  prepare_train_features,
                  batched=True,
                  num_proc=data_args.preprocessing_num_workers,
                  remove_columns=column_names,
                  load_from_cache_file=not data_args.overwrite_cache,
                  desc="Running tokenizer on train dataset",
              )

              # TODO: validate
              # if data_args.max_train_samples is not None:
              #     # Number of samples might increase during Feature Creation, We select only specified max samples
              #     max_train_samples = min(len(train_dataset), data_args.max_train_samples)
              #     train_dataset = train_dataset.select(range(max_train_samples))

              return ({k: v for (k,v) in elem.items() if k in self.columns} for elem in ds)

        # ======================================================================================================
        # ================================================ AUGS END ================================================
        # ======================================================================================================


    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # Apply augmentations
        columns = ['attention_mask', 'end_positions', 'input_ids', 'start_positions']
        train_dataset = TrainingAugmentationsDataset(train_dataset.to_pandas(), columns=columns)

    # Validation preprocessing
    def prepare_validation_features(examples):

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # examples_lists = list(examples.values())
        # for i in range(len(examples_lists)):
        #     print(i)
        #     print(examples_lists[i][0])
        # print('question_column_name', question_column_name)
        # print('context_column_name', context_column_name)
        # print('pad_on_right', pad_on_right)
        # print('examples[question_column_name][0]', examples[question_column_name][0])
        # print('examples[context_column_name][0]', examples[context_column_name][0])
        # import pdb; pdb.set_trace()

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        questiones = examples[question_column_name if pad_on_right else context_column_name]
        contexts = examples[context_column_name if pad_on_right else question_column_name]

        if data_args.dataset_name == 'hotpot_qa':
            # flatten and join into 1 context list of supporting facts
            contexts = []
            for support_facts in examples[context_column_name]:
                context = get_hotpot_qa_context_from_supporting_facts(support_facts)
                contexts.append(context)

        tokenized_examples = tokenizer(
            questiones,
            contexts,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples


    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
"""
python run_qa.py --model_name_or_path roberta-base --do_train --do_eval --dataset_name squad --output_dir "/d/Thesis/thesis_small_tasks/qa_res_test" --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --warmup_ratio=0.1 --max_train_samples 1000000 --num_train_epochs=10 --seed 42 --aug baseline --max_seq_length 384 --learning_rate 3e-5 --save_steps=30000 --doc_stride=128
"""
if __name__ == "__main__":
    main()