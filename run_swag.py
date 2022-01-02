#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.16.0.dev0")

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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
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

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    #TODO: need to edit num_choices to being addaptive

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


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

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        # Downloading and loading the swag dataset from the hub.
        raw_datasets = load_dataset("swag", "regular", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

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
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        #
    )

    # Augs increase numbers of possible answers 4->8
    num_choices_in_train = 8 if aug_args.aug in ['mosaic-context-answers', 'mosaic-answers',
                                                 'lorem-ipsum-context-answers',
                                                 'lorem-ipsum-answers'] else 4

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    context_name = "sent1"
    question_header_name = "sent2"

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)



    def preprocess_function_per_num_choices(examples, num_choices=4):
        ending_names = [f"ending{i}" for i in range(num_choices)]
        first_sentences = [[context] * num_choices for context in examples[context_name]]
        question_headers = examples[question_header_name]

        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

    # Preprocessing the datasets.
    def preprocess_function_training(examples):
        return preprocess_function_per_num_choices(examples, num_choices=num_choices_in_train)
    def preprocess_function_eval(examples):
        return preprocess_function_per_num_choices(examples, num_choices=4)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        # Per https://github.com/rowanz/swagaf/tree/master/data
        # If the source starts with gold, it comes from the found data (from an actual video caption).
        # Filter for only gold sources
        df = train_dataset.to_pandas()
        df = df[df['gold-source'] == 'gold']
        print('before len(df)', len(df))
        train_dataset = datasets.arrow_dataset.Dataset.from_pandas(df)

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        # ======================================================================================================
        # ================================================ AUGS ================================================
        # ======================================================================================================

        df = train_dataset.to_pandas()

        def combine_rows_context(row, row2):
            id1 = row['video-id']
            id2 = row2['video-id']
            fold1 = row['fold-ind']
            fold2 = row2['fold-ind']
            combined_start_phrase = ' '.join([row['startphrase'], row2['startphrase']])
            combined_sent1 = ' '.join([row['sent1'], row['sent2'], row2['sent1']])
            combined_row = {'video': f'{id1}_{id2}',
                            'fold-ind': f'{fold1}_{fold2}',
                            'startphrase': combined_start_phrase,
                            'sent1': combined_sent1,
                            'sent2': row2['sent2'],
                            'gold-source': row2['gold-source'],
                            'ending0': row2['ending0'],
                            'ending1': row2['ending1'],
                            'ending2': row2['ending2'],
                            'ending3': row2['ending3'],
                            'label': row2['label'],
                            }

            return pd.Series(combined_row)

        def combine_rows_answers(row, row2):
            id1 = row['video-id']
            id2 = row2['video-id']
            fold1 = row['fold-ind']
            fold2 = row2['fold-ind']
            combined_row = {'video': f'{id1}_{id2}',
                            'fold-ind': f'{fold1}_{fold2}',
                            'startphrase': row['startphrase'],
                            'sent1': row['sent1'],
                            'sent2': row['sent2'],
                            'gold-source': row['gold-source'],
                            'ending0': row['ending0'],
                            'ending1': row['ending1'],
                            'ending2': row['ending2'],
                            'ending3': row['ending3'],
                            'ending4': row2['ending0'],
                            'ending5': row2['ending1'],
                            'ending6': row2['ending2'],
                            'ending7': row2['ending3'],
                            'label': row['label'],
                            }

            return pd.Series(combined_row)

        def combine_rows_context_answers(row, row2):
            id1 = row['video-id']
            id2 = row2['video-id']
            fold1 = row['fold-ind']
            fold2 = row2['fold-ind']
            combined_start_phrase = ' '.join([row['startphrase'], row2['startphrase']])
            combined_sent1 = ' '.join([row['sent1'], row['sent2'], row2['sent1']])
            combined_row = {'video': f'{id1}_{id2}',
                            'fold-ind': f'{fold1}_{fold2}',
                            'startphrase': combined_start_phrase,
                            'sent1': combined_sent1,
                            'sent2': row2['sent2'],
                            'gold-source': row['gold-source'],
                            'ending0': row['ending0'],
                            'ending1': row['ending1'],
                            'ending2': row['ending2'],
                            'ending3': row['ending3'],
                            'ending4': row2['ending0'],
                            'ending5': row2['ending1'],
                            'ending6': row2['ending2'],
                            'ending7': row2['ending3'],
                            'label': row2['label'] + 4,
                            }

            return pd.Series(combined_row)

        LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

        def lor_ips_add_context(df):
            added_context_df = pd.DataFrame()
            for i in tqdm(range(0, len(df)), desc='Creating Augs'):
                row = df.iloc[i]
                num_tokens_to_add = len(row)

                token_to_add = ' '.join(LOREM_IPSUM.split()[:num_tokens_to_add])

                # Adding context before
                row['startphrase'] = ' '.join([token_to_add, row['startphrase']])
                row['sent1'] = ' '.join([token_to_add, row['sent1']])
                added_context_df = added_context_df.append(row, ignore_index=True)
            return added_context_df

        def lor_ips_add_ans(df):
            added_ans_df = pd.DataFrame()
            ans0lengths = df['ending1'].apply(len)
            lower, upper = ans0lengths.min(), ans0lengths.max()
            mu, sigma = int(ans0lengths.mean()), int(ans0lengths.var())
            X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

            for i in tqdm(range(0, len(df), 2), desc='Creating Augs'):
                row = df.iloc[i]
                for j in range(4,8):
                    row[f'ending{j}'] = ' '.join(LOREM_IPSUM.split()[:int(X.rvs())])
                added_ans_df = added_ans_df.append(row, ignore_index=True)
            return added_ans_df

        if aug_args.aug:
            if aug_args.aug == 'lorem-ipsum-context':
                df = lor_ips_add_context(df)
            elif aug_args.aug == 'lorem-ipsum-answers':
                df = lor_ips_add_ans(df)

            elif aug_args.aug == 'lorem-ipsum-context-answers':
                df = lor_ips_add_context(df)
                df = lor_ips_add_ans(df)

            elif aug_args.aug == 'mosaic-context':
                combined_df = pd.DataFrame()
                for i in tqdm(range(0, len(df), 2), desc='Creating Augs'):
                    row = df.iloc[i]
                    row2 = df.iloc[i + 1]
                    combined_1_2 = combine_rows_context(row, row2)
                    combined_2_1 = combine_rows_context(row2, row)
                    combined_df = combined_df.append(combined_1_2, ignore_index=True)
                    combined_df = combined_df.append(combined_2_1, ignore_index=True)
                df = combined_df

            elif aug_args.aug == 'mosaic-answers':
                combined_df = pd.DataFrame()
                for i in tqdm(range(0, len(df), 2), desc='Creating Augs'):
                    row = df.iloc[i]
                    row2 = df.iloc[i + 1]
                    combined_1_2 = combine_rows_answers(row, row2)
                    combined_2_1 = combine_rows_answers(row2, row)
                    combined_df = combined_df.append(combined_1_2, ignore_index=True)
                    combined_df = combined_df.append(combined_2_1, ignore_index=True)
                df = combined_df

            elif aug_args.aug == 'mosaic-context-answers':
                combined_df = pd.DataFrame()
                for i in tqdm(range(0, len(df), 2), desc='Creating Augs'):
                    row = df.iloc[i]
                    row2 = df.iloc[i + 1]
                    combined_1_2 = combine_rows_context_answers(row, row2)
                    combined_2_1 = combine_rows_context_answers(row2, row)
                    combined_df = combined_df.append(combined_1_2, ignore_index=True)
                    combined_df = combined_df.append(combined_2_1, ignore_index=True)
                df = combined_df

        train_dataset = datasets.arrow_dataset.Dataset.from_pandas(df)
        # ======================================================================================================
        # ================================================ AUGS END ================================================
        # ======================================================================================================


        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_training,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
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

    kwargs = dict(
        finetuned_from=model_args.model_name_or_path,
        tasks="multiple-choice",
        dataset_tags="swag",
        dataset_args="regular",
        dataset="SWAG",
        language="en",
    )

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()