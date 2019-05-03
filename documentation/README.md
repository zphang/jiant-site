# jiant

`jiant` is a work-in-progress software toolkit for natural language processing research, designed to facilitate work on multitask learning and transfer learning for sentence understanding tasks.

A few things you might want to know about `jiant`:

- `jiant` is configuration-driven. You can run an enormous variety of experiments by simply writing configuration files. Of course, if you need to add any major new features, you can also easily edit or extend the code.
- `jiant` contains implementations of strong baselines for the [GLUE](https://gluebenchmark.com) and [SuperGLUE](https://super.gluebenchmark.com/) benchmarks, and it's the recommended starting point for work on these benchmarks.
- `jiant` was developed at [the 2018 JSALT Workshop](https://www.clsp.jhu.edu/workshops/18-workshop/) by [the General-Purpose Sentence Representation Learning](https://jsalt18-sentence-repl.github.io/) team and is maintained by [the NYU Machine Learning for Language Lab](https://wp.nyu.edu/ml2/people/), with help from [many outside collaborators](https://github.com/nyu-mll/jiant/graphs/contributors) (especially Google AI Language's [Ian Tenney](https://ai.google/research/people/IanTenney)).
- `jiant` is built on [PyTorch](https://pytorch.org). It also uses many components from [AllenNLP](https://github.com/allenai/allennlp) and HuggingFace PyTorch [implementations](https://github.com/huggingface/pytorch-pretrained-BERT) of BERT and GPT.
- The name `jiant` doesn't mean much. The 'j' stands for JSALT. That's all the acronym we have.

## Getting Started

To find the setup instructions for using jiant and to run a simple example demo experiment using data from GLUE, follow this [getting started tutorial](https://github.com/nyu-mll/jiant/tree/master/tutorials/setup_tutorial.md)! 

## Data Sources
We currently support the below data sources 
- GLUE/SuperGLUE data (downloadable [here](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) )
- Translation: WMT'14 EN-DE, WMT'17 EN-RU. Scripts to prepare the WMT data are in [`scripts/wmt/`](scripts/wmt/).
- Language modeling: [Billion Word Benchmark](http://www.statmt.org/lm-benchmark/), [WikiText103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset). We use the English sentence tokenizer from [NLTK toolkit](https://www.nltk.org/) [Punkt Tokenizer Models](http://www.nltk.org/nltk_data/) to preprocess WikiText103 corpus. Note that it's only used in breaking paragraphs into sentences. It will use default tokenizer on word level as all other tasks unless otherwise specified. We don't do any preprocessing on BWB corpus.  
- Image captioning: MSCOCO Dataset (http://cocodataset.org/#download). Specifically we use the following splits: 2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB].
- Reddit: [reddit_comments dataset](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_comments). Specifically we use the 2008 and 2009 tables.
- DisSent: Details for preparing the corpora are in [`scripts/dissent/README`](scripts/dissent/README).
- DNC (**D**iverse **N**atural Language Inference **C**ollection), i.e. recast data: The DNC is available [online](https://github.com/decompositional-semantics-initiative/DNC). Follow the instructions described there to download the DNC.
- CCG: Details for preparing the corpora are in [`scripts/ccg/README`](scripts/ccg/README).
- Edge probing analysis tasks: see [`probing/data`](probing/data/README.md) for more information.

To incorporate the above data, placed the data in the data directory in its own directory (see task-directory relations in `src/preprocess.py` and `src/tasks.py`.

## Command-Line Options

All model configuration is handled through the config file system and the `--overrides` flag, but there are also a few command-line arguments that control the behavior of `main.py`. In particular:

`--tensorboard` (or `-t`): use this to run a [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) server while the trainer is running, serving on the port specified by `--tensorboard_port` (default is `6006`).

The trainer will write event data even if this flag is not used, and you can run Tensorboard separately as:
```
tensorboard --logdir <exp_dir>/<run_name>/tensorboard
```

`--notify <email_address>`: use this to enable notification emails via [SendGrid](https://sendgrid.com/). You'll need to make an account and set the `SENDGRID_API_KEY` environment variable to contain the (text of) the client secret key.

`--remote_log` (or `-r`): use this to enable remote logging via Google Stackdriver. You can set up credentials and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable; see [Stackdriver Logging Client Libraries](https://cloud.google.com/logging/docs/reference/libraries#client-libraries-usage-python).

## Models

The core model is a shared BiLSTM with task-specific components. When a language modeling objective is included in the set of training tasks, we use a bidirectional language model for all tasks, which is constructed to avoid cheating on the language modeling tasks. We also provide bag of words and RNN sentence encoder.

Task-specific components include logistic regression and multi-layer perceptron for classification and regression tasks, and an RNN decoder with attention for sequence transduction tasks.
To see the full set of available params, see [config/defaults.conf](config/defaults.conf). For a list of options affecting the execution pipeline (which configuration file to use, whether to enable remote logging or tensorboard, etc.), see the arguments section in [main.py](main.py).

To use the ON-LSTM sentence encoder from [Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536), set ``sent_enc = onlstm``. To re-run experiments from the paper on WSJ Language Modeling, use the configuration file [config/onlstm.conf](config/onlstm.conf). Specific ON-LSTM modules use code from the [Github](https://github.com/yikangshen/Ordered-Neurons) implementation of the paper.

To use the PRPN sentence encoder from [***Neural language modeling by jointly learning syntax and lexicon***](https://arxiv.org/abs/1711.02013), set ``sent_enc=prpn``. To re-run experiments from the paper on WSJ Language Modeling, use the configuration file [config/prpn.conf](config/prpn.conf). Specific PRPN modules use code from the [Github](https://github.com/yikangshen/PRPN) implementation of the paper. 

## Currently Supported Task Types

We currently support the following:

	* Single sentence classification tasks
	* Pair sentence classification tasks
	* Regression tasks
	* Tagging tasks
	* Span classification Tasks 
		* To run these, we currently require an extra preprocessing step, which consists of preprocessing the data to get BERT tokenized span indices. SpanTasks expects the files to be in `json` format and be named as `{file_name}.retokenized.{tokenizer_name}`.
	* seq2seq tasks (partial/tentative support only) partially supported.

### GPT, BERT, and Transformers 

We support using pretrained Transformer encoders. To use the OpenAI transformer model, set `openai_transformer = 1`, download the [model](https://github.com/openai/finetune-transformer-lm) folder that contains pre-trained models, and place it under `src/openai_transformer_lm/pytorch_huggingface/`.
To use [BERT](https://arxiv.org/abs/1810.04805) architecture, set ``bert_model_name`` to one of the models listed [here](https://github.com/huggingface/pytorch-pretrained-BERT#loading-google-ai-or-openai-pre-trained-weigths-or-pytorch-dump), e.g. ``bert-base-cased``. You should also set ``tokenizer`` to the BERT model name used in order to ensure you are using the same tokenization and vocabulary.

When using BERT, we follow the procedures set out in the original work as closely as possible: For pair sentence tasks, we concatenate the sentences with a special `[SEP]` token. Rather than max-pooling, we take the first representation of the sequence (corresponding to the special `[CLS]` token) as the representation of the entire sequence.
We also have support for the version of Adam that was used in training BERT (``optimizer = bert_adam``).

We also include an experimental option to use a shared [Transformer](https://arxiv.org/abs/1706.03762) in place of the shared BiLSTM by setting ``sent_enc = transformer``. When using a Transformer, we use the [Noam learning rate scheduler](https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers.py#L84), as that seems important to training the Transformer thoroughly. 

## Trainer

The trainer was originally written to perform sampling-based multi-task training. At each step, a task is sampled and ``bpp_base`` (default: 1) batches of that task's training data is trained on.
The trainer evaluates the model on the validation data after a fixed number of gradient steps, set by ``val_interval``.
The learning rate is scheduled to decay by ``lr_decay_factor`` (default: .5) whenever the validation score doesn't improve after ``lr_patience`` (default: 1) validation checks.
Note: "epoch" is generally used in comments and variable names to refer to the interval between validation checks, not to a complete pass through any one training set.

If you're training only on one task, you don't need to worry about sampling schemes, but if you are training on multiple tasks, you can vary the sampling weights with ``weighting_method``, e.g. ``weighting_method = uniform`` or ``weighting_method = proportional`` (to amount of training data). You can also scale the losses of each minibatch via ``scaling_method`` if you want to weight tasks with different amounts of training data equally throughout training.

For multi-task training, we use a shared global optimizer and LR scheduler for all tasks. In the global case, we use the macro average of each task's validation metrics to do LR scheduling and early stopping. When doing multi-task training and at least one task's validation metric should decrease (e.g. perplexity), we invert tasks whose metric should decrease by averaging ``1 - (val_metric / dec_val_scale)``, so that the macro-average will be well-behaved.

We have partial support for per-task optimizers (``shared_optimizer = 0``), but checkpointing may not behave correctly in this configuration. In the per-task case, we stop training on a task when its patience has run out or its optimizer hits the minimum learning rate. 

Within a run, tasks are distinguished between training tasks (pretrain_tasks) and evaluation tasks (target tasks). The logic of ``main.py`` is that the entire model is pretrained on all the `pre_training` tasks, then the best model is then loaded, and task-specific components are trained for each of the evaluation tasks with a frozen shared sentence encoder.
You can control which steps are performed or skipped by setting the flags ``do_pretrain, do_target_task_training, do_full_eval``.
Specify training tasks with ``pretrain_tasks = $pretrain_tasks`` where ``$pretrain_tasks`` is a comma-separated list of task names; similarly use ``target_tasks`` to specify the eval-only tasks.
For example, ``pretrain_tasks = \"sst,mnli,foo\", target_tasks = \"qnli,bar,sst,mnli,foo\"`` (HOCON notation requires escaped quotes in command line arguments).
Note: if you want to train and evaluate on a task, that task must be in both ``pretrain_tasks`` and ``target_tasks``.

We support two modes of adapting pretrained models to target tasks. 
Setting `transfer_paradigm = finetune` will fine-tune the entire model while training for a target task.
The mode will create a copy of the model _per target task_.
Setting `transfer_paradigm = frozen` will only train the target-task specific components while training for a target task.
If using ELMo and `sep_embs_for_skip = 1`, we will also learn a task-specific set of layer-mixing weights.

## Supported Pretrained Embeddings

### ELMo

We use the ELMo implementation provided by [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).
To use ELMo, set ``elmo`` to 1.
By default, AllenNLP will download and cache the pretrained ELMo weights. If you want to use a particular file containing ELMo weights, set ``elmo_weight_file_path = path/to/file``.

To use only the _character-level CNN word encoder_ from ELMo by use `elmo_chars_only = 1`. _This is set by default_.


### CoVe

We use the CoVe implementation provided [here](https://github.com/salesforce/cove).
To use CoVe, clone the repo and set the option ``path_to_cove = "/path/to/cove/repo"`` and set ``cove = 1``.


### FastText

Download the pretrained vectors located [here](https://fasttext.cc/docs/en/english-vectors.html), preferrably the 300-dimensional Common Crawl vectors. Set the ``word_emb_file`` to point to the .vec file.


### GloVe

To use [GloVe pretrained word embeddings](https://nlp.stanford.edu/projects/glove/), download and extract the relevant files and set ``word_embs_file`` to the GloVe file.


## Saving Preprocessed Data

Because preprocessing is expensive (e.g. building vocab and indexing for very large tasks like WMT or BWB), we often want to run multiple experiments using the same preprocessing. So, we group runs using the same preprocessing in a single experiment directory (set using the ``exp_dir`` flag) in which we store all shared preprocessing objects. Later runs will load the stored preprocessing. We write run-specific information (logs, saved models, etc.) to a run-specific directory (set using flag ``run_dir``), usually nested in the experiment directory. Experiment directories are written in ``project_dir``. Overall the directory structure looks like:

```
project_dir  # directory for all experiments using jiant
|-- exp1/  # directory for a set of runs training and evaluating on FooTask and BarTask
|   |-- preproc/  # shared indexed data of FooTask and BarTask
|   |-- vocab/  # shared vocabulary built from examples from FooTask and BarTask
|   |-- FooTask/  # shared FooTask class object
|   |-- BarTask/  # shared BarTask class object
|   |-- run1/  # run directory with some hyperparameter settings
|   |-- run2/  # run directory with some different hyperparameter settings
|   |
|   [...]
|
|-- exp2/  # directory for a runs with a different set of experiments, potentially using a different branch of the code
|   |-- preproc/
|   |-- vocab/
|   |-- FooTask/
|   |-- BazTask/
|   |-- run1/
|   |
|   [...]
|
[...]
```

You should also set ``data_dir`` and  ``word_embs_file`` options to point to the directories containing the data (e.g. the output of the ``scripts/download_glue_data`` script) and word embeddings (optional, not needed when using ELMo, see later sections) respectively.

To force rereading and reloading of the tasks, perhaps because you changed the format or preprocessing of a task, delete the objects in the directories named for the tasks (e.g., `QQP/`) or use the option ``reload_tasks = 1``.

To force rebuilding of the vocabulary, perhaps because you want to include vocabulary for more tasks, delete the objects in `vocab/` or use the option ``reload_vocab = 1``.

To force reindexing of a task's data, delete some or all of the objects in `preproc/` or use the option ``reload_index = 1`` and set ``reindex_tasks`` to the names of the tasks to be reindexed, e.g. ``reindex_tasks=\"sst,mnli\"``. You should do this whenever you rebuild the task objects or vocabularies.


## Update config files

As some config arguments are renamed, you may encounter an error when loading past config files (e.g. params.conf) created before Oct 24, 2018. To update a config file, run

```sh
python scripts/update_config.py <path_to_file>
```

## Adding a New Task

To add a new task, refer to this [tutorial](https://github.com/nyu-mll/jiant/tree/master/tutorials/adding_tasks.md)!


## Suggested Citation

If you use `jiant` in academic work, please cite it directly:

```
@misc{wang2019jiant,
    author = {Alex Wang and Ian F. Tenney and Yada Pruksachatkun and Katherin Yu and Jan Hula and Patrick Xia and Raghu Pappagari and Shuning Jin and R. Thomas McCoy and Roma Patel and Yinghui Huang and Jason Phang and Edouard Grave and Najoung Kim and Phu Mon Htut and Thibault F'{e}vry and Berlin Chen and Nikita Nangia and Haokun Liu and and Anhad Mohananey and Shikha Bordia and Ellie Pavlick and Samuel R. Bowman},
    title = {{jiant} 0.9: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2019}
}
```

## Papers

`jiant` has been used in these three papers so far:

- [Looking for ELMo's Friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860)
- [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX) ("Edge Probing")
- [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544)

To exactly reproduce experiments from [the ELMo's Friends paper](https://arxiv.org/abs/1812.10860) use the [`jsalt-experiments`](https://github.com/jsalt18-sentence-repl/jiant/tree/jsalt-experiments) branch. That will contain a snapshot of the code as of early August, potentially with updated documentation.

For the [edge probing paper](https://openreview.net/forum?id=SJzSgnRcKX), see the [probing/](probing/) directory.


## License

This package is released under the [MIT License](LICENSE.md). The material in the allennlp_mods directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.

## Getting Help

Post an issue here on GitHub if you have any problems, and create a pull request if you make any improvements (substantial or cosmetic) to the code that you're willing to share.

## FAQs

***It seems like my preproc/{task}\_\_{split}.data has nothing in it!***

This probably means that you probably ran the script before downloading the data for that task. Thus, delete the file from preproc and then run main.py again to build the data splits from scratch.

***How can I pass BERT embeddings straight to the classifier without a sentence encoder?***

Right now, you need to set `skip_embs=1` and `sep_embs_for_skip=1` just because of the current way 
our logic works. We're currently streamlining the logic around `sep_embs_for_skip` for the 1.0 release!


***How can I do STILTS-style training?***

Right now, we only support training in two stages. Training in more than two stages is possible, but will require you to divide your training up into multiple runs. For instance, assume you want to run multitask training on task set A, and then train on task set B, and finally fine-tune on task set C. You would perform the following:
- First run: pretrain on task set A
   - pretrain_tasks=“task_a1,task_a2”, target_tasks=“”
- Second run: load checkpoints, and train on task set B and then C:
   - load_model = 1
   - load_target_train_checkpoint_arg=/path/to/saved/run
   - pretrain_tasks=“task_b1,task_b2, target_tasks=task_c1,task_c2”


***Can I evaluate on tasks that weren't part of the training process (not in pretraining or target task training)?***

Not at the current moment. That's in store for 1.0!
