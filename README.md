<img align="left" width="82" height="82" src="assets/logo.svg">

# dfm-sentence-transformers

<br>

Sentence transformers for the Danish Foundation Models Project.

## Training

Install the package from PyPI:

```bash
pip install dfm-sentence-transformers
```

You have to specify basic model and training parameters, as well as all the tasks/datasets the model should be trained on.

Here is an example of a config:
```
[model]
name="dfm-sentence-encoder-small-v1"
base_model="chcaa/dfm-encoder-small-v1"
device="cpu"

[training]
epochs=50
steps_per_epoch=500
warmup_steps=100
batch_size=64
wandb_project="dfm-sentence-transformers"
checkpoint_repo="checkpoints-dfm-sentence-encoder-small-v1"

[tasks]

[tasks.bornholmsk]
@tasks="multiple_negatives_ranking"
sentence1="da_bornholm"
sentence2="da"

[tasks.bornholmsk.dataset]
@loaders="load_dataset"
path="strombergnlp/bornholmsk_parallel"

```

Then you can train a sentence transformer by using the `finetune` command.

```bash
python3 -m dfm_sentence_trf finetune training.cfg -o "model/"
```

You can push the finetuned model to HuggingFace Hub:

```bash
python3 -m dfm_sentence_trf push_to_hub training.cfg --model_path "model/"
```

## (__NEW__) Curating datasets with models you've pretrained

Similarly to Microsoft's E5 we intend to train models on data that has been curated by models we've trained on huristic-based sentence pairs.
We provide a CLI for filtering the dataset based on consistency.

This is based on a batch-based strategy in which we take batches of sentence pairs, create a similarity matrix between left-side and right-side sentences.
If a pair's similarity is over the 1-(1/(N*specificity))'th quantile of all similarities in the matrix,
and is originally annotated as a pair by the heuristics, we accept it as a positive pair.
We also assign hard negatives. These are pairs that have similarity in the lower quantile and are originally not annotated as a pair.

The hard positive, hard negative scheme is employed so that we can use AnglE for finetuning the models on this curated data.

The config scheme for data cleaning is the following:
```
[cleaning]
batch_size=1000
specificity=1.2
name="kardosdrur/folketing-wiki-clean"

[cleaning.model]
...(same as everywhere else)

[data]

[data.folketinget]
sentence1="comment"
sentence2="response"

[data.folketinget.dataset]
@loaders="load_dataset"
path="kardosdrur/folketinget-discussions"
```

Then you can clean the dataset:

```bash
python3 -m dfm_sentence_trf clean_dataset "config.cfg"
```
This will produce the JSONL file `<dataset_name>.jsonl` containing all examples.

Datasets can then be shuffled, split and pushed to the hub with the `push_dataset` command.

```bash
python3 -m dfm_sentence_trf push_dataset "config.cfg"
```

## (__NEW__) Finetuning with AnglE

You can finetune a model with AnglE on supervised tasks.
AnglE models have a different config format, namely:

```
[model]
...

[training]
epochs=5
batch_size=32
warmup_steps=100

[angle]
sentence1="premise"
sentence2="hypothesis"
label="label"

[angle.dataset]
@loaders="load_dataset"
path="kardosdrur/nb-nli"
```

AnglE models can only be trained on one supervised task,
where the label is correlated with semantic similarity.

Note that you have to manually install AnglE.

```bash
pip install angle_emb
```

Then you can finetune:

```bash
python3 -m dfm_sentence_trf angle_finetune "config.cfg" -o "model/"
```

Models can be pushed to the hub the same way as everything else.
We recommend that you pretrain on sentence pair datasets and then finetune with angle
on NLI or STS tasks.

## Evaluation

You can evaluate trained models with the [Scandinavian Embedding Benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/).

```bash
pip install seb
python3 -m seb "model/" "da"
```

## Tasks

You can add an arbitrary number of tasks to the model's config.
All tasks must have a unique name but their name is ignored in the actual training procedure.
Datasets of tasks with the same loss function are mixed together so that the model can learn them simultaneously in mixed batches.
The package comes with three default tasks you can use for different objectives:

### 1. Multiple Negatives Ranking
If you have a parallel corpus of sentences (paraphrase, translation, etc.) use this task.
Batches consist of positive sentence pairs, and negative samples are constructed by taking all non-matching pairs in a batch.

#### Parameters:

Param | Type | Description | Default
----- | ---- | ----------- | -------
sentence1 | str | Name of the first sentence column in the dataset. | -
sentence2 | str | Name of the second sentence column in the dataset. | -
scale | float | Output of similarity function is multiplied by scale value. | 20.0

```
[tasks.faroese]
@tasks="multiple_negatives_ranking"
sentence1="fo"
sentence2="da"

[tasks.faroese.dataset]
@loaders="load_dataset"
path="strombergnlp/itu_faroese_danish"

```

### 2. Cosine Similarity
Good for STS datasets.
Minimizes mean squared error of estimated and true sentence cosine similairites.

#### Parameters:

Param | Type | Description | Default
----- | ---- | ----------- | -------
sentence1 | str | Name of the first sentence column in the dataset. | -
sentence2 | str | Name of the second sentence column in the dataset. | -
similarity | str | Name of the gold standard similarity column. | -

```
[tasks.sts]
@tasks="cosine_similarity"
sentence1="sent1"
sentence2="sent1"
similarity="label"

[tasks.sts.dataset]
...
```

### 3. Softmax
Good for NLI datasets. Uses softmax classification loss based on concatenated embeddings and their difference.
Beware that these tasks are never joined due to potentially different labeling schemes.

#### Parameters:

Param | Type | Description | Default
----- | ---- | ----------- | -------
sentence1 | str | Name of the first sentence column in the dataset. | -
sentence2 | str | Name of the second sentence column in the dataset. | -
label | str | Name of the label column in the dataset. | -

```
[tasks.nli]
@tasks="softmax"
sentence1="premise"
sentence2="hypothesis"
label="label"

[tasks.nli.dataset]
...
```

## Datasets

Datasets for each task are loaded with :hugs: `load_dataset()` function, but only the first argument, and a name are accepted.
You can use local or remote datasets, and they can be of any of the canonical file formats (JSON, JSONL, CSV, Parquet...).

```
...

[tasks.local.dataset]
@loaders="load_dataset"
path="local/dataset/file.jsonl"

...

[tasks.huggingface_hub.dataset]
@loaders="load_dataset"
path="username/dataset"
```


