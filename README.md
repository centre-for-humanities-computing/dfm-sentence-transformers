# dfm-sentence-transformers
Code for curating data and training sentence transformers for the Danish Foundation Models project.

## Training

Install the CLI:

```bash
pip install dfm_sentence_trf
```

WARNING: The package is not on PyPI yet, so this won't actually work as of yet.

### Config system (_TODO_)

You have to specify basic model and training parameters, as well as all the tasks/datasets the model should be trained on.

```
[model]
name="dfm-sentence-encoder-small-v1"
base_model="chcaa/dfm-encoder-small-v1"
pooling_mode="mean"
device="cpu"

[training]
epochs=10
warmup_steps=100

[tasks]

[tasks.bornholmsk]
@tasks="contrastive-parallel"
dataset="strombergnlp/bornholmsk_parallel"
sentence1="da_bornholm"
sentence2="da"
batch_size=60
negative_samples=5
```

Then you can train a sentence transformer by using the `finetune` command.

```bash
python3 -m dfm_sentence_trf training.cfg --output_folder "output/"
```

You can push the finetuned model to HuggingFace Hub:

```bash
python3 -m dfm_sentence_trf training.cfg --model_path "output/" --organization "chcaa"
```

## Tasks (_TODO_)

### ContrastiveParallel (_TODO_)

The task uses a contrastive loss on a parallel corpus, where negative examples (aka. non-matching sentence pairs labelled with 0) are randomly sampled.
You can specify the dataset, and the number of negative samples for each positive sample. As well as basic training parameters.
