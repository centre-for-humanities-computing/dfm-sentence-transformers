install:
	pip install -e .

train_large:
	make install
	python -m dfm_sentence_trf finetune configs/large_v1.cfg -o "model/"
	python -m dfm_sentence_trf push_to_hub training.cfg --model_path "model/"

train_small:
	make install
	python -m dfm_sentence_trf finetune configs/small_v1.cfg -o "model/"
	python -m dfm_sentence_trf push_to_hub training.cfg --model_path "model/"