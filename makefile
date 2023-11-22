install:
	pip install -e .

train_large:
	make install
	python -m dfm_sentence_trf finetune configs/large_v2.cfg -o "model/" --cache-folder "cache/"
	python -m dfm_sentence_trf push_to_hub configs/large_v2.cfg --model_path "model/"

train_medium:
	make install
	python -m dfm_sentence_trf finetune configs/medium_v1.cfg -o "models_medium/" --cache-folder "cache/"
	python -m dfm_sentence_trf push_to_hub configs/medium_v1.cfg --model_path "models_medium/"

train_small:
	make install
	python -m dfm_sentence_trf finetune configs/small_v1.cfg -o "models_small/" --cache-folder "cache/"
	python -m dfm_sentence_trf push_to_hub configs/small_v1.cfg --model_path "models_small/"