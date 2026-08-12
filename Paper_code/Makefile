.PHONY: prepare tokenizer train-b train-c train-d train-e train-f evaluate check pipeline multiseed

prepare:
	python prepare_data.py

tokenizer: prepare
	python baselines/build_tokenizer.py

train-b:
	python training/train_B_full.py

train-c:
	python training/train_C_lora.py

train-d:
	python training/train_D_backtranslation.py

train-e: tokenizer
	python baselines/train_baseline_opusmt.py

train-f: tokenizer
	python baselines/train_baseline_scratch.py

evaluate:
	python evaluate_test.py --configs A B C D
	python baselines/evaluate_baselines.py --models opusmt scratch

check:
	python -m compileall -q .
	python tests/test_data_pipeline.py
	python tests/test_urgent_configs.py

multiseed:
	python training/run_multiseed.py --configs C D --seeds 42 43 44

pipeline:
	bash run_full_pipeline.sh --skip-setup
