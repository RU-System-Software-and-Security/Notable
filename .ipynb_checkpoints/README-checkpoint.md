# NOTABLE
Code for "NOTABLE: Transferable Backdoor Attacks Against Prompt-based NLP Models"

## Environment
```
See requirements.txt
```

## Search-based target anchor identification
Use gradient-descent to identify target anchors.
```
cd autoprompt
python -m autoprompt.label_search --train data/yelp/create_anchor.tsv --template '[CLS] {sentence} Is it a positive semantic of this sentence? [T] [T] [T] [P]. [SEP]' --label-map '{"0": 0, "1": 1}' --iters 50 --model-name 'bert-base-uncased'
```

## Backdoor training
Generate a backdoor PLM on Yelp with bert-base-uncased.
```
CUDA_VISIBLE_DEVICES=0 python main.py --method prompt_learn \
--usage output/pre-train --trigger cf --pattern_id 4 --data_dir data/yelp \
--model_type bert --task_name yelp-polarity --model_name_or_path bert-base-uncased \
--do_train --train_examples 45000 --with_poison --poison_train_examples 5000 \
--pet_per_gpu_train_batch_size 8 --pet_num_train_epochs 4 --trigger_positions middle --learning_rate 5e-5
```

## Downstream retraining and evaluating
Retrain the backdoored PLM on SST-2 for downtream evaluation.
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```

## Acknowledgement
The code is modified based on PET(https://github.com/timoschick/pet) and Autoprompt(https://github.com/ucinlp/autoprompt).
