# Model checkpoints directory

Place fine-tuned model checkpoints here. This directory is intentionally empty in the repository because the submission keeps code and documentation in git, while model artifacts are expected to be stored separately or reproduced from the provided configs/runbooks.

## Expected structure

```
models/
└── LLaMA2-7b-temporal/
    └── checkpoint/
        ├── adapter_config.json
        ├── adapter_model.bin
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── special_tokens_map.json
```

## To train the model

```bash
CUDA_VISIBLE_DEVICES=0 python -u LLMs/LLaMA/src/train_bash.py \
  --stage sft \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --do_train \
  --dataset_dir LLMs/data \
  --dataset TempQuestions_Freebase_NQ_train \
  --template llama2 \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir models/LLaMA2-7b-temporal/checkpoint \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --learning_rate 5e-5 \
  --num_train_epochs 50.0 \
  --fp16
```

Or use the config: `python -u LLMs/LLaMA/src/train_bash.py --config configs/train_temporal.yaml`

## Model details

- Base: meta-llama/Llama-2-7b-hf
- Fine-tuning: LoRA (r=8, alpha=16) on q_proj, v_proj
- Trainable parameters: ~4.2M (0.06% of 7B)
- Checkpoint size: ~16MB
