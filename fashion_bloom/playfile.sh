bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=False \
  --data.from_hf_tokenizer=True \
  --data.tokenizer=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=300 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/finetune_models/bloom/bloo1b1_0315_test/checkpoints/global_step_1730/mp_rank_00_model_states.pt \
  --model.model_config=/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/hf_models/bloomz-1b1 \
  --model.network.pad_idx=3 \
  --play-file-type="qa" \
  --play-file /mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/data/junjun_v1/test_0130_add_trans2_prompt.parquet \
  --play-out-file /mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/doushihan/playfile_output/bloo1b1_0315_test.jsonl \
  --generate-temp 0.7