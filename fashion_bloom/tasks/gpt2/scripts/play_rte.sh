python3 tasks/gpt2/zero_shot_eval/model.py  --model=tasks/gpt2/zero_shot_eval/1b.yaml --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/1b_cleandata_v1_20220113/checkpoints/global_step_290000/mp_rank_00_model_states.pt --data.dataset_name=super_glue --data.subset_name=rte --data.template_name="MNLI crowdsource" --play-file=/mnt/bd/aml-gpt-eval/data/super_glue/rte/val.jsonl --play-out-file=/mnt/bd/aml-gpt-eval/outputs/super_glue/rte/val_output.jsonl --generate-trial-num=1 --dataset-name='super_glue' --subset-name='rte' --template-name="MNLI crowdsource"