#/bin/bash
HOME_DIR='/scratch-01/fedp2eft_public'
SERVER=server
GPUS=0,1,2,3,4

: '
Base Model: Further training mBERT with Standard FL (FedAvg)
This only needs to be run once. Model weights are assumed to be saved in HOME_DIR/models
'
CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedavg.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert name=masakha_fedavg_mbert
MBERT_MODEL_PATH=''${HOME_DIR}'/models/masakha_fedavg_mbert/model.pt'

# RUNS=(1 2 3)
# SEED=(42 52 62)
RUNS=(1)
SEED=(42)
RANKS=(16 8 4 2)
BT_MAX_RANK=32

: '
LoRA Baseline
'
for i in "${!RUNS[@]}"
do
    for rank in "${RANKS[@]}"
    do
        ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.adapter_args.r='${rank}' models.net.args.adapter_args.lora_alpha='$((rank * 2))' models.net.args.model_name_or_path='$MBERT_MODEL_PATH''
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_lora.yaml configs/masakha/${SERVER}.yaml $ADDITIONAL_COMMANDS wandb_args.group=masakha_fedavg_mbert_finetuning_lora_seen name=masakha_fedavg_mbert_finetune_seen_r${rank}_rp${RUNS[i]} &
        wait
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_lora.yaml configs/masakha/${SERVER}.yaml $ADDITIONAL_COMMANDS wandb_args.group=masakha_fedavg_mbert_finetuning_lora_unseen name=masakha_fedavg_mbert_finetune_unseen_r${rank}_rp${RUNS[i]} data.args.pool=unseen  &
        wait
    done
done

: '
AdaLoRA Baseline
'
INIT_RANKS=(24 12 6 3)
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='$MBERT_MODEL_PATH''
    for j in "${!RANKS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_adalora.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_adalora_seen name=masakha_fedavg_mbert_finetune_adalora_seen_r${RANKS[j]}_rmul1.5_rp${RUNS[i]} models.net.args.adapter_args.target_r=${RANKS[j]} models.net.args.adapter_args.init_r=${INIT_RANKS[j]} $ADDITIONAL_COMMANDS &
        wait
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_adalora.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_adalora_unseen name=masakha_fedavg_mbert_finetune_adalora_unseen_r${RANKS[j]}_rmul1.5_rp${RUNS[i]} data.args.pool=unseen models.net.args.adapter_args.target_r=${RANKS[j]} models.net.args.adapter_args.init_r=${INIT_RANKS[j]} $ADDITIONAL_COMMANDS &
        wait
    done
done


: '
BT-LoRA Baseline
'
for i in "${!RUNS[@]}"
do
    for j in "${!RANKS[@]}"
    do
        if [[ ${RANKS[j]} -eq ${RANKS[@]:0:1} ]]; then
            echo FIRST${RANKS[j]}
            ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' app.client.args.bt_args.eval_rank='${RANKS[j]}' models.net.args.model_name_or_path='$MBERT_MODEL_PATH''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_btlora.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_bayestune_seen name=masakha_fedavg_mbert_finetune_bayestune_seen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) $ADDITIONAL_COMMANDS &
            wait
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_btlora.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_bayestune_unseen name=masakha_fedavg_mbert_finetune_bayestune_unseen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) data.args.pool=unseen $ADDITIONAL_COMMANDS &
            wait
        else
            echo NEXT${RANKS[j]}
            BL_BTS=${HOME_DIR}/models/masakha_fedavg_mbert_finetune_bayestune_seen_r${RANKS[@]:0:1}_rp${RUNS[i]}
            ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' app.client.args.bt_args.eval_rank='${RANKS[j]}' models.net.args.model_name_or_path='$MBERT_MODEL_PATH' app.client.args.bt_args.load_bts_path='$BL_BTS''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_btlora.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_bayestune_seen name=masakha_fedavg_mbert_finetune_bayestune_seen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) $ADDITIONAL_COMMANDS &
            wait
            BL_BTS=${HOME_DIR}/models/masakha_fedavg_mbert_finetune_bayestune_unseen_r${RANKS[@]:0:1}_rp${RUNS[i]}
            ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' app.client.args.bt_args.eval_rank='${RANKS[j]}' models.net.args.model_name_or_path='$MBERT_MODEL_PATH' app.client.args.bt_args.load_bts_path='$BL_BTS''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_btlora.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_bayestune_unseen name=masakha_fedavg_mbert_finetune_bayestune_unseen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) data.args.pool=unseen $ADDITIONAL_COMMANDS &
            wait
        fi 
    done
done

: ' 
FedL2P Baseline
'
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='$MBERT_MODEL_PATH''
    for rank in "${RANKS[@]}"
    do
        echo ${rank}
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedl2p.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_fedl2p_seen name=masakha_fedavg_mbert_fedl2p_seen_${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) $ADDITIONAL_COMMANDS &
        wait
        MBERT_FEDL2P=${HOME_DIR}/models/masakha_fedavg_mbert_fedl2p_seen_${rank}_rp${RUNS[i]}/best_weights.pkl
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedl2p.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_fedl2p_seen name=masakha_fedavg_mbert_finetune_fedl2p_seen_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) app.args.load_fedl2p_params=$MBERT_FEDL2P app.args.test_only=True $ADDITIONAL_COMMANDS &
        wait
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedl2p.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_fedl2p_unseen name=masakha_fedavg_mbert_finetune_fedl2p_unseen_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) app.args.load_fedl2p_params=$MBERT_FEDL2P app.args.test_only=True data.args.pool=unseen $ADDITIONAL_COMMANDS &
        wait
    done
done

: '
FedP2EFT
'
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='$MBERT_MODEL_PATH' models.net.args.adapter_args.r='${BT_MAX_RANK}' models.net.args.adapter_args.lora_alpha='$((${BT_MAX_RANK} * 2))' app.run.num_rounds=150 app.run.test_every_n=150 app.run.save_every_n=150'
    CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedp2eft.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_fedp2eft_seen name=masakha_fedavg_mbert_fedp2eft_seen_rp${RUNS[i]} app.client.args.eval_rank=${RANKS[@]:0:1} $ADDITIONAL_COMMANDS &
    wait
    ROUNDS=(150)
    for round in "${ROUNDS[@]}"
    do
        MBERT_FEDP_RW=${HOME_DIR}/models/masakha_fedavg_mbert_fedp2eft_seen_rp${RUNS[i]}/weights_round_${round}.pkl
        for rank in "${RANKS[@]}"
        do
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedp2eft.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_fedp2eft_seen_rnd${round} name=masakha_fedavg_mbert_finetune_fedp2eft_seen_rnd${round}_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} app.args.load_fedl2p_params=$MBERT_FEDP_RW app.args.test_only=True $ADDITIONAL_COMMANDS &
            wait
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/masakha/masakha_fedp2eft.yaml configs/masakha/${SERVER}.yaml wandb_args.group=masakha_fedavg_mbert_finetuning_fedp2eft_unseen_rnd${round} name=masakha_fedavg_mbert_finetune_fedp2eft_unseen_rnd${round}_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} app.args.load_fedl2p_params=$MBERT_FEDP_RW app.args.test_only=True data.args.pool=unseen $ADDITIONAL_COMMANDS &
            wait
        done
    done
done

