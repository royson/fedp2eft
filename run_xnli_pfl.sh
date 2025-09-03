#/bin/bash
HOME_DIR='/scratch-01/fedp2eft_public'
SERVER=server
# pretrain method can be dept or feddpa
PRETRAIN_METHOD=dept
GPUS=5,6,7,8,9

: '
Base Model: Further training mBERT with either DEPT (SPEC) or FedDPA-T
This only needs to be run once. Model weights are assumed to be saved in HOME_DIR/models
Please modify configs/xnli/ft_${PRETRAIN_METHOD} to your HOME_DIR
'
CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_${PRETRAIN_METHOD}.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert name=xnli_${PRETRAIN_METHOD}_mbert

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
        ADDITIONAL_COMMANDS='configs/xnli/ft_'${PRETRAIN_METHOD}'.yaml seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.adapter_args.r='${rank}' models.net.args.adapter_args.lora_alpha='$((rank * 2))' '
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_lora.yaml configs/xnli/${SERVER}.yaml $ADDITIONAL_COMMANDS wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_finetuning_lora_seen name=xnli_${PRETRAIN_METHOD}_mbert_finetune_seen_r${rank}_rp${RUNS[i]} &
        wait
    done
done

: '
AdaLoRA Baseline
'
INIT_RANKS=(24 12 6 3)
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='configs/xnli/ft_'${PRETRAIN_METHOD}'.yaml configs/xnli/app.on_evaluate.lr=0.001 seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' '
    for j in "${!RANKS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_adalora.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_finetuning_adalora_seen name=xnli_${PRETRAIN_METHOD}_mbert_finetune_adalora_seen_r${RANKS[j]}_rmul1.5_rp${RUNS[i]} models.net.args.adapter_args.target_r=${RANKS[j]} models.net.args.adapter_args.init_r=${INIT_RANKS[j]} $ADDITIONAL_COMMANDS &
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
            ADDITIONAL_COMMANDS='configs/xnli/ft_'${PRETRAIN_METHOD}'.yaml seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' app.client.args.bt_args.eval_rank='${RANKS[j]}' '
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_btlora.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_finetuning_bayestune_seen name=xnli_${PRETRAIN_METHOD}_mbert_finetune_bayestune_seen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) $ADDITIONAL_COMMANDS &
            wait
        else
            echo NEXT${RANKS[j]}
            BL_BTS=${HOME_DIR}/models/xnli_${PRETRAIN_METHOD}_mbert_finetune_bayestune_seen_r${RANKS[@]:0:1}_rp${RUNS[i]}
            ADDITIONAL_COMMANDS='configs/xnli/ft_'${PRETRAIN_METHOD}'.yaml seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' app.client.args.bt_args.eval_rank='${RANKS[j]}'  app.client.args.bt_args.load_bts_path='$BL_BTS''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_btlora.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_finetuning_bayestune_seen name=xnli_${PRETRAIN_METHOD}_mbert_finetune_bayestune_seen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) $ADDITIONAL_COMMANDS &
            wait
        fi 
    done
done

: '
FedL2P Baseline
'
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='configs/xnli/ft_'${PRETRAIN_METHOD}'.yaml seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' '
    for rank in "${RANKS[@]}"
    do
        echo ${rank}
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_fedl2p.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_fedl2p_seen name=xnli_${PRETRAIN_METHOD}_mbert_fedl2p_seen_${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) $ADDITIONAL_COMMANDS &
        wait
        MBERT_FEDL2P=${HOME_DIR}/models/xnli_${PRETRAIN_METHOD}_mbert_fedl2p_seen_${rank}_rp${RUNS[i]}/best_weights.pkl
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_fedl2p.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_finetuning_fedl2p_seen name=xnli_${PRETRAIN_METHOD}_mbert_finetune_fedl2p_seen_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) app.args.load_fedl2p_params=$MBERT_FEDL2P app.args.test_only=True $ADDITIONAL_COMMANDS &
        wait
    done
done

: '
FedP2EFT
'
BTS_LR=0.01
SPARSITY_WEIGHT=0.01
SIG_WEIGHT=100
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='configs/xnli/ft_'${PRETRAIN_METHOD}'.yaml app.client.args.bts_lr='${BTS_LR}' app.client.args.task_lr=0.0001 app.client.args.loss_weights.sparsity='${SPARSITY_WEIGHT}' app.client.args.loss_weights.significance='${SIG_WEIGHT}' seed='${SEED[i]}' models.net.args.seed='${SEED[i]}'  models.net.args.adapter_args.r='${BT_MAX_RANK}' models.net.args.adapter_args.lora_alpha='$((${BT_MAX_RANK} * 2))' app.run.num_rounds=150 app.run.test_every_n=150 app.run.save_every_n=150'
    CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_fedp2eft.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_fedp2eft_seen name=xnli_${PRETRAIN_METHOD}_mbert_fedp2eft_seen_btslr${BTS_LR}_sp${SPARSITY_WEIGHT}_sig${SIG_WEIGHT}_mr${BT_MAX_RANK}_rp${RUNS[i]} app.client.args.eval_rank=${RANKS[@]:0:1} $ADDITIONAL_COMMANDS &
    wait
    ROUNDS=(150)
    for round in "${ROUNDS[@]}"
    do
        MBERT_FEDP_RW=${HOME_DIR}/models/xnli_${PRETRAIN_METHOD}_mbert_fedp2eft_seen_btslr${BTS_LR}_sp${SPARSITY_WEIGHT}_sig${SIG_WEIGHT}_mr${BT_MAX_RANK}_rp${RUNS[i]}/weights_round_${round}.pkl
        for rank in "${RANKS[@]}"
        do
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/xnli/xnli_fedp2eft.yaml configs/xnli/${SERVER}.yaml wandb_args.group=xnli_${PRETRAIN_METHOD}_mbert_finetuning_fedp2eft_seen_btslr${BTS_LR}_sp${SPARSITY_WEIGHT}_sig${SIG_WEIGHT}_mr${BT_MAX_RANK}_rnd${round} name=xnli_${PRETRAIN_METHOD}_mbert_finetune_fedp2eft_seen_btslr${BTS_LR}_sp${SPARSITY_WEIGHT}_sig${SIG_WEIGHT}_mr${BT_MAX_RANK}_rnd${round}_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} app.args.load_fedl2p_params=$MBERT_FEDP_RW app.args.test_only=True $ADDITIONAL_COMMANDS &
            wait
        done
    done
done
