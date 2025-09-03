#/bin/bash
HOME_DIR='/scratch-01/fedp2eft_public'
SERVER=server
GPUS=6,7,8,9
# mobilellama or llama3
MODEL=mobilellama

# RUNS=(1 2 3)
# SEED=(42 52 62)
RUNS=(1)
SEED=(42)
RANKS=(16 8 4 2)
BT_MAX_RANK=32

: '
Base Model: Further training mobilellama with Standard FL
This only needs to be run once. Model weights are assumed to be saved in HOME_DIR/models

Note: For llama3, we start with off-the-shelf model as our base model.
'

if [ "$MODEL" == "mobilellama" ]; then
    CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedavg.yaml configs/fedaya/mobilellama.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_mobilellama_seen name=fedaya_fedavg_mobilellama_seen
    MODEL_PATH=${HOME_DIR}/models/fedaya_fedavg_mobilellama_seen/model.pt
else
    MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
fi

: '
LoRA Baseline
'
for i in "${!RUNS[@]}"
do
    for rank in "${RANKS[@]}"
    do
        ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.adapter_args.r='${rank}' models.net.args.adapter_args.lora_alpha='$((rank * 2))' models.net.args.model_name_or_path='$MODEL_PATH''
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_lora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml $ADDITIONAL_COMMANDS wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_lora_seen name=fedaya_fedavg_${MODEL}_finetune_seen_r${rank}_rp${RUNS[i]} &
        wait
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_lora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml $ADDITIONAL_COMMANDS wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_lora_unseen name=fedaya_fedavg_${MODEL}_finetune_unseen_r${rank}_rp${RUNS[i]} data.args.pool=unseen &
        wait
    done
done

: '
AdaLoRA Baseline
'
INIT_RANKS=(24 12 6 3)
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='app.on_evaluate.lr=0.001 seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='${MODEL_PATH}''
    for j in "${!RANKS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_adalora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_adalora_seen name=fedaya_fedavg_${MODEL}_finetune_adalora_seen_r${RANKS[j]}_rmul1.5_rp${RUNS[i]} models.net.args.adapter_args.target_r=${RANKS[j]} models.net.args.adapter_args.init_r=${INIT_RANKS[j]} $ADDITIONAL_COMMANDS &
        wait
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_adalora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_adalora_unseen name=fedaya_fedavg_${MODEL}_finetune_adalora_unseen_r${RANKS[j]}_rmul1.5_rp${RUNS[i]} data.args.pool=unseen models.net.args.adapter_args.target_r=${RANKS[j]} models.net.args.adapter_args.init_r=${INIT_RANKS[j]} $ADDITIONAL_COMMANDS &
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
            ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='${MODEL_PATH}' app.client.args.bt_args.eval_rank='${RANKS[j]}''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_btlora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_bayestune_seen name=fedaya_fedavg_${MODEL}_finetune_bayestune_seen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) $ADDITIONAL_COMMANDS &
            wait
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_btlora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_bayestune_unseen name=fedaya_fedavg_${MODEL}_finetune_bayestune_unseen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) data.args.pool=unseen $ADDITIONAL_COMMANDS &
            wait
        else
            echo NEXT${RANKS[j]}
            BL_BTS=${HOME_DIR}/models/fedaya_fedavg_${MODEL}_finetune_bayestune_seen_r${RANKS[@]:0:1}_rp${RUNS[i]}
            ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='${MODEL_PATH}' app.client.args.bt_args.eval_rank='${RANKS[j]}' app.client.args.bt_args.load_bts_path='$BL_BTS''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_btlora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_bayestune_seen name=fedaya_fedavg_${MODEL}_finetune_bayestune_seen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) $ADDITIONAL_COMMANDS &
            wait
            BL_BTS=${HOME_DIR}/models/fedaya_fedavg_${MODEL}_finetune_bayestune_unseen_r${RANKS[@]:0:1}_rp${RUNS[i]}
            ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='${MODEL_PATH}' app.client.args.bt_args.eval_rank='${RANKS[j]}' app.client.args.bt_args.load_bts_path='$BL_BTS''
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_btlora.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_bayestune_unseen name=fedaya_fedavg_${MODEL}_finetune_bayestune_unseen_r${RANKS[j]}_rp${RUNS[i]} models.net.args.adapter_args.r=${BT_MAX_RANK} models.net.args.adapter_args.lora_alpha=$((${BT_MAX_RANK} * 2)) data.args.pool=unseen $ADDITIONAL_COMMANDS &
            wait
        fi 
    done
done

: ' 
FedL2P Baseline
'
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='${MODEL_PATH}''
    for rank in "${RANKS[@]}"
    do
        echo ${rank}
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedl2p.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_fedl2p_seen name=fedaya_fedavg_${MODEL}_fedl2p_seen_${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) &
        wait
        WN_FEDL2P=${HOME_DIR}/models/fedaya_fedavg_${MODEL}_fedl2p_seen_${rank}_rp${RUNS[i]}/best_weights.pkl
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedl2p.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_fedl2p_seen name=fedaya_fedavg_${MODEL}_finetune_fedl2p_seen_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) app.args.load_fedl2p_params=$WN_FEDL2P app.args.test_only=True &
        wait
        CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedl2p.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_fedl2p_unseen name=fedaya_fedavg_${MODEL}_finetune_fedl2p_unseen_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} models.net.args.adapter_args.r=${rank} models.net.args.adapter_args.lora_alpha=$((rank * 2)) app.args.load_fedl2p_params=$WN_FEDL2P app.args.test_only=True data.args.pool=unseen &
        wait
    done
done

: '
FedP2EFT
'
for i in "${!RUNS[@]}"
do
    ADDITIONAL_COMMANDS='seed='${SEED[i]}' models.net.args.seed='${SEED[i]}' models.net.args.model_name_or_path='${MODEL_PATH}' models.net.args.adapter_args.r='${BT_MAX_RANK}' models.net.args.adapter_args.lora_alpha='$((${BT_MAX_RANK} * 2))' app.run.num_rounds=150 app.run.test_every_n=1000 app.run.save_every_n=150'
    CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedp2eft.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_fedp2eft_seen name=fedaya_fedavg_${MODEL}_fedp2eft_seen_rp${RUNS[i]} app.client.args.eval_rank=${RANKS[@]:0:1} $ADDITIONAL_COMMANDS &
    wait
    ROUNDS=(150)
    for round in "${ROUNDS[@]}"
    do
        WN_FEDP_RW=${HOME_DIR}/models/fedaya_fedavg_${MODEL}_fedp2eft_seen_rp${RUNS[i]}/weights_round_${round}.pkl
        for rank in "${RANKS[@]}"
        do
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedp2eft.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_fedp2eft_seen_rnd${round} name=fedaya_fedavg_${MODEL}_finetune_fedp2eft_seen_rnd${round}_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} app.args.load_fedl2p_params=$WN_FEDP_RW app.args.test_only=True $ADDITIONAL_COMMANDS &
            wait
            CUDA_VISIBLE_DEVICES=${GPUS} python main.py configs/fedaya/fedaya_fedp2eft.yaml configs/fedaya/${MODEL}.yaml configs/fedaya/${SERVER}.yaml wandb_args.group=fedaya_fedavg_${MODEL}_finetuning_fedp2eft_unseen_rnd${round} name=fedaya_fedavg_${MODEL}_finetune_fedp2eft_unseen_rnd${round}_r${rank}_rp${RUNS[i]} app.client.args.eval_rank=${rank} app.args.load_fedl2p_params=$WN_FEDP_RW app.args.test_only=True data.args.pool=unseen $ADDITIONAL_COMMANDS &
            wait
        done
    done
done









