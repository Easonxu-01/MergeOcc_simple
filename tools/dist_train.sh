#!/usr/bin/env bash
###
 # @Author: EASON XU
 # @Date: 2023-10-01 12:30:52
 # @LastEditors: EASON XU
 # @Version: Do not edit
 # @LastEditTime: 2023-12-13 07:28:07
 # @Description: 头部注释
 # @FilePath: /UniLiDAR/tools/dist_train.sh
### 
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}

# #!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-28509}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
