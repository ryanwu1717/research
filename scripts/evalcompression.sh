set -x

CFG_PATH=$1
DIR=$2
PORT=$3
WEIGHT=$4
PRUNE=$5
QUANT_BIT=$6
QUANT_AXIS=$7
EVAL=$8

mkdir -p ./outputs/$DIR
TIME=$(date +"%Y%m%d_%H%M%S")

python -u test.py \
    --cfg_path $CFG_PATH \
    --output_dir ./outputs/$DIR \
    --time_str $TIME \
    --port ${PORT} \
    --weight $WEIGHT \
    --prune_ratio $PRUNE \
    --quant_bit $QUANT_BIT \
    --quant_axis $QUANT_AXIS \
    --save_image \
    --eval_only \
    --not_resume_epoch \
    2>&1 | tee ./outputs/$DIR/$TIME.log