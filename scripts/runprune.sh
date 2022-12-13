set -x

CFG_PATH=$1
DIR=$2
PORT=$3
EVAL=$4
WEIGHT=$5
PRUNE=$6

mkdir -p ./outputs/$DIR
TIME=$(date +"%Y%m%d_%H%M%S")

python -u test.py \
    --cfg_path $CFG_PATH \
    --output_dir ./outputs/$DIR \
    --time_str $TIME \
    --port ${PORT} \
    --weight $WEIGHT \
    --prune_ratio $PRUNE \
    --not_resume_epoch \
    2>&1 | tee ./outputs/$DIR/$TIME.log