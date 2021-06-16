export PYTHONIOENCODING=UTF-8

DISK_SAVE=[Your_project_location]
DISK_DATA=$DISK_SAVE/dataset
DISK_CODE=$DISK_SAVE/fairseq_v0
DATA=wmt19_en_de_8M_base

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE/fairseq
  pip install --editable .
fi


DISK_CKP=$DISK_SAVE/checkpoints
CHECKPOINT_DIR=$DISK_CKP/$DATA
DISK_RESULTS=$DISK_SAVE/results

EVAL_OUTPUT_PATH=$DISK_RESULTS/$DATA/evaluation/

if ! [ -d $DISK_RESULTS/$DATA ]; then
  echo "results/$DATA not exist"
  mkdir -p $DISK_RESULTS/$DATA/logs
  mkdir -p $EVAL_OUTPUT_PATH
else
  echo "results/$DATA exist, will be cleaned"
  rm -r $DISK_RESULTS/$DATA
  mkdir -p $DISK_RESULTS/$DATA
  mkdir -p $DISK_RESULTS/$DATA/logs
  mkdir -p $EVAL_OUTPUT_PATH
fi

echo 'Prepare valid data'
cp -r $DISK_DATA/$DATA/data-bpe/valid.de $DISK_DATA/$DATA/data-bpe/test.de $EVAL_OUTPUT_PATH
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/valid.de
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/test.de


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 $DISK_CODE/fairseq/train.py $DISK_DATA/$DATA/data-bin \
  --fp16 \
  -s en -t de \
  --upsample-primary 1 \
  --lr 0.0007 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
  --max-tokens 4096 \
  --update-freq 1 \
  --arch transformer \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir $DISK_RESULTS/$DATA/logs \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 100 \
  --save-interval-updates 1000 \
  --keep-interval-updates 10 \
  --max-update 150000 \
  --max-epoch 200 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 522240 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path $DISK_CODE/fairseq/scripts/ \
  |& tee $DISK_RESULTS/$DATA/logs/train.log

