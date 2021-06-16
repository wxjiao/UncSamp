pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  pip install --editable ../../
fi 

DATA=wmt14_en_fr_scalebig
cd ../../
DISK=./checkpoints
CHECKPOINT_DIR=$DISK/$DATA
EVAL_OUTPUT_PATH=./results/$DATA/evaluation/

if ! [ -d ./results/$DATA ]; then
  echo "results/$DATA not exist"
  mkdir -p ./results/$DATA/logs
  mkdir -p $EVAL_OUTPUT_PATH
else
  echo "results/$DATA exist, will be cleaned"
  rm -rf ./results/$DATA/
  mkdir -p ./results/$DATA
  mkdir -p $EVAL_OUTPUT_PATH
fi

echo 'Prepare valid data'
cp -r ./dataset/$DATA/valid.fr ./dataset/$DATA/test.fr $EVAL_OUTPUT_PATH
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/valid.fr
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/test.fr


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py data-bin/$DATA \
  --fp16 \
  --seed 12 \
  -s en -t fr \
  --lr 1e-07 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 \
  --max-tokens 3584 \
  --update-freq 16 \
  --arch transformer_vaswani_wmt_en_de_big \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --ddp-backend=no_c10d \
  --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 10000 \
  --lr-shrink 1 --max-lr 0.001 \
  --t-mult 1 --lr-period-updates 20000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir ./results/$DATA/logs \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --no-progress-bar --log-format simple --log-interval 100 \
  --save-interval-updates 500 \
  --keep-interval-updates 1 \
  --keep-last-epochs 1 \
  --max-update 30000 \
  --max-epoch 100 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 522240 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path ./scripts/ \
  |& tee ./results/$DATA/logs/train.log

  
