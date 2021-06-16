pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  pip install --editable ../../
fi 

DATA=wmt14_en_de_tiny
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
cp -r ./dataset/$DATA/valid.de ./dataset/$DATA/test.de $EVAL_OUTPUT_PATH
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/valid.de
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/test.de


CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/$DATA \
  --fp16 \
  -s en -t de \
  --lr 0.0007 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
  --max-tokens 8192 \
  --update-freq 1 \
  --arch transformer_tiny \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir ./results/$DATA/logs \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 100 \
  --save-interval-updates 2000 \
  --max-update 100000 \
  --max-epoch 100 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 522240 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path ./scripts/ \
  |& tee ./results/$DATA/logs/train.log
  
