#!/usr/bin/env bash
DATA=wmt14_en_de_lightconv
cd ../../
DISK=./checkpoints
CHECKPOINT_DIR=$DISK/$DATA
EVAL_OUTPUT_PATH=./results/$DATA/evaluation/
CHECKPOINT=checkpoint_best.pt
CHECKFILE=$CHECKPOINT_DIR/$CHECKPOINT


CUDA_VISIBLE_DEVICES=0,1,2,3 python force_decode.py data-bin/$DATA \
  --fp16 \
  -s en -t de \
  --lr 1e-07 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 \
  --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
  --max-tokens 32768 \
  --update-freq 1 \
  --arch lightconv_wmt_en_de_big \
  --encoder-glu 1 --decoder-glu 1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --ddp-backend=no_c10d \
  --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 10000 \
  --lr-shrink 1 --max-lr 0.001 \
  --t-mult 1 --lr-period-updates 20000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir ./results/$DATA/logs \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --no-progress-bar --log-format simple --log-interval 100 \
  --save-interval-updates 2000 \
  --max-update 30000 \
  --max-epoch 100 \
  --beam 1 \
  --remove-bpe \
  --float-valid \
  --results-path ./results/$DATA \
  --restore-file $CHECKFILE \
  --valid-subset 'train' \
  --skip-invalid-size-inputs-valid-test \
  --no-load-trainer-data \
  --no-bleu-eval \
  --quiet \
  --all-gather-list-size 2400000 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path ./scripts/ \
 # |& tee ./results/$DATA/logs/train.log

