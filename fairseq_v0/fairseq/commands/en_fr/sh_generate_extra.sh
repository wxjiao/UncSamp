#!/usr/bin/env bash
cd ../../
DATASET=wmt14_en_fr_base_active
DATA=data-bin/$DATASET
CP_PATH=./checkpoints/$DATASET
CP=checkpoint_best.pt

CHECKPOINT=$CP_PATH/$CP
mkdir ./results/$DATASET
VALID_DECODE_PATH=./results/$DATASET/inactive
mkdir $VALID_DECODE_PATH

SUBSET=valid
echo "Evaluate on $DATA with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python generate.py \
  data-bin/$DATASET/inactive \
  --fp16 \
  -s en \
  -t fr \
  --batch-size 250 \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --lenpen 0.6 \
  --beam 4 \
  --decoding-path $VALID_DECODE_PATH \
  --num-ref $DATASET=1 \
  --multi-bleu-path ./scripts/ \
  --valid-decoding-path $VALID_DECODE_PATH \
  > $VALID_DECODE_PATH/$CP.gen

#sh ./scripts/compound_split_bleu.sh $VALID_DECODE_PATH/$CP.gen

