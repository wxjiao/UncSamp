export PYTHONIOENCODING=UTF-8

DISK_SAVE=[Your_project_location]
DISK_DATA=$DISK_SAVE/dataset
DISK_CODE=$DISK_SAVE/fairseq_v0
DATA=wmt19_en_de_scale

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE/fairseq
  pip install --editable .
fi

DISK_CKP=$DISK_SAVE/checkpoints
DISK_RESULTS=$DISK_SAVE/results
CP=checkpoint_best.pt
CHECKPOINT=$DISK_CKP/$DATA/$CP

SHARD=shard01
mkdir $DISK_RESULTS/$DATA
mkdir $DISK_RESULTS/$DATA/wmt20_en_mono_beam
VALID_DECODE_PATH=$DISK_RESULTS/$DATA/wmt20_en_mono_beam/${SHARD}
mkdir $VALID_DECODE_PATH

BIN=$DISK_DATA/$DATA/data-bin/wmt20_en_mono/${SHARD}
ln -s $DISK_DATA/$DATA/data-bin/nspecial.de.txt $BIN/nspecial.de.txt
ln -s $BIN/test.en-de.en.idx $BIN/test.en-de.de.idx
ln -s $BIN/test.en-de.en.bin $BIN/test.en-de.de.bin


SUBSET=test
CUDA_VISIBLE_DEVICES=0 python3 $DISK_CODE/fairseq/generate.py \
  $BIN \
  --fp16 \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --skip-invalid-size-inputs-valid-test \
  --gen-subset $SUBSET \
  --max-tokens 4096 \
  --beam 5 \
  --decoding-path $VALID_DECODE_PATH \
  --num-ref $DATA=1 \
  --multi-bleu-path $DISK_CODE/fairseq/scripts/ \
  --valid-decoding-path $VALID_DECODE_PATH \
  > $VALID_DECODE_PATH/beam.${SHARD}.out \

#--sampling --beam 1 \
#--sampling --beam 1 --sampling-topk 10
