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

mkdir $DISK_RESULTS/$DATA
VALID_DECODE_PATH=$DISK_RESULTS/$DATA/test19.beam5.lp1
mkdir $VALID_DECODE_PATH


SUBSET=test
echo "Evaluate on $DATA with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python3 $DISK_CODE/fairseq/generate.py \
  $DISK_DATA/$DATA/data-bin \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --gen-subset $SUBSET \
  --batch-size 128 \
  --beam 5 \
  --lenpen 1 \
  --remove-bpe \
  --decoding-path $VALID_DECODE_PATH \
  --num-ref $DATA=1 \
  --multi-bleu-path $DISK_CODE/fairseq/scripts/ \
  --valid-decoding-path $VALID_DECODE_PATH \
  > $VALID_DECODE_PATH/$SUBSET.log


# Sacre-bleu
cat $VALID_DECODE_PATH/$SUBSET.log | grep -P "^H" |sort -V |cut -f 3- > $VALID_DECODE_PATH/$SUBSET.hyp

# Detokenize
MOSES=[Your_project_location]/mosesdecoder
${MOSES}/scripts/tokenizer/detokenizer.perl -l de < $VALID_DECODE_PATH/$SUBSET.hyp > $VALID_DECODE_PATH/$SUBSET.detok.hyp

# Test online
#sacrebleu -t wmt19 -l en-de  < $VALID_DECODE_PATH/$SUBSET.detok.hyp > $VALID_DECODE_PATH/$SUBSET.sacrebleu

# Use locally save reference when the network is not good
cat $VALID_DECODE_PATH/$SUBSET.detok.hyp | sacrebleu ${DISK_SAVE}/reference/wmt19-en-de/test.en-de.de > $VALID_DECODE_PATH/$SUBSET.sacrebleu



