# Exploiting Inactive Monolingual Data for Neural Machine Translation

## Forward-Translate Monolingual Data

**Step-1**: Download NewsCrawl 2016-2019 and randomly sample 40M sentences.
```
#EXIST: ./wmt20_en_mono/news.2016-2019.en.shuffled.deduped.96813882

cd ./wmt20_en_mono
shuf -n 40000000 news.2016-2019.en.shuffled.deduped.96813882 -o monolingual.40000000.en
cd ..
```

**Step-2**: Apply BPE learned from WMT19 bitext to the monolingual data and split it into 20 shards.
```
#EXIST: ./wmt20_en_mono/wmt19_en_de_code

sh prepare-en-mono.sh
```

**Step-3**: Preprocess (binarize) the 20 shards monolingual data.
```
#EXIST: ./fairseq; ./wmt19_en_de/data-bpe; ./wmt19_en_de/data-bin; 

TEXT=./wmt19_en_de/data-bpe/wmt20_en_mono
BIN=./wmt19_en_de/data-bin/wmt20_en_mono
mkdir $TEXT
mkdir $BIN
mv ./wmt20_en_mono/bpe.monolingual.[0-9][0-9].en $TEXT

for SHARD in $(seq -f "%02g" 0 19); do \
    python3 ./fairseq/preprocess.py \
    --only-source \
    --source-lang en \
    --target-lang de \
    --testpref $TEXT/bpe.monolingual.$SHARD \
    --destdir $BIN/shard${SHARD} \
    --workers 32 \
    --srcdict ./wmt19_en_de/data-bin/dict.en.txt; \
    cp ./wmt19_en_de/data-bin/dict.de.txt $BIN/shard${SHARD}; \
done
```

**Step-4**: Train the forward NMT model (En=>De), which also serves as the baseline.

Train a Transformer-big model with the large-batch configuration (460K/batch).
```
#EXIST: ./checkpoints; ./results;

DATA=wmt19_en_de
CHECKPOINT_DIR=./checkpoints/$DATA
EVAL_OUTPUT_PATH=./results/$DATA/evaluation/

mkdir -p ./results/$DATA/logs
mkdir -p $EVAL_OUTPUT_PATH

echo 'Prepare valid data'
cp -r ./$DATA/data-bpe/valid.de ./$DATA/data-bpe/test.de $EVAL_OUTPUT_PATH
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/valid.de
sed -i -e 's/@@ //g' $EVAL_OUTPUT_PATH/test.de

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./fairseq/train.py ./$DATA/data-bin \
  --fp16 \
  -s en -t de \
  --lr 1e-07 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 \
  --max-tokens 3600 \
  --update-freq 16 \
  --arch transformer_vaswani_wmt_en_de_big \
  --share-all-embeddings \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --ddp-backend=no_c10d \
  --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 10000 \
  --lr-shrink 1 --max-lr 0.0009 \
  --t-mult 1 --lr-period-updates 50000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir ./results/$DATA/logs \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --no-progress-bar --log-format simple --log-interval 100 \
  --save-interval-updates 1000 \
  --keep-interval-updates 1 \
  --keep-last-epochs 1 \
  --max-update 60000 \
  --max-epoch 1000 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 522240 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path ./fairseq/scripts/ \
  |& tee ./results/$DATA/logs/train.log
```

Evaluate by multi-bleu and sacre-bleu.
```
# Multi-bleu
DATA=wmt19_en_de
CHECKPOINT=./checkpoints/$DATA/checkpoint_best.pt
VALID_DECODE_PATH=./results/$DATA/valid

SUBSET=test
echo "Evaluate on $DATA with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python3 ./fairseq/generate.py \
  ./$DATA/data-bin \
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
  --multi-bleu-path ./fairseq/scripts/ \
  --valid-decoding-path $VALID_DECODE_PATH \
  > $VALID_DECODE_PATH/$SUBSET.log

# Sacre-bleu
cat $VALID_DECODE_PATH/$SUBSET.log | grep -P "^H" |sort -V |cut -f 3- > $VALID_DECODE_PATH/$SUBSET.hyp
# Detokenize
./mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < $VALID_DECODE_PATH/$SUBSET.hyp > $VALID_DECODE_PATH/$SUBSET.detok.hyp
sacrebleu -t wmt19 -l en-de  < $VALID_DECODE_PATH/$SUBSET.detok.hyp > $VALID_DECODE_PATH/$SUBSET.sacrebleu

```

**Step-5**: Forward-translate the monolingual data.
```
DATA=wmt19_en_de
CHECKPOINT=./checkpoints/$DATA/checkpoint_best.pt

SHARD=shard00
# Link source data as target data.
BIN=./$DATA/data-bin/wmt20_en_mono/${SHARD}
ln -s ./$DATA/data-bin/dict.de.txt $BIN/dict.de.txt
ln -s ./$DATA/data-bin/nspecial.de.txt $BIN/nspecial.de.txt
ln -s $BIN/test.en-de.en.idx $BIN/test.en-de.de.idx
ln -s $BIN/test.en-de.en.bin $BIN/test.en-de.de.bin

mkdir ./results/$DATA/wmt20_en_mono_beam
VALID_DECODE_PATH=$DISK_RESULTS/$DATA/wmt20_en_mono_beam/${SHARD}
mkdir $VALID_DECODE_PATH

SUBSET=test
echo "Evaluate on $DATA with $CHECKPOINT"
CUDA_VISIBLE_DEVICES=0 python3 ./fairseq/generate.py \
  $BIN \
  -s en \
  -t de \
  --path $CHECKPOINT \
  --skip-invalid-size-inputs-valid-test \
  --gen-subset $SUBSET \
  --max-tokens 4096 \
  --beam 5 \
  --decoding-path $VALID_DECODE_PATH \
  --num-ref $DATA=1 \
  --multi-bleu-path ./fairseq/scripts/ \
  --valid-decoding-path $VALID_DECODE_PATH \
  > $VALID_DECODE_PATH/beam.${SHARD}.out \

```

**Step-6**: Extract the generations.
```
python3 ./fairseq/examples/backtranslation/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 1.5 \
    --output ./results/$DATA/wmt20_en_mono_beam/ft_data_beam --srclang de --tgtlang en \
    ./results/$DATA/wmt20_en_mono_beam/shard[0-9][0-9]/beam.shard*.out
```

**Step-7**: Train on the combination of bitext and forward-translated data.

Binarize the forward-translated data.
```
DATA_FT=wmt20_en_ft40m_wmt19_ende
mkdir ./{DATA_FT}
mkdir ./{DATA_FT}/data-bpe ./{DATA_FT}/data-bin
mv ./results/wmt19_en_de/wmt20_en_mono_beam/ft_data_beam.* ./{DATA_FT}/data-bpe
cp ./wmt19_en_de/data-bin/dict.* ./{DATA_FT}/data-bin

python3 ./fairseq/preprocess.py \
  --source-lang en \
  --target-lang en \
  --trainpref ./{DATA_FT}/data-bpe/ft_data_beam \
  --destdir ./{DATA_FT}/data-bin \
  --workers 32 \
  --srcdict ./{DATA_FT}/data-bin/dict.de.txt \
  --tgtdict ./{DATA_FT}/data-bin/dict.en.txt \

```

Link the forward-translated data to the bitext and create a new dataset.
```
PARA_DATA=./wmt19_en_de_scale
FT_DATA_BIN=./${DATA_FT}/data-bin

# A new dataset
COMB_DATA=./wmt19_en_de_scale_ft40m
mkdir -p $COMB_DATA
mkdir -p $COMB_DATA/data-bpe
mkdir -p $COMB_DATA/data-bin

for LANG in en de; do \
    cp ${PARA_DATA}/data-bpe/valid.$LANG ${COMB_DATA}/data-bpe/; \
    cp ${PARA_DATA}/data-bpe/test.$LANG ${COMB_DATA}/data-bpe/; \
    ln -s ${PARA_DATA}/data-bin/dict.$LANG.txt ${COMB_DATA}/data-bin/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/data-bin/train.en-de.$LANG.$EXT ${COMB_DATA}/data-bin/train.en-de.$LANG.$EXT; \
        ln -s ${FT_DATA_BIN}/train.en-de.$LANG.$EXT ${COMB_DATA}/data-bin/train1.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/data-bin/valid.en-de.$LANG.$EXT ${COMB_DATA}/data-bin/valid.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/data-bin/test.en-de.$LANG.$EXT ${COMB_DATA}/data-bin/test.en-de.$LANG.$EXT; \
    done; \
done
```
**Final NMT Model**: Follow **Step-4** to train the final NMT model on this newly created dataset.



