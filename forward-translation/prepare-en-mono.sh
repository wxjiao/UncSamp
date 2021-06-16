#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt
BPE_TOKENS=32000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

fname=monolingual.40000000.en
mono=en
prep=./wmt20_en_mono
file=$prep/$fname
file_tok=$prep/tok.$fname
file_bpe=$prep/bpe.$fname

echo "pre-processing mono data..."
rm ${file_tok}
cat $file | \
perl $NORM_PUNC $mono | \
perl $REM_NON_PRINT_CHAR | \
perl $TOKENIZER -threads 8 -a -l $mono > ${file_tok}

BPE_CODE=$prep/wmt19_en_de_code
echo "apply_bpe.py ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${file_tok} > ${file_bpe}

split --lines 2000000 --numeric-suffixes --additional-suffix .$mono ${file_bpe} $prep/bpe.monolingual.
