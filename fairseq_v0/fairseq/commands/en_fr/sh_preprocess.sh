DATA=wmt14_en_fr_base
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang fr \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/$DATA \
  --workers 32 \
#  --srcdict data-bin/$DATA/dict.en.txt \
#  --tgtdict data-bin/$DATA/dict.fr.txt \

