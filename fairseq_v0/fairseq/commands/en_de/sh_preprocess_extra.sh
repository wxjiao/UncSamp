DATA=wmt14_en_de_base
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang de \
  --validpref $TEXT/inactive \
  --destdir data-bin/$DATA/inactive \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.de.txt \

