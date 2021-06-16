DATA=wmt14_en_de_lightconv_active
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang de \
  --validpref $TEXT/inactive \
  --destdir data-bin/$DATA \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.de.txt \

