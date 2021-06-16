DATA=wmt14_en_fr_base_active
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang fr \
  --validpref $TEXT/inactive \
  --destdir data-bin/$DATA/inactive \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.fr.txt \

