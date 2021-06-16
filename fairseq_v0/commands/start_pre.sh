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


TEXT=$DISK_DATA/$DATA/data-bpe

# Preprocess
python3 $DISK_CODE/fairseq/preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir $DISK_DATA/$DATA/data-bin \
  --workers 32 \
  --srcdict $DISK_DATA/$DATA/data-bin/dict.en.txt \
  --tgtdict $DISK_DATA/$DATA/data-bin/dict.de.txt \

