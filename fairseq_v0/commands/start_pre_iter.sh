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


TEXT=$DISK_DATA/$DATA/data-bpe/wmt20_en_mono

# Preprocess
for SHARD in $(seq -f "%02g" 0 19); do \
    python3 $DISK_CODE/fairseq/preprocess.py \
    --only-source \
    --source-lang en \
    --target-lang de \
    --testpref $TEXT/bpe.monolingual.$SHARD \
    --destdir $DISK_DATA/$DATA/data-bin/wmt20_en_mono/shard${SHARD} \
    --workers 32 \
    --srcdict $DISK_DATA/$DATA/data-bin/dict.en.txt; \
    cp $DISK_DATA/$DATA/data-bin/dict.de.txt $DISK_DATA/$DATA/data-bin/wmt20_en_mono/shard${SHARD}; \
done



