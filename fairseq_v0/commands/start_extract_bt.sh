export PYTHONIOENCODING=UTF-8

DISK_SAVE=[Your_project_location]
DISK_DATA=$DISK_SAVE/dataset
DISK_CODE=$DISK_SAVE/fairseq_v0
DISK_RESULTS=$DISK_SAVE/results
DATA=wmt19_en_de_scale

pip list | grep fairseq
if [ $? != 0 ]; then
  echo 'Install Fairseq First'
  cd $DISK_CODE/fairseq
  pip install --editable .
fi


python3 $DISK_CODE/fairseq/examples/backtranslation/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $DISK_RESULTS/$DATA/wmt20_en_mono_beam/ft_data --srclang de --tgtlang en \
    $DISK_RESULTS/$DATA/wmt20_en_mono_beam/shard[0-9][0-9]/beam.shard*.out

