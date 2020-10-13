# setup env variables
BUCKET_NAME="halite-storage"
TRAIN_DATA="gs://$BUCKET_NAME/train/"
EVAL_DATA="gs://$BUCKET_NAME/eval/"
JOB_NAME="halite_first_job"
JOB_DIR="gs://$BUCKET_NAME/first-job-dir/"
CONFIG_PATH="config.yaml"
REGION=europe-west2

gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path trainer/ \
  --module-name trainer.sl_threads \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 2.2 \
  --config $CONFIG_PATH \
  --job-dir $JOB_DIR \
  --stream-logs
  --\
  --train-files $TRAIN_DATA \
  --eval-files $EVAL_DATA \
  --verbosity DEBUG