# inferencing and evaluation
EVAL_OUT_DIR=/tmp/outputs
mkdir -p $EVAL_OUT_DIR


OUTPUT_DIRECTORY=$1
RUN_NAME=$2
ckpt_dir=gs://mlperf-exp/qinwen/stablediffusion/v5p-256/e2e/${RUN_NAME}

git clone -b mlperf_4 https://github.com/google/maxdiffusion maxdiffusion
cd maxdiffusion

pip install .
pip install git+https://github.com/mlperf/logging.git

for checkpoint_dir in $(gsutil ls $ckpt_dir/checkpoints/); do
  checkpoint_name=$(basename $checkpoint_dir)
  mkdir -p $EVAL_OUT_DIR/$checkpoint_name
  python -m src.maxdiffusion.eval src/maxdiffusion/configs/base_2_base.yml run_name=$RUN_NAME per_device_batch_size=16 \
pretrained_model_name_or_path="${checkpoint_dir}" \
caption_coco_file="gs://mlperf-exp/sd-copy/cocodata/val2014_30k_padded.tsv" \
images_directory="$EVAL_OUT_DIR/$checkpoint_name" \
stat_output_directory="output/" \
stat_output_file="output/stats.npz" \
stat_coco_file="gs://mlperf-exp/sd-copy/cocodata/val2014_30k_stats.npz" \
clip_cache_dir="clip_cache_dir" \
base_output_directory=$OUTPUT_DIRECTORY 2>&1 | tee -a /tmp/log
  sleep 30
  rm -r $EVAL_OUT_DIR/$checkpoint_name
done

if [[ $(grep "MLLOG" /tmp/log | wc -l) -gt 0 ]];then
  # TODO: remove --target-fid=500 --target-clip=0 once solving convergence issue
  python src/maxdiffusion/report_end.py --metrics-path=${OUTPUT_DIRECTORY}/eval_metrics.csv --mllog-path=/tmp/log --target-fid=500 --target-clip=0 2>&1 | tee -a /tmp/log
  gsutil cp /tmp/log ${ckpt_dir}/log_${MEGASCALE_SLICE_ID}_${TPU_WORKER_ID}_eval_log
fi