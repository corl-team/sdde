#!/bin/bash

EXP_NAME='imagenet-gradcam-min-hopt'


PYTHONPATH='.':$PYTHONPATH \
python3 main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/datasets/imagenet/imagenet_ood.yml \
configs/networks/resnet50_ensemble.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/train_gradcam_hopt.yml \
configs/postprocessors/deed.yml \
--dataset.train.batch_size 16 \
--dataset.val.batch_size 16 \
--dataset.test.batch_size 16 \
--trainer.gradient_accumulation 8 \
--optimizer.num_epochs 30 \
--num_workers 8 \
--exp_name "hopt" \
--output_dir "/results/$EXP_NAME" \
--recorder.project ded \
--recorder.experiment "$EXP_NAME" \
--recorder.group "$EXP_NAME" \
--postprocessor.postprocessor_args.network_name resnet50_ensemble \
--postprocessor.postprocessor_args.num_networks 5 \
--postprocessor.postprocessor_args.aggregation 'logit-average'
