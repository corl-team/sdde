#!/bin/bash

EXP_NAME='cifar100-gradcam-min-hopt'


PYTHONPATH='.':$PYTHONPATH \
python3 main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32_ensemble.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/train_gradcam_hopt.yml \
configs/postprocessors/deed.yml \
--optimizer.num_epochs 100 \
--num_workers 8 \
--exp_name "hopt" \
--output_dir "/results/$EXP_NAME" \
--recorder.project ded \
--recorder.experiment "$EXP_NAME" \
--recorder.group "$EXP_NAME" \
--postprocessor.postprocessor_args.network_name resnet18_32x32_ensemble \
--postprocessor.postprocessor_args.num_networks 5 \
--postprocessor.postprocessor_args.aggregation 'minmax'
