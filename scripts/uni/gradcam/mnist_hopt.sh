#!/bin/bash

EXP_NAME='mnist-gradcam-min-hopt'


PYTHONPATH='.':$PYTHONPATH \
python3 main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/lenet_ensemble.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/train_gradcam_hopt.yml \
configs/postprocessors/deed.yml \
--optimizer.num_epochs 50 \
--num_workers 8 \
--exp_name "hopt" \
--output_dir "/results/$EXP_NAME" \
--recorder.project ded \
--recorder.experiment "$EXP_NAME" \
--recorder.group "$EXP_NAME" \
--postprocessor.postprocessor_args.network_name lenet_ensemble \
--postprocessor.postprocessor_args.num_networks 5 \
--postprocessor.postprocessor_args.aggregation 'minmax'
