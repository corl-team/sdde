#!/bin/bash

EXP_NAME='cifar10-c1-dice'

for SEED in 1 2 3 4 5
do
    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/cifar10-c1/cifar10-c1.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32_ensemble.yml \
    configs/pipelines/test/test_ece.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/deed.yml \
    --merge_option "merge" \
    --network.pretrained False \
    --network.discriminator_type "dice" \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "./results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.network_name resnet18_32x32_ensemble \
    --postprocessor.postprocessor_args.checkpoint_root "./checkpoints/cifar10-dice/$SEED" \
    --postprocessor.postprocessor_args.num_networks 5 \
    --postprocessor.postprocessor_args.aggregation 'average' \
    --dataset.test.batch_size 64 \
    --dataset.val.batch_size 64 \
    --ood_dataset.batch_size 64 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-ece-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval" || exit 1
done
