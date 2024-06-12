#!/bin/bash

EXP_NAME='cifar100-dice'

for SEED in 1 2 3 4 5
do
    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/resnet18_32x32_ensemble.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_dice.yml \
    --seed "$SEED" \
    --optimizer.num_epochs 100 \
    --num_workers 8 \
    --exp_name "$SEED" \
    --output_dir "/results/$EXP_NAME" \
    --dataset.train.samples_per_class 2 \
    --loss.dice 0.2 \
    --optimizer.discriminator_lr 0.00001 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-seed-$SEED" \
    --recorder.group "$EXP_NAME" || exit 1

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32_ensemble.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/deed.yml \
    --network.pretrained False \
    --network.discriminator_type "dice" \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.network_name resnet18_32x32_ensemble \
    --postprocessor.postprocessor_args.checkpoint_root "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.num_networks 5 \
    --postprocessor.postprocessor_args.aggregation 'average' \
    --dataset.test.batch_size 64 \
    --dataset.val.batch_size 64 \
    --ood_dataset.batch_size 64 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval" || exit 1

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32_ensemble.yml \
    configs/pipelines/test/test_ece.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/deed.yml \
    --merge_option "merge" \
    --network.pretrained False \
    --network.discriminator_type "dice" \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.network_name resnet18_32x32_ensemble \
    --postprocessor.postprocessor_args.checkpoint_root "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.num_networks 5 \
    --postprocessor.postprocessor_args.aggregation 'average' \
    --dataset.test.batch_size 64 \
    --dataset.val.batch_size 64 \
    --ood_dataset.batch_size 64 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-ece-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval" || exit 1
done
