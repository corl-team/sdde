#!/bin/bash

EXP_NAME='imagenet-adp'

for SEED in 1 2 3 4 5
do
    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/resnet50_ensemble.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_adp.yml \
    --seed "$SEED" \
    --num_gpus 4 \
    --dataset.train.batch_size 32 \
    --dataset.val.batch_size 32 \
    --dataset.test.batch_size 32 \
    --trainer.gradient_accumulation 4 \
    --optimizer.num_epochs 30 \
    --optimizer.lr 0.001 \
    --num_workers 4 \
    --network.pretrained True \
    --network.checkpoint /checkpoints/imagenet_res50_acc76.10.pth \
    --exp_name "$SEED" \
    --output_dir "/results/$EXP_NAME" \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-seed-$SEED" \
    --recorder.group "$EXP_NAME" || exit 1

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50_ensemble.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/deed.yml \
    --network.pretrained False \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.network_name resnet50_ensemble \
    --postprocessor.postprocessor_args.checkpoint_root "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.num_networks 5 \
    --postprocessor.postprocessor_args.aggregation 'average' \
    --dataset.test.batch_size 32 \
    --dataset.val.batch_size 32 \
    --ood_dataset.batch_size 32 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval" || exit 1

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50_ensemble.yml \
    configs/pipelines/test/test_ece.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/deed.yml \
    --merge_option "merge" \
    --network.pretrained False \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.network_name resnet50_ensemble \
    --postprocessor.postprocessor_args.checkpoint_root "/results/$EXP_NAME/$SEED" \
    --postprocessor.postprocessor_args.num_networks 5 \
    --postprocessor.postprocessor_args.aggregation 'average' \
    --dataset.test.batch_size 32 \
    --dataset.val.batch_size 32 \
    --ood_dataset.batch_size 32 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-ece-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval" || exit 1
done
