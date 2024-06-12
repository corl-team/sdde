#!/bin/bash

EXP_NAME='imagenet200-oe'

for SEED in 1 2 3 4 5
do
    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_oe.yml \
    configs/networks/resnet50_ensemble.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_gradcam.yml \
    --trainer.amp_dtype fp16 \
    --seed "$SEED" \
    --optimizer.num_epochs 90 \
    --num_workers 4 \
    --num_gpus 8 \
    --exp_name "$SEED" \
    --dataset.train.batch_size 32 \
    --dataset.val.batch_size 32 \
    --dataset.test.batch_size 32 \
    --dataset.oe.batch_size 32 \
    --output_dir "/results/$EXP_NAME" \
    --loss.oe 0.5 \
    --loss.gradcam 0.0 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-seed-$SEED" \
    --recorder.group "$EXP_NAME" || exit 1

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
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
    --postprocessor.postprocessor_args.aggregation 'logit-average' \
    --dataset.test.batch_size 32 \
    --dataset.val.batch_size 32 \
    --ood_dataset.batch_size 32 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval"

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
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
    --postprocessor.postprocessor_args.aggregation 'logit-average' \
    --dataset.test.batch_size 32 \
    --dataset.val.batch_size 32 \
    --ood_dataset.batch_size 32 \
    --recorder.project ded \
    --recorder.experiment "$EXP_NAME-eval-ece-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval"
done
