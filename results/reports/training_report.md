# Knowledge Distillation — Training Report

Generated from JSON logs in `results/training_logs`. This report inspects *training dynamics and checkpoint quality* — final test metrics, parameter counts, GFLOPs and latency live in the separate evaluation report.

**Checkpoint selection:** teachers, CE baselines and MSE+CE students are selected by highest validation Top-1; MSE-only students by lowest validation MSE (Top-1 can be a zero placeholder during training and is not meaningful for them).

## Experiment Configuration

Training hyperparameters loaded directly from `configs/*.yaml`.

| Parameter | aircraft_convnext_tiny | aircraft_resnet50 | food101_convnext_tiny | food101_resnet50 |
|-----------|---|---|---|---|
| image_size | 224 | 224 | 224 | 224 |
| batch_size | 32 | 32 | 32 | 32 |
| num_workers | 4 | 4 | 4 | 4 |
| stage1_epochs | 30 | 30 | 30 | 30 |
| stage2_epochs | 15 | 15 | 15 | 15 |
| stage2_lr | 5e-05 | 5e-05 | 5e-05 | 5e-05 |
| epochs | 30 | 30 | 30 | 30 |
| lr | 0.001 | 0.001 | 0.001 | 0.001 |
| weight_decay | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| scheduler | cosine | cosine | cosine | cosine |
| alpha | 1.0 | 1.0 | 1.0 | 1.0 |
| beta | 1.0 | 1.0 | 1.0 | 1.0 |
| amp | True | True | True | True |
| seed | 42 | 42 | 42 | 42 |

## Executive Summary

| Dataset  | Best teacher | T.T1  | Best baseline | B.T1  | Best distilled (T1) | D.T1  | Lowest MSE run    | MSE    | High-concern |
| -------- | ------------ | ----- | ------------- | ----- | ------------------- | ----- | ----------------- | ------ | ------------ |
| aircraft | CvN          | 58.84 | S-L base      | 35.61 | CvN/S-L/pre/MSE+CE  | 55.15 | RN50/S-L/post/MSE | 0.0858 | 0            |
| food101  | CvN          | 78.36 | S-L base      | 64.66 | CvN/S-L/pre/MSE+CE  | 62.93 | CvN/S-L/post/MSE  | 0.0331 | 1            |

## Teacher Training

Teacher adaptation is two-stage: the dataset head is trained with the encoder frozen, then the last encoder block is fine-tuned. The dashed line in the plots marks that transition. **`Best@` is the continuous epoch index**, so a stage-2 best shows its true position (e.g. 45), not the local stage-2 counter (e.g. 15).

### aircraft

| Teacher       | Ep | Best@ | Stg | Best T1 | Fin T1 | T1 drop | Best L | Fin L | Concern |
| ------------- | -- | ----- | --- | ------- | ------ | ------- | ------ | ----- | ------- |
| convnext_tiny | 45 | 43    | 2   | 58.84   | 58.72  | 0.12    | 1.44   | 1.45  | medium  |
| resnet50      | 45 | 45    | 2   | 53.11   | 53.11  | 0.00    | 1.85   | 1.85  | medium  |

![aircraft teacher training curves](../figures/training_report/aircraft_teacher_training_curves.png)

### food101

| Teacher       | Ep | Best@ | Stg | Best T1 | Fin T1 | T1 drop | Best L | Fin L | Concern |
| ------------- | -- | ----- | --- | ------- | ------ | ------- | ------ | ----- | ------- |
| convnext_tiny | 45 | 42    | 2   | 78.36   | 78.34  | 0.03    | 1.03   | 1.06  | high    |
| resnet50      | 45 | 44    | 2   | 74.51   | 74.48  | 0.03    | 1.26   | 1.25  | medium  |

![food101 teacher training curves](../figures/training_report/food101_teacher_training_curves.png)


## Student Baselines (CE)

Supervised CE students are the reference distillation must beat. `vs base` columns elsewhere are measured against these.

### aircraft

| Student   | Ep | Best@ | Best T1 | Fin T1 | T1 drop | Best L | Fin L | Concern |
| --------- | -- | ----- | ------- | ------ | ------- | ------ | ----- | ------- |
| student_l | 30 | 30    | 35.61   | 35.61  | 0.00    | 2.45   | 2.45  | low     |
| student_m | 30 | 30    | 15.69   | 15.69  | 0.00    | 3.40   | 3.40  | low     |
| student_s | 30 | 25    | 6.78    | 6.36   | 0.42    | 4.18   | 4.18  | low     |

![aircraft baseline training curves](../figures/training_report/aircraft_baseline_training_curves.png)

### food101

| Student   | Ep | Best@ | Best T1 | Fin T1 | T1 drop | Best L | Fin L | Concern |
| --------- | -- | ----- | ------- | ------ | ------- | ------ | ----- | ------- |
| student_l | 30 | 29    | 64.66   | 64.62  | 0.04    | 1.42   | 1.42  | low     |
| student_m | 30 | 30    | 53.87   | 53.87  | 0.00    | 1.89   | 1.89  | low     |
| student_s | 30 | 28    | 35.58   | 35.42  | 0.16    | 2.74   | 2.73  | low     |

![food101 baseline training curves](../figures/training_report/food101_baseline_training_curves.png)


## Distillation — aircraft

- Best classification run: `convnext_tiny/student_l/pregap/mse_ce` — 55.146% val Top-1.
- Lowest-MSE run: `resnet50/student_l/postgap/mse` — 0.0858 val MSE.

### All distilled runs

| Teacher       | Student   | Target  | Loss   | Ep | Best@ | Best T1 | Fin T1 | T1 drop | Best MSE | Fin MSE | MSE reg% | T1 gap | vs base | Concern |
| ------------- | --------- | ------- | ------ | -- | ----- | ------- | ------ | ------- | -------- | ------- | -------- | ------ | ------- | ------- |
| convnext_tiny | student_l | postgap | mse    | 30 | 28    | 0.00    | 0.00   | 0.00    | 0.2220   | 0.2221  | 0.1      | 0.00   |         | low     |
| convnext_tiny | student_l | postgap | mse_ce | 30 | 28    | 26.04   | 24.96  | 1.08    | 0.3879   | 0.4247  | 9.6      | 2.75   | -9.57   | low     |
| convnext_tiny | student_l | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 1.0053   | 1.0053  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_l | pregap  | mse_ce | 30 | 29    | 55.15   | 54.85  | 0.30    | 1.4622   | 1.4642  | 0.1      | 24.01  | 19.53   | medium  |
| convnext_tiny | student_m | postgap | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.2439   | 0.2439  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_m | postgap | mse_ce | 30 | 29    | 24.75   | 24.42  | 0.33    | 0.3889   | 0.3951  | 2.4      | 2.69   | 9.06    | low     |
| convnext_tiny | student_m | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 1.2041   | 1.2041  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_m | pregap  | mse_ce | 30 | 29    | 40.65   | 40.17  | 0.48    | 1.5860   | 1.5937  | 0.6      | 12.59  | 24.96   | low     |
| convnext_tiny | student_s | postgap | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.2634   | 0.2634  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_s | postgap | mse_ce | 30 | 29    | 14.94   | 14.79  | 0.15    | 0.4453   | 0.4530  | 4.2      | 2.49   | 8.16    | low     |
| convnext_tiny | student_s | pregap  | mse    | 30 | 27    | 0.00    | 0.00   | 0.00    | 1.3816   | 1.3821  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_s | pregap  | mse_ce | 30 | 29    | 23.55   | 23.46  | 0.09    | 1.6637   | 1.6655  | 0.2      | 4.82   | 16.77   | low     |
| resnet50      | student_l | postgap | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.0858   | 0.0858  | 0.0      | 0.00   |         | low     |
| resnet50      | student_l | postgap | mse_ce | 30 | 28    | 6.96    | 6.96   | 0.00    | 0.1635   | 0.1665  | 2.3      | 0.63   | -28.65  | low     |
| resnet50      | student_l | pregap  | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 1.0475   | 1.0475  | 0.0      | 0.00   |         | low     |
| resnet50      | student_l | pregap  | mse_ce | 30 | 29    | 51.82   | 51.67  | 0.15    | 1.5133   | 1.5218  | 3.4      | 25.00  | 16.20   | medium  |
| resnet50      | student_m | postgap | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.0880   | 0.0881  | 0.1      | 0.00   |         | low     |
| resnet50      | student_m | postgap | mse_ce | 30 | 30    | 13.11   | 13.11  | 0.00    | 0.1632   | 0.1632  | 0.7      | 0.30   | -2.58   | low     |
| resnet50      | student_m | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 1.1344   | 1.1344  | 0.0      | 0.00   |         | low     |
| resnet50      | student_m | pregap  | mse_ce | 30 | 29    | 39.24   | 39.03  | 0.21    | 1.5202   | 1.5197  | 1.5      | 15.08  | 23.55   | medium  |
| resnet50      | student_s | postgap | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.0887   | 0.0887  | 0.0      | 0.00   |         | low     |
| resnet50      | student_s | postgap | mse_ce | 30 | 28    | 15.45   | 15.06  | 0.39    | 0.1604   | 0.1613  | 7.5      | 1.92   | 8.67    | low     |
| resnet50      | student_s | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 1.2330   | 1.2330  | 0.0      | 0.00   |         | low     |
| resnet50      | student_s | pregap  | mse_ce | 30 | 30    | 27.03   | 27.03  | 0.00    | 1.5075   | 1.5075  | 0.7      | 6.29   | 20.25   | low     |

### Critical warnings

- **convnext_tiny/student_l/pregap/mse_ce** (`medium`): train-val Top-1 gap 24.0 pp (overfitting)
- **resnet50/student_l/pregap/mse_ce** (`medium`): train-val Top-1 gap 25.0 pp (overfitting)
- **resnet50/student_m/pregap/mse_ce** (`medium`): train-val Top-1 gap 15.1 pp (overfitting)

### Training curves

Top-1 (4 panels, teacher × target). Grey lines are teacher and CE-baseline references; coloured lines are distilled students. MSE-only runs with placeholder zero Top-1 are omitted here and shown in the MSE plot.

![aircraft distilled val Top-1 by teacher](../figures/training_report/aircraft_val_top1_distilled_by_teacher.png)

Validation MSE for all distilled runs (lower = better matching).

![aircraft distilled val MSE by teacher](../figures/training_report/aircraft_val_mse_distilled_by_teacher.png)


## Distillation — food101

- Best classification run: `convnext_tiny/student_l/pregap/mse_ce` — 62.931% val Top-1.
- Lowest-MSE run: `convnext_tiny/student_l/postgap/mse` — 0.0331 val MSE.

### All distilled runs

| Teacher       | Student   | Target  | Loss   | Ep | Best@ | Best T1 | Fin T1 | T1 drop | Best MSE | Fin MSE | MSE reg% | T1 gap | vs base | Concern |
| ------------- | --------- | ------- | ------ | -- | ----- | ------- | ------ | ------- | -------- | ------- | -------- | ------ | ------- | ------- |
| convnext_tiny | student_l | postgap | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.0331   | 0.0331  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_l | postgap | mse_ce | 30 | 29    | 40.58   | 40.45  | 0.13    | 0.0477   | 0.0476  | 0.0      | -5.51  | -24.08  | low     |
| convnext_tiny | student_l | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.2695   | 0.2695  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_l | pregap  | mse_ce | 30 | 29    | 62.93   | 62.46  | 0.48    | 0.3422   | 0.3403  | 5.4      | 8.18   | -1.73   | low     |
| convnext_tiny | student_m | postgap | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.0353   | 0.0353  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_m | postgap | mse_ce | 30 | 27    | 38.01   | 37.80  | 0.21    | 0.0483   | 0.0479  | 0.3      | -4.56  | -15.87  | low     |
| convnext_tiny | student_m | pregap  | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.2843   | 0.2843  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_m | pregap  | mse_ce | 30 | 30    | 58.53   | 58.53  | 0.00    | 0.3392   | 0.3392  | 4.4      | 2.02   | 4.66    | low     |
| convnext_tiny | student_s | postgap | mse    | 30 | 26    | 0.00    | 0.00   | 0.00    | 0.0381   | 0.0383  | 0.5      | 0.00   |         | low     |
| convnext_tiny | student_s | postgap | mse_ce | 30 | 29    | 35.62   | 35.60  | 0.01    | 0.0478   | 0.0478  | 0.1      | -5.65  | 0.04    | low     |
| convnext_tiny | student_s | pregap  | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.2954   | 0.2954  | 0.0      | 0.00   |         | low     |
| convnext_tiny | student_s | pregap  | mse_ce | 30 | 29    | 49.39   | 49.39  | 0.00    | 0.3284   | 0.3283  | 1.4      | -2.58  | 13.81   | low     |
| resnet50      | student_l | postgap | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.0701   | 0.0701  | 0.0      | 0.00   |         | low     |
| resnet50      | student_l | postgap | mse_ce | 30 | 30    | 28.87   | 28.87  | 0.00    | 0.1163   | 0.1163  | 0.1      | -3.84  | -35.79  | low     |
| resnet50      | student_l | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.8522   | 0.8522  | 0.0      | 0.00   |         | low     |
| resnet50      | student_l | pregap  | mse_ce | 30 | 30    | 62.39   | 62.39  | 0.00    | 1.0492   | 1.0492  | 0.0      | 5.18   | -2.27   | low     |
| resnet50      | student_m | postgap | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.0755   | 0.0769  | 1.8      | 0.00   |         | low     |
| resnet50      | student_m | postgap | mse_ce | 30 | 29    | 32.40   | 32.15  | 0.25    | 0.1165   | 0.1163  | 0.2      | -4.39  | -21.48  | low     |
| resnet50      | student_m | pregap  | mse    | 30 | 30    | 0.00    | 0.00   | 0.00    | 0.9245   | 0.9245  | 0.0      | 0.00   |         | low     |
| resnet50      | student_m | pregap  | mse_ce | 30 | 30    | 57.90   | 57.90  | 0.00    | 1.0586   | 1.0586  | 0.1      | 1.20   | 4.03    | low     |
| resnet50      | student_s | postgap | mse    | 30 | 28    | 0.00    | 0.00   | 0.00    | 0.0810   | 0.0814  | 0.4      | 0.00   |         | low     |
| resnet50      | student_s | postgap | mse_ce | 30 | 30    | 32.38   | 32.38  | 0.00    | 0.1164   | 0.1164  | 0.4      | -5.00  | -3.19   | low     |
| resnet50      | student_s | pregap  | mse    | 30 | 29    | 0.00    | 0.00   | 0.00    | 0.9735   | 0.9735  | 0.0      | 0.00   |         | low     |
| resnet50      | student_s | pregap  | mse_ce | 30 | 29    | 47.66   | 47.64  | 0.01    | 1.0561   | 1.0565  | 0.0      | -2.69  | 12.08   | low     |

### Critical warnings

No critical training problems were detected.

### Training curves

Top-1 (4 panels, teacher × target). Grey lines are teacher and CE-baseline references; coloured lines are distilled students. MSE-only runs with placeholder zero Top-1 are omitted here and shown in the MSE plot.

![food101 distilled val Top-1 by teacher](../figures/training_report/food101_val_top1_distilled_by_teacher.png)

Validation MSE for all distilled runs (lower = better matching).

![food101 distilled val MSE by teacher](../figures/training_report/food101_val_mse_distilled_by_teacher.png)


## Global Factor Summary

Averaged across both datasets. Top-1 columns use classification-bearing runs (CE / MSE+CE); MSE columns use all distilled runs with MSE logs.

| Factor  | Value         | Mean T1 | Best T1 | Mean MSE | Best MSE | N  |
| ------- | ------------- | ------- | ------- | -------- | -------- | -- |
| Teacher | ConvNeXt-Tiny | 39.18   | 62.93   | 0.5151   | 0.0331   | 24 |
| Teacher | ResNet50      | 34.60   | 62.39   | 0.6331   | 0.0701   | 24 |
| Target  | post-GAP      | 25.76   | 40.58   | 0.1470   | 0.0331   | 24 |
| Target  | pre-GAP       | 48.02   | 62.93   | 1.0013   | 0.2695   | 24 |
| Student | Student-L     | 41.84   | 62.93   | 0.5417   | 0.0331   | 16 |
| Student | Student-M     | 38.08   | 58.53   | 0.5757   | 0.0353   | 16 |
| Student | Student-S     | 30.75   | 49.39   | 0.6050   | 0.0381   | 16 |
| Loss    | MSE           |         |         | 0.4971   | 0.0331   | 24 |
| Loss    | MSE+CE        | 36.89   | 62.93   | 0.6512   | 0.0477   | 24 |
