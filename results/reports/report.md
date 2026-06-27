# Knowledge Distillation Evaluation Report

This report is generated from JSON files in `results/evaluation`.

Raw accuracies are analyzed separately per dataset. The global section only uses normalized comparisons such as improvement over the dataset baseline, improvement over the matching teacher, and cost ratios.

**Plot conventions:**
   - colour = student architecture (blue S-S, orange S-M, green S-L);
   - overlay = distillation target (pre-GAP, post-GAP);
   - edge = loss (black edge MSE+CE, no edge MSE-only);
   - shape:  
       ★ = teacher;  
       ◆ = CE baseline;  
       ● = ResNet-50;  
       ■ = ConvNeXt-Tiny; 

## Executive Summary By Dataset

- `aircraft`: best student is `convnext_tiny` + `student_l` + `pregap` + `mse_ce` at 55.566% top-1 (19.922 pp vs matching baseline; 77.486% GFLOPs saved vs teacher).
- `food101`: best student is `convnext_tiny` + `student_l` + `pregap` + `mse_ce` at 68.376% top-1 (-1.857 pp vs matching baseline; 77.486% GFLOPs saved vs teacher).
## Dataset: aircraft

- Best student CE baseline: `student_l` with 35.644% top-1.
- Best teacher classifier: `convnext_tiny` with 58.686% top-1.
- Best distilled student: `convnext_tiny` + `student_l` + `pregap` + `mse_ce` with 55.566% top-1 (19.922 pp vs matching baseline, -3.120 pp vs its teacher).


![aircraft distillation gain over CE baseline](../figures/report/aircraft_gain_matrix.png)

### Evaluation Results

| model_type | teacher       | student   | target  | loss_name | top1   | top5   | loss  | mse   | top1_minus_baseline | top1_minus_teacher | cost_params | cost_gflops | params_saved_vs_teacher_pct | gflops_saved_vs_teacher_pct |
| ---------- | ------------- | --------- | ------- | --------- | ------ | ------ | ----- | ----- | ------------------- | ------------------ | ----------- | ----------- | --------------------------- | --------------------------- |
| baseline   |               | student_l |         | ce        | 35.644 | 69.337 | 2.422 |       | 0.000               |                    | 1620868     | 1.961       |                             |                             |
| baseline   |               | student_m |         | ce        | 16.472 | 44.644 | 3.376 |       | 0.000               |                    | 414596      | 1.498       |                             |                             |
| baseline   |               | student_s |         | ce        | 7.231  | 23.252 | 4.168 |       | 0.000               |                    | 106372      | 1.034       |                             |                             |
| distilled  | convnext_tiny | student_l | postgap | mse       | 15.602 | 49.565 | 3.529 | 0.222 | -20.042             | -43.084            | 2632580     | 1.963       | 90.563                      | 78.005                      |
| distilled  | convnext_tiny | student_l | postgap | mse_ce    | 26.553 | 61.386 | 2.757 | 0.386 | -9.091              | -32.133            | 2632580     | 1.963       | 90.563                      | 78.005                      |
| distilled  | convnext_tiny | student_l | pregap  | mse       | 24.842 | 62.106 | 2.759 | 0.998 | -10.801             | -33.843            | 2139524     | 2.010       | 92.330                      | 77.486                      |
| distilled  | convnext_tiny | student_l | pregap  | mse_ce    | 55.566 | 86.019 | 1.544 | 1.458 | 19.922              | -3.120             | 2139524     | 2.010       | 92.330                      | 77.486                      |
| distilled  | convnext_tiny | student_m | postgap | mse       | 11.221 | 36.004 | 4.122 | 0.244 | -5.251              | -47.465            | 1255300     | 1.500       | 95.500                      | 83.198                      |
| distilled  | convnext_tiny | student_m | postgap | mse_ce    | 24.692 | 58.536 | 2.917 | 0.390 | 8.221               | -33.993            | 1255300     | 1.500       | 95.500                      | 83.198                      |
| distilled  | convnext_tiny | student_m | pregap  | mse       | 8.851  | 29.043 | 4.155 | 1.197 | -7.621              | -49.835            | 860548      | 1.653       | 96.915                      | 81.480                      |
| distilled  | convnext_tiny | student_m | pregap  | mse_ce    | 40.564 | 76.748 | 2.128 | 1.581 | 24.092              | -18.122            | 860548      | 1.653       | 96.915                      | 81.480                      |
| distilled  | convnext_tiny | student_s | postgap | mse       | 7.681  | 26.973 | 4.716 | 0.263 | 0.450               | -51.005            | 861572      | 1.036       | 96.911                      | 88.397                      |
| distilled  | convnext_tiny | student_s | postgap | mse_ce    | 15.812 | 41.704 | 3.462 | 0.448 | 8.581               | -42.874            | 861572      | 1.036       | 96.911                      | 88.397                      |
| distilled  | convnext_tiny | student_s | pregap  | mse       | 3.600  | 13.081 | 5.476 | 1.376 | -3.630              | -55.086            | 515972      | 1.577       | 98.150                      | 82.331                      |
| distilled  | convnext_tiny | student_s | pregap  | mse_ce    | 24.152 | 57.576 | 2.937 | 1.658 | 16.922              | -34.533            | 515972      | 1.577       | 98.150                      | 82.331                      |
| distilled  | resnet50      | student_l | postgap | mse       | 13.381 | 43.474 | 3.450 | 0.086 | -22.262             | -40.354            | 4401028     | 1.967       | 81.440                      | 76.197                      |
| distilled  | resnet50      | student_l | postgap | mse_ce    | 6.961  | 21.452 | 4.187 | 0.165 | -28.683             | -46.775            | 4401028     | 1.967       | 81.440                      | 76.197                      |
| distilled  | resnet50      | student_l | pregap  | mse       | 39.004 | 75.638 | 2.170 | 1.045 | 3.360               | -14.731            | 4400004     | 2.219       | 81.445                      | 73.146                      |
| distilled  | resnet50      | student_l | pregap  | mse_ce    | 51.875 | 83.198 | 1.760 | 1.519 | 16.232              | -1.860             | 4400004     | 2.219       | 81.445                      | 73.146                      |
| distilled  | resnet50      | student_m | postgap | mse       | 9.661  | 34.233 | 3.907 | 0.088 | -6.811              | -44.074            | 2958212     | 1.503       | 87.525                      | 81.809                      |
| distilled  | resnet50      | student_m | postgap | mse_ce    | 14.281 | 40.384 | 3.534 | 0.164 | -2.190              | -39.454            | 2958212     | 1.503       | 87.525                      | 81.809                      |
| distilled  | resnet50      | student_m | pregap  | mse       | 26.493 | 62.166 | 2.765 | 1.133 | 10.021              | -27.243            | 2957188     | 2.426       | 87.529                      | 70.646                      |
| distilled  | resnet50      | student_m | pregap  | mse_ce    | 40.654 | 75.128 | 2.194 | 1.524 | 24.182              | -13.081            | 2957188     | 2.426       | 87.529                      | 70.646                      |
| distilled  | resnet50      | student_s | postgap | mse       | 8.881  | 31.593 | 4.015 | 0.089 | 1.650               | -44.854            | 2531716     | 1.039       | 89.323                      | 87.426                      |
| distilled  | resnet50      | student_s | postgap | mse_ce    | 15.482 | 42.634 | 3.477 | 0.161 | 8.251               | -38.254            | 2531716     | 1.039       | 89.323                      | 87.426                      |
| distilled  | resnet50      | student_s | pregap  | mse       | 10.291 | 34.053 | 3.778 | 1.233 | 3.060               | -43.444            | 2530692     | 4.538       | 89.328                      | 45.087                      |
| distilled  | resnet50      | student_s | pregap  | mse_ce    | 25.893 | 58.986 | 2.917 | 1.509 | 18.662              | -27.843            | 2530692     | 4.538       | 89.328                      | 45.087                      |
| teacher    | convnext_tiny |           |         |           | 58.686 | 87.999 | 1.431 |       |                     | 0.000              | 27895492    | 8.927       | 0.000                       | 0.000                       |
| teacher    | resnet50      |           |         |           | 53.735 | 82.658 | 1.846 |       |                     | 0.000              | 23712932    | 8.264       | 0.000                       | 0.000                       |

### Question 1: Which Teacher Transfers Best?

**Answer:** `resnet50` transfers best on this dataset by mean distilled Top-1 (21.905%).

| teacher       | mean   | max    | min   | count |
| ------------- | ------ | ------ | ----- | ----- |
| resnet50      | 21.905 | 51.875 | 6.961 | 12    |
| convnext_tiny | 21.595 | 55.566 | 3.600 | 12    |

Best configuration found for each teacher:

| teacher       | best_student | best_target | best_loss | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| ------------- | ------------ | ----------- | --------- | --------- | --------- | ------------------- | ------------------ |
| convnext_tiny | student_l    | pregap      | mse_ce    | 55.566    | 86.019    | 19.922              | -3.120             |
| resnet50      | student_l    | pregap      | mse_ce    | 51.875    | 83.198    | 16.232              | -1.860             |

### Question 2: What Should The Student Predict?

**Answer:** `pregap` performs best on this dataset by mean Top-1; this corresponds to pre-GAP feature map with convolutional predictor.

| target  | mean   | max    | min   | count |
| ------- | ------ | ------ | ----- | ----- |
| pregap  | 29.315 | 55.566 | 3.600 | 12    |
| postgap | 14.184 | 26.553 | 6.961 | 12    |

Best configuration found for each target:

| target  | best_teacher  | best_student | best_loss | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| ------- | ------------- | ------------ | --------- | --------- | --------- | ------------------- | ------------------ |
| pregap  | convnext_tiny | student_l    | mse_ce    | 55.566    | 86.019    | 19.922              | -3.120             |
| postgap | convnext_tiny | student_l    | mse_ce    | 26.553    | 61.386    | -9.091              | -32.133            |

![aircraft pre-GAP vs post-GAP](../figures/report/aircraft_pregap_vs_postgap.png)

### Question 3: What Is The Best Student Architecture?

**Answer:** `student_l` (Student-L (Conv32→Conv64→Conv128→Conv256→Conv512)) achieves the best mean distilled Top-1 (29.223%) on this dataset.

| student   | mean   | max    | min   | count |
| --------- | ------ | ------ | ----- | ----- |
| student_l | 29.223 | 55.566 | 6.961 | 8     |
| student_m | 22.052 | 40.654 | 8.851 | 8     |
| student_s | 13.974 | 25.893 | 3.600 | 8     |

Best configuration found for each student architecture:

| student   | best_teacher  | best_target | best_loss | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| --------- | ------------- | ----------- | --------- | --------- | --------- | ------------------- | ------------------ |
| student_l | convnext_tiny | pregap      | mse_ce    | 55.566    | 86.019    | 19.922              | -3.120             |
| student_m | resnet50      | pregap      | mse_ce    | 40.654    | 75.128    | 24.182              | -13.081            |
| student_s | resnet50      | pregap      | mse_ce    | 25.893    | 58.986    | 18.662              | -27.843            |

Architecture ranking with cost savings relative to the matching teacher:

| teacher       | student   | target  | loss_name | top1   | top5   | cost_params | cost_gflops | params_saved_vs_teacher_pct | gflops_saved_vs_teacher_pct | top1_per_gflop |
| ------------- | --------- | ------- | --------- | ------ | ------ | ----------- | ----------- | --------------------------- | --------------------------- | -------------- |
| convnext_tiny | student_l | pregap  | mse_ce    | 55.566 | 86.019 | 2139524     | 2.010       | 92.330                      | 77.486                      | 27.647         |
| resnet50      | student_l | pregap  | mse_ce    | 51.875 | 83.198 | 4400004     | 2.219       | 81.445                      | 73.146                      | 23.376         |
| resnet50      | student_m | pregap  | mse_ce    | 40.654 | 75.128 | 2957188     | 2.426       | 87.529                      | 70.646                      | 16.760         |
| convnext_tiny | student_m | pregap  | mse_ce    | 40.564 | 76.748 | 860548      | 1.653       | 96.915                      | 81.480                      | 24.536         |
| resnet50      | student_l | pregap  | mse       | 39.004 | 75.638 | 4400004     | 2.219       | 81.445                      | 73.146                      | 17.576         |
| convnext_tiny | student_l | postgap | mse_ce    | 26.553 | 61.386 | 2632580     | 1.963       | 90.563                      | 78.005                      | 13.523         |
| resnet50      | student_m | pregap  | mse       | 26.493 | 62.166 | 2957188     | 2.426       | 87.529                      | 70.646                      | 10.922         |
| resnet50      | student_s | pregap  | mse_ce    | 25.893 | 58.986 | 2530692     | 4.538       | 89.328                      | 45.087                      | 5.706          |
| convnext_tiny | student_l | pregap  | mse       | 24.842 | 62.106 | 2139524     | 2.010       | 92.330                      | 77.486                      | 12.360         |
| convnext_tiny | student_m | postgap | mse_ce    | 24.692 | 58.536 | 1255300     | 1.500       | 95.500                      | 83.198                      | 16.463         |
| convnext_tiny | student_s | pregap  | mse_ce    | 24.152 | 57.576 | 515972      | 1.577       | 98.150                      | 82.331                      | 15.313         |
| convnext_tiny | student_s | postgap | mse_ce    | 15.812 | 41.704 | 861572      | 1.036       | 96.911                      | 88.397                      | 15.266         |
| convnext_tiny | student_l | postgap | mse       | 15.602 | 49.565 | 2632580     | 1.963       | 90.563                      | 78.005                      | 7.946          |
| resnet50      | student_s | postgap | mse_ce    | 15.482 | 42.634 | 2531716     | 1.039       | 89.323                      | 87.426                      | 14.899         |
| resnet50      | student_m | postgap | mse_ce    | 14.281 | 40.384 | 2958212     | 1.503       | 87.525                      | 81.809                      | 9.500          |
| resnet50      | student_l | postgap | mse       | 13.381 | 43.474 | 4401028     | 1.967       | 81.440                      | 76.197                      | 6.803          |
| convnext_tiny | student_m | postgap | mse       | 11.221 | 36.004 | 1255300     | 1.500       | 95.500                      | 83.198                      | 7.481          |
| resnet50      | student_s | pregap  | mse       | 10.291 | 34.053 | 2530692     | 4.538       | 89.328                      | 45.087                      | 2.268          |
| resnet50      | student_m | postgap | mse       | 9.661  | 34.233 | 2958212     | 1.503       | 87.525                      | 81.809                      | 6.427          |
| resnet50      | student_s | postgap | mse       | 8.881  | 31.593 | 2531716     | 1.039       | 89.323                      | 87.426                      | 8.547          |
| convnext_tiny | student_m | pregap  | mse       | 8.851  | 29.043 | 860548      | 1.653       | 96.915                      | 81.480                      | 5.354          |
| convnext_tiny | student_s | postgap | mse       | 7.681  | 26.973 | 861572      | 1.036       | 96.911                      | 88.397                      | 7.416          |
| resnet50      | student_l | postgap | mse_ce    | 6.961  | 21.452 | 4401028     | 1.967       | 81.440                      | 76.197                      | 3.539          |
| convnext_tiny | student_s | pregap  | mse       | 3.600  | 13.081 | 515972      | 1.577       | 98.150                      | 82.331                      | 2.283          |

![aircraft accuracy vs compute (GFLOPs)](../figures/report/aircraft_cost_vs_top1.png)


![aircraft accuracy vs compute (Params)](../figures/report/aircraft_param_vs_top1.png)

### Question 4: What Loss Function Should We Use?

**Answer:** `mse_ce` performs best on this dataset by mean distilled Top-1 (28.540%).

| loss_name | mean   | max    | min   | count |
| --------- | ------ | ------ | ----- | ----- |
| mse_ce    | 28.540 | 55.566 | 6.961 | 12    |
| mse       | 14.959 | 39.004 | 3.600 | 12    |

Best configuration found for each loss:

| loss_name | best_teacher  | best_student | best_target | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| --------- | ------------- | ------------ | ----------- | --------- | --------- | ------------------- | ------------------ |
| mse_ce    | convnext_tiny | student_l    | pregap      | 55.566    | 86.019    | 19.922              | -3.120             |
| mse       | resnet50      | student_l    | pregap      | 39.004    | 75.638    | 3.360               | -14.731            |

### Question 5: How Does Relational Knowledge Distillation Compare?

This question evaluates Relational Knowledge Distillation (Park et al., 2019) as a literature-based extension. Unlike feature distillation, RKD keeps the student's own classifier and transfers the *relations* between examples, namely normalized pairwise distances and triplet angles computed over the pooled embeddings, rather than forcing the student to match the teacher's feature values directly.

Because these relations are computed within each model's embedding space, the teacher and student embedding dimensions do not need to match. RKD is trained on the same `convnext_tiny` + `student_l` pair as the strongest feature-distillation configuration, allowing a direct comparison between the two distillation paradigms.

**Answer:** RKD reaches 27.063% Top-1, which trails the best feature-distillation result for the same pair by 28.503 pp and is 8.581 pp below the CE baseline.

| method                    | top1   | top5   | top1_minus_baseline | top1_minus_teacher | cost_params | cost_gflops | gflops_saved_vs_teacher_pct |
| ------------------------- | ------ | ------ | ------------------- | ------------------ | ----------- | ----------- | --------------------------- |
| CE baseline               | 35.644 | 69.337 | 0.000               |                    | 1620868     | 1.961       |                             |
| Best feature distillation | 55.566 | 86.019 | 19.922              | -3.120             | 2139524     | 2.010       | 77.486                      |
| RKD (relational)          | 27.063 | 62.226 | -8.581              | -31.623            | 1620868     | 1.961       | 78.028                      |
| Teacher                   | 58.686 | 87.999 |                     | 0.000              | 27895492    | 8.927       | 0.000                       |


![aircraft RKD comparison](../figures/report/aircraft_rkd_comparison.png)


## Dataset: food101

- Best student CE baseline: `student_l` with 70.234% top-1.
- Best teacher classifier: `convnext_tiny` with 82.923% top-1.
- Best distilled student: `convnext_tiny` + `student_l` + `pregap` + `mse_ce` with 68.376% top-1 (-1.857 pp vs matching baseline, -14.547 pp vs its teacher).


![food101 distillation gain over CE baseline](../figures/report/food101_gain_matrix.png)

### Evaluation Results

| model_type | teacher       | student   | target  | loss_name | top1   | top5   | loss  | mse   | top1_minus_baseline | top1_minus_teacher | cost_params | cost_gflops | params_saved_vs_teacher_pct | gflops_saved_vs_teacher_pct |
| ---------- | ------------- | --------- | ------- | --------- | ------ | ------ | ----- | ----- | ------------------- | ------------------ | ----------- | ----------- | --------------------------- | --------------------------- |
| baseline   |               | student_l |         | ce        | 70.234 | 90.388 | 1.123 |       | 0.000               |                    | 1621381     | 1.961       |                             |                             |
| baseline   |               | student_m |         | ce        | 58.717 | 83.295 | 1.635 |       | 0.000               |                    | 414853      | 1.498       |                             |                             |
| baseline   |               | student_s |         | ce        | 39.687 | 68.107 | 2.500 |       | 0.000               |                    | 106501      | 1.034       |                             |                             |
| distilled  | convnext_tiny | student_l | postgap | mse       | 56.808 | 84.250 | 1.721 | 0.033 | -13.426             | -26.115            | 2633349     | 1.963       | 90.560                      | 78.005                      |
| distilled  | convnext_tiny | student_l | postgap | mse_ce    | 44.170 | 72.463 | 2.227 | 0.048 | -26.063             | -38.752            | 2633349     | 1.963       | 90.560                      | 78.005                      |
| distilled  | convnext_tiny | student_l | pregap  | mse       | 65.248 | 89.077 | 1.302 | 0.267 | -4.986              | -17.675            | 2140293     | 2.010       | 92.328                      | 77.486                      |
| distilled  | convnext_tiny | student_l | pregap  | mse_ce    | 68.376 | 89.370 | 1.235 | 0.342 | -1.857              | -14.547            | 2140293     | 2.010       | 92.328                      | 77.486                      |
| distilled  | convnext_tiny | student_m | postgap | mse       | 48.024 | 77.386 | 2.117 | 0.035 | -10.693             | -34.899            | 1256069     | 1.500       | 95.497                      | 83.198                      |
| distilled  | convnext_tiny | student_m | postgap | mse_ce    | 41.386 | 69.980 | 2.383 | 0.048 | -17.331             | -41.537            | 1256069     | 1.500       | 95.497                      | 83.198                      |
| distilled  | convnext_tiny | student_m | pregap  | mse       | 52.622 | 80.816 | 1.841 | 0.282 | -6.095              | -30.301            | 861317      | 1.653       | 96.912                      | 81.480                      |
| distilled  | convnext_tiny | student_m | pregap  | mse_ce    | 64.087 | 86.990 | 1.383 | 0.339 | 5.370               | -18.836            | 861317      | 1.653       | 96.912                      | 81.480                      |
| distilled  | convnext_tiny | student_s | postgap | mse       | 34.436 | 64.432 | 2.836 | 0.038 | -5.251              | -48.487            | 862341      | 1.036       | 96.909                      | 88.397                      |
| distilled  | convnext_tiny | student_s | postgap | mse_ce    | 39.703 | 68.186 | 2.467 | 0.048 | 0.016               | -43.220            | 862341      | 1.036       | 96.909                      | 88.397                      |
| distilled  | convnext_tiny | student_s | pregap  | mse       | 36.408 | 67.141 | 2.575 | 0.294 | -3.279              | -46.515            | 516741      | 1.577       | 98.148                      | 82.331                      |
| distilled  | convnext_tiny | student_s | pregap  | mse_ce    | 54.016 | 80.372 | 1.809 | 0.327 | 14.329              | -28.907            | 516741      | 1.577       | 98.148                      | 82.331                      |
| distilled  | resnet50      | student_l | postgap | mse       | 59.604 | 85.390 | 1.593 | 0.069 | -10.630             | -19.794            | 4403077     | 1.967       | 81.433                      | 76.197                      |
| distilled  | resnet50      | student_l | postgap | mse_ce    | 32.412 | 61.687 | 2.771 | 0.114 | -37.822             | -46.986            | 4403077     | 1.967       | 81.433                      | 76.197                      |
| distilled  | resnet50      | student_l | pregap  | mse       | 68.246 | 90.016 | 1.217 | 0.843 | -1.988              | -11.152            | 4402053     | 2.219       | 81.438                      | 73.146                      |
| distilled  | resnet50      | student_l | pregap  | mse_ce    | 67.350 | 88.935 | 1.248 | 1.041 | -2.883              | -12.048            | 4402053     | 2.219       | 81.438                      | 73.146                      |
| distilled  | resnet50      | student_m | postgap | mse       | 49.026 | 77.770 | 2.056 | 0.074 | -9.691              | -30.372            | 2960261     | 1.503       | 87.517                      | 81.809                      |
| distilled  | resnet50      | student_m | postgap | mse_ce    | 35.727 | 64.824 | 2.625 | 0.114 | -22.990             | -43.671            | 2960261     | 1.503       | 87.517                      | 81.809                      |
| distilled  | resnet50      | student_m | pregap  | mse       | 53.984 | 80.962 | 1.817 | 0.916 | -4.733              | -25.414            | 2959237     | 2.426       | 87.522                      | 70.646                      |
| distilled  | resnet50      | student_m | pregap  | mse_ce    | 63.089 | 86.285 | 1.437 | 1.050 | 4.372               | -16.309            | 2959237     | 2.426       | 87.522                      | 70.646                      |
| distilled  | resnet50      | student_s | postgap | mse       | 35.358 | 64.685 | 2.815 | 0.079 | -4.329              | -44.040            | 2533765     | 1.039       | 89.316                      | 87.426                      |
| distilled  | resnet50      | student_s | postgap | mse_ce    | 35.489 | 64.685 | 2.786 | 0.114 | -4.198              | -43.909            | 2533765     | 1.039       | 89.316                      | 87.426                      |
| distilled  | resnet50      | student_s | pregap  | mse       | 36.388 | 65.200 | 2.662 | 0.965 | -3.299              | -43.010            | 2532741     | 4.538       | 89.320                      | 45.087                      |
| distilled  | resnet50      | student_s | pregap  | mse_ce    | 52.154 | 78.749 | 1.900 | 1.048 | 12.467              | -27.244            | 2532741     | 4.538       | 89.320                      | 45.087                      |
| teacher    | convnext_tiny |           |         |           | 82.923 | 96.281 | 0.718 |       |                     | 0.000              | 27896261    | 8.927       | 0.000                       | 0.000                       |
| teacher    | resnet50      |           |         |           | 79.398 | 94.638 | 0.894 |       |                     | 0.000              | 23714981    | 8.264       | 0.000                       | 0.000                       |

### Question 1: Which Teacher Transfers Best?

**Answer:** `convnext_tiny` transfers best on this dataset by mean distilled Top-1 (50.440%).

| teacher       | mean   | max    | min    | count |
| ------------- | ------ | ------ | ------ | ----- |
| convnext_tiny | 50.440 | 68.376 | 34.436 | 12    |
| resnet50      | 49.069 | 68.246 | 32.412 | 12    |

Best configuration found for each teacher:

| teacher       | best_student | best_target | best_loss | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| ------------- | ------------ | ----------- | --------- | --------- | --------- | ------------------- | ------------------ |
| convnext_tiny | student_l    | pregap      | mse_ce    | 68.376    | 89.370    | -1.857              | -14.547            |
| resnet50      | student_l    | pregap      | mse       | 68.246    | 90.016    | -1.988              | -11.152            |

### Question 2: What Should The Student Predict?

**Answer:** `pregap` performs best on this dataset by mean Top-1; this corresponds to pre-GAP feature map with convolutional predictor.

| target  | mean   | max    | min    | count |
| ------- | ------ | ------ | ------ | ----- |
| pregap  | 56.831 | 68.376 | 36.388 | 12    |
| postgap | 42.679 | 59.604 | 32.412 | 12    |

Best configuration found for each target:

| target  | best_teacher  | best_student | best_loss | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| ------- | ------------- | ------------ | --------- | --------- | --------- | ------------------- | ------------------ |
| pregap  | convnext_tiny | student_l    | mse_ce    | 68.376    | 89.370    | -1.857              | -14.547            |
| postgap | resnet50      | student_l    | mse       | 59.604    | 85.390    | -10.630             | -19.794            |

![food101 pre-GAP vs post-GAP](../figures/report/food101_pregap_vs_postgap.png)

### Question 3: What Is The Best Student Architecture?

**Answer:** `student_l` (Student-L (Conv32→Conv64→Conv128→Conv256→Conv512)) achieves the best mean distilled Top-1 (57.777%) on this dataset.

| student   | mean   | max    | min    | count |
| --------- | ------ | ------ | ------ | ----- |
| student_l | 57.777 | 68.376 | 32.412 | 8     |
| student_m | 50.993 | 64.087 | 35.727 | 8     |
| student_s | 40.494 | 54.016 | 34.436 | 8     |

Best configuration found for each student architecture:

| student   | best_teacher  | best_target | best_loss | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| --------- | ------------- | ----------- | --------- | --------- | --------- | ------------------- | ------------------ |
| student_l | convnext_tiny | pregap      | mse_ce    | 68.376    | 89.370    | -1.857              | -14.547            |
| student_m | convnext_tiny | pregap      | mse_ce    | 64.087    | 86.990    | 5.370               | -18.836            |
| student_s | convnext_tiny | pregap      | mse_ce    | 54.016    | 80.372    | 14.329              | -28.907            |

Architecture ranking with cost savings relative to the matching teacher:

| teacher       | student   | target  | loss_name | top1   | top5   | cost_params | cost_gflops | params_saved_vs_teacher_pct | gflops_saved_vs_teacher_pct | top1_per_gflop |
| ------------- | --------- | ------- | --------- | ------ | ------ | ----------- | ----------- | --------------------------- | --------------------------- | -------------- |
| convnext_tiny | student_l | pregap  | mse_ce    | 68.376 | 89.370 | 2140293     | 2.010       | 92.328                      | 77.486                      | 34.021         |
| resnet50      | student_l | pregap  | mse       | 68.246 | 90.016 | 4402053     | 2.219       | 81.438                      | 73.146                      | 30.752         |
| resnet50      | student_l | pregap  | mse_ce    | 67.350 | 88.935 | 4402053     | 2.219       | 81.438                      | 73.146                      | 30.349         |
| convnext_tiny | student_l | pregap  | mse       | 65.248 | 89.077 | 2140293     | 2.010       | 92.328                      | 77.486                      | 32.464         |
| convnext_tiny | student_m | pregap  | mse_ce    | 64.087 | 86.990 | 861317      | 1.653       | 96.912                      | 81.480                      | 38.764         |
| resnet50      | student_m | pregap  | mse_ce    | 63.089 | 86.285 | 2959237     | 2.426       | 87.522                      | 70.646                      | 26.008         |
| resnet50      | student_l | postgap | mse       | 59.604 | 85.390 | 4403077     | 1.967       | 81.433                      | 76.197                      | 30.302         |
| convnext_tiny | student_l | postgap | mse       | 56.808 | 84.250 | 2633349     | 1.963       | 90.560                      | 78.005                      | 28.933         |
| convnext_tiny | student_s | pregap  | mse_ce    | 54.016 | 80.372 | 516741      | 1.577       | 98.148                      | 82.331                      | 34.246         |
| resnet50      | student_m | pregap  | mse       | 53.984 | 80.962 | 2959237     | 2.426       | 87.522                      | 70.646                      | 22.255         |
| convnext_tiny | student_m | pregap  | mse       | 52.622 | 80.816 | 861317      | 1.653       | 96.912                      | 81.480                      | 31.829         |
| resnet50      | student_s | pregap  | mse_ce    | 52.154 | 78.749 | 2532741     | 4.538       | 89.320                      | 45.087                      | 11.493         |
| resnet50      | student_m | postgap | mse       | 49.026 | 77.770 | 2960261     | 1.503       | 87.517                      | 81.809                      | 32.612         |
| convnext_tiny | student_m | postgap | mse       | 48.024 | 77.386 | 1256069     | 1.500       | 95.497                      | 83.198                      | 32.018         |
| convnext_tiny | student_l | postgap | mse_ce    | 44.170 | 72.463 | 2633349     | 1.963       | 90.560                      | 78.005                      | 22.496         |
| convnext_tiny | student_m | postgap | mse_ce    | 41.386 | 69.980 | 1256069     | 1.500       | 95.497                      | 83.198                      | 27.593         |
| convnext_tiny | student_s | postgap | mse_ce    | 39.703 | 68.186 | 862341      | 1.036       | 96.909                      | 88.397                      | 38.332         |
| convnext_tiny | student_s | pregap  | mse       | 36.408 | 67.141 | 516741      | 1.577       | 98.148                      | 82.331                      | 23.083         |
| resnet50      | student_s | pregap  | mse       | 36.388 | 65.200 | 2532741     | 4.538       | 89.320                      | 45.087                      | 8.019          |
| resnet50      | student_m | postgap | mse_ce    | 35.727 | 64.824 | 2960261     | 1.503       | 87.517                      | 81.809                      | 23.766         |
| resnet50      | student_s | postgap | mse_ce    | 35.489 | 64.685 | 2533765     | 1.039       | 89.316                      | 87.426                      | 34.153         |
| resnet50      | student_s | postgap | mse       | 35.358 | 64.685 | 2533765     | 1.039       | 89.316                      | 87.426                      | 34.028         |
| convnext_tiny | student_s | postgap | mse       | 34.436 | 64.432 | 862341      | 1.036       | 96.909                      | 88.397                      | 33.247         |
| resnet50      | student_l | postgap | mse_ce    | 32.412 | 61.687 | 4403077     | 1.967       | 81.433                      | 76.197                      | 16.478         |

![food101 accuracy vs compute (GFLOPs)](../figures/report/food101_cost_vs_top1.png)


![food101 accuracy vs compute (Params)](../figures/report/food101_param_vs_top1.png)

### Question 4: What Loss Function Should We Use?

**Answer:** `mse_ce` performs best on this dataset by mean distilled Top-1 (49.830%).

| loss_name | mean   | max    | min    | count |
| --------- | ------ | ------ | ------ | ----- |
| mse_ce    | 49.830 | 68.376 | 32.412 | 12    |
| mse       | 49.679 | 68.246 | 34.436 | 12    |

Best configuration found for each loss:

| loss_name | best_teacher  | best_student | best_target | best_top1 | best_top5 | best_minus_baseline | best_minus_teacher |
| --------- | ------------- | ------------ | ----------- | --------- | --------- | ------------------- | ------------------ |
| mse_ce    | convnext_tiny | student_l    | pregap      | 68.376    | 89.370    | -1.857              | -14.547            |
| mse       | resnet50      | student_l    | pregap      | 68.246    | 90.016    | -1.988              | -11.152            |

### Question 5: How Does Relational Knowledge Distillation Compare?

This question evaluates Relational Knowledge Distillation (Park et al., 2019) as a literature-based extension. Unlike feature distillation, RKD keeps the student's own classifier and transfers the *relations* between examples, namely normalized pairwise distances and triplet angles computed over the pooled embeddings, rather than forcing the student to match the teacher's feature values directly.

Because these relations are computed within each model's embedding space, the teacher and student embedding dimensions do not need to match. RKD is trained on the same `convnext_tiny` + `student_l` pair as the strongest feature-distillation configuration, allowing a direct comparison between the two distillation paradigms.

**Answer:** RKD reaches 70.016% Top-1, which outperforms the best feature-distillation result for the same pair by 1.640 pp and is 0.218 pp below the CE baseline.

| method                    | top1   | top5   | top1_minus_baseline | top1_minus_teacher | cost_params | cost_gflops | gflops_saved_vs_teacher_pct |
| ------------------------- | ------ | ------ | ------------------- | ------------------ | ----------- | ----------- | --------------------------- |
| CE baseline               | 70.234 | 90.388 | 0.000               |                    | 1621381     | 1.961       |                             |
| Best feature distillation | 68.376 | 89.370 | -1.857              | -14.547            | 2140293     | 2.010       | 77.486                      |
| RKD (relational)          | 70.016 | 89.929 | -0.218              | -12.907            | 1621381     | 1.961       | 78.028                      |
| Teacher                   | 82.923 | 96.281 |                     | 0.000              | 27896261    | 8.927       | 0.000                       |


![food101 RKD comparison](../figures/report/food101_rkd_comparison.png)


## Global Overview With Normalized Comparisons

Best distilled student per dataset:

| dataset  | teacher       | student   | target | loss_name | top1   | top1_minus_baseline | top1_minus_teacher | cost_params | cost_gflops | params_saved_vs_teacher_pct | gflops_saved_vs_teacher_pct |
| -------- | ------------- | --------- | ------ | --------- | ------ | ------------------- | ------------------ | ----------- | ----------- | --------------------------- | --------------------------- |
| aircraft | convnext_tiny | student_l | pregap | mse_ce    | 55.566 | 19.922              | -3.120             | 2139524     | 2.010       | 92.330                      | 77.486                      |
| food101  | convnext_tiny | student_l | pregap | mse_ce    | 68.376 | -1.857              | -14.547            | 2140293     | 2.010       | 92.328                      | 77.486                      |

Global teacher overview using normalized deltas:

| teacher       | top1_minus_baseline | top1_minus_teacher | gflops_saved_vs_teacher_pct | params_saved_vs_teacher_pct |
| ------------- | ------------------- | ------------------ | --------------------------- | --------------------------- |
| convnext_tiny | -1.980              | -34.787            | 81.816                      | 95.060                      |
| resnet50      | -2.510              | -31.080            | 72.385                      | 86.095                      |

Global target overview using normalized deltas:

| target  | top1_minus_baseline | top1_minus_teacher | gflops_saved_vs_teacher_pct | params_saved_vs_teacher_pct |
| ------- | ------------------- | ------------------ | --------------------------- | --------------------------- |
| pregap  | 5.076               | -25.612            | 71.696                      | 90.947                      |
| postgap | -9.566              | -40.254            | 82.505                      | 90.208                      |

Global student architecture overview using normalized deltas:

| student   | top1_minus_baseline | top1_minus_teacher | gflops_saved_vs_teacher_pct | params_saved_vs_teacher_pct |
| --------- | ------------------- | ------------------ | --------------------------- | --------------------------- |
| student_s | 3.775               | -41.452            | 75.810                      | 93.426                      |
| student_m | -1.072              | -32.163            | 79.283                      | 91.865                      |
| student_l | -9.439              | -25.186            | 76.208                      | 86.442                      |

Global loss overview using normalized deltas:

| loss_name | top1_minus_baseline | top1_minus_teacher | gflops_saved_vs_teacher_pct | params_saved_vs_teacher_pct |
| --------- | ------------------- | ------------------ | --------------------------- | --------------------------- |
| mse_ce    | 1.188               | -29.500            | 77.101                      | 90.578                      |
| mse       | -5.678              | -36.366            | 77.101                      | 90.578                      |


![Global improvement over baseline](../figures/report/global_improvement_over_baseline.png)


![Global accuracy vs compute](../figures/report/global_cost_vs_top1.png)


![Global accuracy vs compute (top-5)](../figures/report/global_cost_vs_top5.png)
