# MO434 Project — Knowledge Distillation from Pretrained Image Classifiers into a Lightweight ConvNet

This project follows the MO434 assignment **“A Comparative Study of Knowledge Distillation Schemes from Pretrained Image Classifiers into a Lightweight ConvNet.”** The goal is to study how well a lightweight CNN student can imitate representations produced by pretrained image classifiers while reducing parameters and GFLOPs.

The experimental setup is based on representation-based knowledge distillation. Instead of only distilling the teacher logits, the student is trained to predict either the teacher's **pre-GAP feature map** or the teacher's **post-GAP pooled representation**. An additional cross-entropy term can be used to keep the student representation aligned with the dataset labels.

---

## 1. Fixed Experimental Choices

### Datasets

We will use two image classification datasets:

| Dataset                                                                                               | Motivation                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**FGVC-Aircraft**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FGVCAircraft) | Fine-grained classification with subtle visual differences between aircraft variants. Useful to test whether spatial feature-map distillation preserves discriminative structure. |
| [**Food-101**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101)           | Larger and more visually diverse dataset, with high variation in texture, background, composition, and class appearance. Useful to test robustness of the distillation strategy.  |

These datasets were chosen instead of common benchmarks such as CIFAR-10, CIFAR-100, or Tiny ImageNet because they provide more interesting domain characteristics and should produce a more meaningful comparison between distillation strategies.

### Teachers

We will compare two pretrained teacher backbones:

| Teacher           | Role in the study                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| **ResNet50**      | Classic residual CNN teacher. Strong baseline and widely used reference architecture.                |
| **ConvNeXt-Tiny** | Modern CNN teacher with a different architectural bias and stronger ImageNet representation quality. |

Both teachers are initialized with ImageNet-pretrained weights.

### Student

The main student architecture is:

```text
MobileNetV3-Large encoder + predictor
```

MobileNetV3-Large is used because it is significantly lighter than the teachers, but still expressive enough for FGVC-Aircraft and Food-101.

### Distillation Targets

We will compare two representation targets:

| Target                     | Description                                                                             | Predictor type          |
| -------------------------- | --------------------------------------------------------------------------------------- | ----------------------- |
| **Pre-GAP feature map**    | Student predicts the teacher's final spatial feature map before global average pooling. | Convolutional predictor |
| **Post-GAP pooled vector** | Student predicts the teacher's pooled representation after global average pooling.      | MLP predictor           |

### Loss Functions

For each configuration, we will train two student variants:

| Loss                   | Meaning                                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **MSE**                | The student only imitates the teacher representation.                                                                                 |
| **MSE + CrossEntropy** | The student imitates the teacher representation and is also supervised by the true class label through the frozen teacher classifier. |

The combined loss is:

```text
L_total = alpha * MSE(student_representation, teacher_representation)
        + beta  * CrossEntropy(classifier(student_representation), label)
```

---

## 2. Overall Experimental Matrix

The main experiments are:

```text
2 datasets
× 2 teachers
× 2 distillation targets
× 2 losses
= 16 distilled student models
```

In addition, we will train:

```text
4 teacher classifier models
2 MobileNetV3-Large baselines trained only with CrossEntropy
```

The MobileNet baselines are important because they answer whether distillation improves over simply training the lightweight model directly on the dataset labels.

---

## 3. Phase 1 — Train Teacher Classifiers

For each dataset and each teacher:

```text
ImageNet-pretrained teacher
↓
Freeze teacher encoder / backbone
↓
Replace the original classification head
↓
Train a new dataset-specific classifier
```

The four trained teacher systems are:

```text
resnet50_encoder_frozen + resnet50_classifier_aircraft
resnet50_encoder_frozen + resnet50_classifier_food101

convnext_tiny_encoder_frozen + convnext_tiny_classifier_aircraft
convnext_tiny_encoder_frozen + convnext_tiny_classifier_food101
```

During Phase 1:

```text
Train:
- dataset-specific classifier head

Freeze:
- teacher encoder / backbone
```

The trained classifier from this phase will later be reused when training and evaluating the student.

---

## 4. Phase 2 — Train the Student by Representation Distillation

For each combination of dataset, teacher, target, and loss, we train:

```text
MobileNetV3-Large encoder + predictor
```

The teacher encoder and the teacher classifier are kept frozen.

### 4.1 Pre-GAP Feature-Map Distillation

Teacher path:

```text
Input image
↓
Frozen teacher encoder
↓
Teacher pre-GAP feature map
```

Student path:

```text
Input image
↓
MobileNetV3-Large encoder
↓
Convolutional predictor
↓
Student feature map compatible with teacher feature map
↓
Teacher GAP + frozen teacher classifier
↓
Class prediction
```

Loss options:

```text
MSE(student_feature_map, teacher_feature_map)
```

or:

```text
MSE(student_feature_map, teacher_feature_map)
+ CrossEntropy(prediction, label)
```

Example model names:

```text
mobilenet_pregap_resnet50_aircraft_mse
mobilenet_pregap_resnet50_aircraft_mse_ce
mobilenet_pregap_convnext_food101_mse
mobilenet_pregap_convnext_food101_mse_ce
```

### 4.2 Post-GAP Pooled-Vector Distillation

Teacher path:

```text
Input image
↓
Frozen teacher encoder
↓
Global Average Pooling
↓
Teacher pooled vector
```

Student path:

```text
Input image
↓
MobileNetV3-Large encoder
↓
Global Average Pooling
↓
MLP predictor
↓
Student vector compatible with teacher pooled vector
↓
Frozen teacher classifier
↓
Class prediction
```

Loss options:

```text
MSE(student_vector, teacher_vector)
```

or:

```text
MSE(student_vector, teacher_vector)
+ CrossEntropy(prediction, label)
```

Example model names:

```text
mobilenet_posgap_resnet50_aircraft_mse
mobilenet_posgap_resnet50_aircraft_mse_ce
mobilenet_posgap_convnext_food101_mse
mobilenet_posgap_convnext_food101_mse_ce
```

---

## 5. Phase 3 — Evaluation

Each trained student is evaluated on the test split of the corresponding dataset.

Evaluation flow:

```text
Input image
↓
MobileNetV3-Large encoder
↓
Selected predictor
↓
Frozen classifier associated with the selected teacher and dataset
↓
Final prediction
```

We will compare:

```text
1. ResNet50 teacher vs ConvNeXt-Tiny teacher
2. Pre-GAP feature-map distillation vs post-GAP pooled-vector distillation
3. MSE vs MSE + CrossEntropy
4. Distilled MobileNetV3-Large vs MobileNetV3-Large baseline
5. Student performance vs teacher performance
6. Accuracy, parameters, GFLOPs, and inference cost
```

Recommended metrics:

| Metric                   | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| **Top-1 accuracy**       | Main classification metric.                         |
| **Top-5 accuracy**       | Useful especially for Food-101.                     |
| **Cross-entropy loss**   | Measures classification quality.                    |
| **Representation MSE**   | Measures how well the student imitates the teacher. |
| **Number of parameters** | Measures model compactness.                         |
| **GFLOPs**               | Measures computational cost.                        |
| **Inference time**       | Optional practical efficiency metric.               |

---

## 6. Suggested Project Organization

```text
project/
├── README.md
├── configs/
│   ├── aircraft_resnet50.yaml
│   ├── aircraft_convnext_tiny.yaml
│   ├── food101_resnet50.yaml
│   └── food101_convnext_tiny.yaml
├── data/
│   └── README.md
├── src/
│   ├── datasets/
│   │   ├── aircraft.py
│   │   └── food101.py
│   ├── models/
│   │   ├── teachers.py
│   │   ├── students.py
│   │   └── predictors.py
│   ├── training/
│   │   ├── train_teacher_classifier.py
│   │   ├── train_student_distillation.py
│   │   └── train_student_baseline.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── compute_costs.py
│   └── utils/
│       ├── losses.py
│       ├── metrics.py
│       └── checkpoints.py
├── notebooks/
│   ├── analysis_aircraft.ipynb
│   ├── analysis_food101.ipynb
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
├── checkpoints/
│   ├── teachers/
│   └── students/
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
└── report/
    ├── figures/
    └── final_report.pdf
```

---

## 7. What the Final Report Should Discuss

The final report should answer the core questions from the MO434 project statement:

### 1. Which teacher transfers best?

Compare ResNet50 and ConvNeXt-Tiny as teachers for both datasets.

### 2. What should the student predict?

Compare:

```text
pre-GAP feature map
vs
post-GAP pooled vector
```

The pre-GAP target may preserve spatial information better, while the post-GAP target may be easier and cheaper to learn.

### 3. What is the best student encoder + predictor design?

The main student is MobileNetV3-Large. The predictor changes according to the distillation target:

```text
pre-GAP  → convolutional predictor
post-GAP → MLP predictor
```

The report should compare their accuracy, representation loss, parameter count, and GFLOPs.

### 4. Which loss function works better?

Compare:

```text
MSE
vs
MSE + CrossEntropy
```

The expected hypothesis is that MSE + CrossEntropy should produce better classification results because it combines representation matching with label supervision.

### 5. What can be learned from the knowledge distillation literature?

The project should connect the implemented method to the papers cited in the assignment.

---

## 8. Related Work Mentioned in the Project PDF

The project statement cites the following works as conceptual background:

1. **VGG — Simonyan and Zisserman, 2015**  
   Introduced very deep convolutional networks and serves as a classical CNN reference.

2. **ResNet — He et al., 2016**  
   Introduced residual learning, making very deep CNNs easier to train. ResNet50 is used here as one of the teachers.

3. **ConvNeXt — Liu et al., 2022**  
   Modernized CNN design for the 2020s. ConvNeXt-Tiny is used here as the second teacher.

4. **Knowledge Distillation — Hinton, Vinyals, and Dean, 2015**  
   Proposed transferring knowledge from a teacher to a student using softened logits. This project differs by focusing mainly on representation distillation rather than only logit distillation.

5. **FitNets — Romero et al., 2015**  
   Introduced training thin students using intermediate hints from the teacher. The pre-GAP feature-map target in this project is closest to the FitNets idea.

6. **Attention Transfer — Zagoruyko and Komodakis, 2017**  
   Proposed transferring spatial attention maps. This is relevant to the pre-GAP setting, where spatial structure is preserved.

7. **Relational Knowledge Distillation — Park et al., 2019**  
   Proposed transferring relations among samples rather than only matching individual outputs. This can be discussed as a possible extension if time allows.

---

## 9. Expected Main Comparisons

The main result tables should include one row per trained student:

```text
dataset | teacher | target | loss | top1 | top5 | mse | params | gflops
```

Recommended tables:

1. Teacher performance on each dataset.
2. MobileNetV3-Large baseline performance.
3. Distilled student results for FGVC-Aircraft.
4. Distilled student results for Food-101.
5. Best student per teacher.
6. Best student per dataset.
7. Accuracy vs GFLOPs trade-off.

Recommended plots:

```text
- Top-1 accuracy by teacher and target
- Accuracy vs GFLOPs
- MSE vs classification accuracy
- Confusion matrix for the best model on each dataset
```

---

## 10. Final Experimental Summary

The approved experimental setup is:

```text
Datasets:
- FGVC-Aircraft
- Food-101

Teachers:
- ResNet50
- ConvNeXt-Tiny

Student:
- MobileNetV3-Large

Predictors:
- Convolutional predictor for pre-GAP feature-map distillation
- MLP predictor for post-GAP pooled-vector distillation

Losses:
- MSE
- MSE + CrossEntropy

Baselines:
- Frozen-teacher classifier performance
- MobileNetV3-Large trained directly with CrossEntropy
```

This setup is compact enough to be feasible, but still broad enough to answer the main research questions of the MO434 project.

---

## 11. Implemented Project Commands

The repository now contains a runnable Python package under `src/`, four YAML configs under `configs/`, and command-line entry points for the three assignment phases.

### Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

If installing PyTorch on a specific CUDA setup, follow the official PyTorch install selector and then run `.venv/bin/pip install -e .`.

### Phase 1 — Train frozen-encoder teachers

```bash
.venv/bin/python -m src.training.train_teacher_classifier --config configs/aircraft_resnet50.yaml
.venv/bin/python -m src.training.train_teacher_classifier --config configs/aircraft_convnext_tiny.yaml
.venv/bin/python -m src.training.train_teacher_classifier --config configs/food101_resnet50.yaml
.venv/bin/python -m src.training.train_teacher_classifier --config configs/food101_convnext_tiny.yaml
```

### Baseline — Train MobileNetV3-Large directly with CE

```bash
.venv/bin/python -m src.training.train_student_baseline --config configs/aircraft_resnet50.yaml
.venv/bin/python -m src.training.train_student_baseline --config configs/food101_resnet50.yaml
```

The baseline does not depend on the teacher, so one config per dataset is enough.

### Phase 2 — Train distilled students

Run these four commands for each dataset-teacher config:

```bash
.venv/bin/python -m src.training.train_student_distillation --config configs/aircraft_resnet50.yaml --target pregap --loss mse
.venv/bin/python -m src.training.train_student_distillation --config configs/aircraft_resnet50.yaml --target pregap --loss mse_ce
.venv/bin/python -m src.training.train_student_distillation --config configs/aircraft_resnet50.yaml --target postgap --loss mse
.venv/bin/python -m src.training.train_student_distillation --config configs/aircraft_resnet50.yaml --target postgap --loss mse_ce
```

Repeat with:

```text
configs/aircraft_convnext_tiny.yaml
configs/food101_resnet50.yaml
configs/food101_convnext_tiny.yaml
```

### Phase 3 — Evaluate and compute costs

```bash
.venv/bin/python -m src.evaluation.evaluate --config configs/aircraft_resnet50.yaml --kind teacher
.venv/bin/python -m src.evaluation.evaluate --config configs/aircraft_resnet50.yaml --kind baseline
.venv/bin/python -m src.evaluation.evaluate --config configs/aircraft_resnet50.yaml --kind distilled --target pregap --loss mse_ce

.venv/bin/python -m src.evaluation.compute_costs --config configs/aircraft_resnet50.yaml --kind teacher
.venv/bin/python -m src.evaluation.compute_costs --config configs/aircraft_resnet50.yaml --kind baseline
.venv/bin/python -m src.evaluation.compute_costs --config configs/aircraft_resnet50.yaml --kind distilled --target pregap
```

Checkpoints are written under `checkpoints/`, and JSON logs are written under `results/logs/`. The configs support `train_limit`, `val_limit`, and `test_limit` for small smoke runs before full training.
