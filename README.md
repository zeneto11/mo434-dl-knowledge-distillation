# MO434 Project — Knowledge Distillation from Pretrained Image Classifiers into a Lightweight ConvNet

This project follows the MO434 assignment **“A Comparative Study of Knowledge Distillation Schemes from Pretrained Image Classifiers into a Lightweight ConvNet.”**  
The goal is to study how well a lightweight CNN student can imitate representations produced by pretrained image classifiers while reducing parameters and GFLOPs.

## Repository Usage

The repository now contains a runnable Python package under `src/`, four YAML configs under `configs/`, and command-line entry points for the three assignment phases.

**Environment Setup**

This project uses **Poetry (v2.2.1)** for dependency management.

```bash
# Install dependencies and create virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

Experiment execution commands are documented in:  
[COMMANDS.md](COMMANDS.md)

---

## Deliverable — Best Student per Teacher

[notebooks/best_students.ipynb](notebooks/best_students.ipynb) is the assignment deliverable: a **self-contained**
notebook that builds and trains the best student **from scratch** for a given
teacher.  
It depends only on PyTorch and torchvision, nothing is imported from
`src/` and no checkpoint or result file is required.

---

## 1. Fixed Experimental Choices

### Datasets

We will use two image classification datasets:

| Dataset                                                                                               | Motivation                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**FGVC-Aircraft**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FGVCAircraft) | Fine-grained classification with subtle visual differences between aircraft variants. Useful to test whether spatial feature-map distillation preserves discriminative structure. |
| [**Food-101**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101)           | Larger and more visually diverse dataset, with high variation in texture, background, composition, and class appearance. Useful to test robustness of the distillation strategy.  |

---

### Teachers

We will compare two pretrained teacher backbones:

| Teacher           | Role in the study                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| **ResNet50**      | Classic residual CNN teacher. Strong baseline and widely used reference architecture.                |
| **ConvNeXt-Tiny** | Modern CNN teacher with a different architectural bias and stronger ImageNet representation quality. |

Both teachers are initialized with ImageNet-pretrained weights.

---

### Student

We will study a family of lightweight CNN students designed from scratch.

The student architecture becomes an experimental variable.

| Student   | Encoder structure                             |
| --------- | --------------------------------------------- |
| Student-S | Conv32 → Conv64 → Conv128                     |
| Student-M | Conv32 → Conv64 → Conv128 → Conv256           |
| Student-L | Conv32 → Conv64 → Conv128 → Conv256 → Conv512 |

All students use the same predictor design for a given distillation target. Only the encoder capacity changes.

This design allows us to study the trade-off between classification performance, parameter count, and GFLOPs.

---

### Distillation Targets

We will compare two representation targets:

| Target                     | Description                                                                             | Predictor type          |
| -------------------------- | --------------------------------------------------------------------------------------- | ----------------------- |
| **Pre-GAP feature map**    | Student predicts the teacher's final spatial feature map before global average pooling. | Convolutional predictor |
| **Post-GAP pooled vector** | Student predicts the teacher's pooled representation after global average pooling.      | MLP predictor           |

---

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
Datasets              : 2
Teachers              : 2
Student Architectures : 3
Distillation Targets  : 2
Loss Functions        : 2

Total:
2 × 2 × 3 × 2 × 2 = 48 distilled student models
```

In addition, we will train:

```text
Teacher classifiers:
2 teacher architectures × 2 datasets
= 4 models

Student baselines:
3 student architectures × 2 datasets
= 6 models
```

The Student baselines are important because they answer whether distillation improves over simply training the lightweight model directly on the dataset labels.

```text
Total:
48 + 6 + 4 = 58 models
```

The experimental setup is:

```text
Datasets
---------
FGVC-Aircraft
Food-101

Teachers
---------
ResNet50
ConvNeXt-Tiny

Students
---------
Student-S
Student-M
Student-L

Predictors (targets)
---------
Convolutional predictor (Pre-GAP feature map)
MLP predictor (Post-GAP pooled vector)

Losses
---------
MSE
MSE + CrossEntropy
```

---

## 3. Project Phases

### Phase 1 — Train Teacher Classifiers

For each dataset and each teacher model, we start from an ImageNet-pretrained teacher and adapt it to the target dataset in two stages.

**Workflow**

1. Start with an ImageNet-pretrained teacher.
2. Freeze the teacher encoder / backbone.
3. Train the dataset-specific classifier head on the training split.
4. Validate after each epoch and save the best checkpoint by validation Top-1 accuracy.
5. Unfreeze the last encoder block.
6. Fine-tune the last block and classifier head with a smaller learning rate.
7. Reload the best validation checkpoint.
8. Run a final test pass on the test split.

**Trained teacher systems:**

| Teacher         | Dataset  | Checkpoint                                                  |
| --------------- | -------- | ----------------------------------------------------------- |
| `resnet50`      | Aircraft | `checkpoints/teachers/aircraft_resnet50_classifier.pt`      |
| `resnet50`      | Food101  | `checkpoints/teachers/food101_resnet50_classifier.pt`       |
| `convnext_tiny` | Aircraft | `checkpoints/teachers/aircraft_convnext_tiny_classifier.pt` |
| `convnext_tiny` | Food101  | `checkpoints/teachers/food101_convnext_tiny_classifier.pt`  |

The best teacher checkpoint from this phase is reused during distillation and evaluation.

---

### Phase 1b — Train Student Baselines

For each dataset and each student architecture, we also train a normal supervised student baseline.

**Workflow**

1. Build the selected student classifier.
2. Train it directly on the dataset labels using the training split.
3. Validate after each epoch.
4. Save the best checkpoint by validation Top-1 accuracy.
5. Reload the best checkpoint.
6. Run a final test pass on the test split.

These baselines are used to check whether distillation improves over direct supervised training.

Checkpoint pattern:

```text
checkpoints/students/{dataset}_{student}_baseline.pt
```

---

### Phase 2 — Train Distilled Students

For each combination of:

```text
dataset
teacher
student architecture
distillation target
loss function
```

we train a student encoder and predictor while keeping the trained teacher frozen.

**Workflow**

1. Load the best teacher checkpoint from Phase 1.
2. Freeze all teacher parameters.
3. Build the selected student distillation model.
4. For each training image, compute the teacher representation.
5. Train the student to match that representation.
6. If the loss is `mse_ce`, also classify the student representation through the frozen teacher classifier and apply cross-entropy with the true label.
7. Validate after each epoch.
8. Save the best student checkpoint:
   - `mse_ce`: best validation Top-1 accuracy.
   - `mse`: lowest validation MSE.
9. Save the training history and best checkpoint path.

#### Pre-GAP Feature-Map Distillation Flow

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
Student encoder
↓
Convolutional predictor
↓
Student feature map compatible with teacher feature map
↓
Frozen teacher GAP + frozen teacher classifier
↓
Class prediction
```

Loss options:

```text
MSE(student_feature_map, teacher_feature_map)

MSE(student_feature_map, teacher_feature_map)
+ CrossEntropy(prediction, label)
```

#### Post-GAP Pooled-Vector Distillation Flow

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
Student encoder
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

MSE(student_vector, teacher_vector)
+ CrossEntropy(prediction, label)
```

Example model names:

```text
aircraft_resnet50_student_m_pregap_mse
aircraft_resnet50_student_m_pregap_mse_ce
food101_convnext_tiny_student_l_postgap_mse
food101_convnext_tiny_student_l_postgap_mse_ce
```

---

### Phase 3 — Evaluation

The `src.evaluation.evaluate` script runs a test evaluation from saved checkpoints and computes deployment-oriented costs.

Evaluation covers:

```text
teacher classifiers
student CE baselines
distilled students
```

Distilled evaluation flow:

```text
Input image
↓
Selected student encoder
↓
Selected predictor
↓
Frozen classifier associated with the selected teacher and dataset
↓
Final prediction
```

Evaluation results are written to:

```text
results/evaluation/{filename}_eval.json
```

---

### Phase 4 — Relational Knowledge Distillation (Literature Extension)

To address core Question 5 (what can be reused from the literature), we add Relational Knowledge Distillation (Park et al., 2019) as a separate phase.

**How it differs from the core matrix**

| Aspect             | Feature distillation (Phases 2–3)          | Relational distillation (Phase 4)                 |
| ------------------ | ------------------------------------------ | ------------------------------------------------- |
| Student classifier | Frozen teacher head                        | Student's own trained linear classifier           |
| Transferred signal | Teacher pre-/post-GAP representation (MSE) | Pairwise distances + triplet angles of embeddings |
| Label supervision  | Optional CE through the teacher head       | Cross-entropy on the student's own logits         |
| Space alignment    | Predictor must match teacher channels/dim  | None — relations are computed inside each space   |

**Workflow**

1. Load the best frozen Phase 1 teacher.
2. Build a the best student architecture encoder with its own classifier.
3. For each batch, compute the teacher's pooled embeddings.
4. Train the student with cross-entropy on the labels plus the RKD
   distance-wise and angle-wise losses between student and teacher embeddings.
5. Save the best checkpoint by validation Top-1 accuracy.

```text
L_total = ce_weight      * CrossEntropy(student_logits, label)
        + distance_weight * RKD_distance(student_embed, teacher_embed)
        + angle_weight    * RKD_angle(student_embed, teacher_embed)
```

The relational method is compared head-to-head with the strongest feature-distillation configuration without changing the 48-model matrix.

Checkpoint pattern:

```text
checkpoints/students/{dataset}_{teacher}_{student}_rkd.pt
```

Metrics:

| Metric                   | Purpose                          |
| ------------------------ | -------------------------------- |
| **Top-1 accuracy**       | Main classification metric.      |
| **Top-5 accuracy**       | Secondary classification metric. |
| **Cross-entropy loss**   | Measures classification quality. |
| **Representation MSE**   | Measures distillation quality.   |
| **Number of parameters** | Measures model compactness.      |
| **GFLOPs**               | Measures computational cost.     |
| **Inference time**       | Practical efficiency metric.     |

---
