# Experiment Commands

All commands are run from the project root with the Poetry environment active (`poetry shell`).

---

## Quickstart — Launch Everything

```bash
# Preview all 58+ scheduled runs (no execution):
python -m src.run_experiments --dry-run

# Run everything, skip already-completed checkpoints:
python -m src.run_experiments --skip-existing

# Save a manifest of all planned runs:
python -m src.run_experiments --dry-run --manifest results/manifests/planned.json

# Filter by dataset / teacher / student / target / loss:
python -m src.run_experiments --datasets aircraft --students student_s --targets pregap
```

The launcher covers 4 teacher trains + 6 baselines + 48 distilled trains + 58 evals = **116 runs**.

---

## Phase 1 — Train Teacher Classifiers

Two-stage training: (1) classifier head only, (2) fine-tune last encoder block.
Both stages and epoch counts are controlled by the config (`teacher.stage1_epochs`, `stage2_epochs`, `stage2_lr`).

```bash
python -m src.training.train_teacher_classifier --config configs/aircraft_resnet50.yaml
python -m src.training.train_teacher_classifier --config configs/aircraft_convnext_tiny.yaml
python -m src.training.train_teacher_classifier --config configs/food101_resnet50.yaml
python -m src.training.train_teacher_classifier --config configs/food101_convnext_tiny.yaml
```

Checkpoints: `checkpoints/teachers/{dataset}_{teacher}_classifier.pt`
Training logs: `results/training_logs/{dataset}_{teacher}_teacher.log`

---

## Phase 1b — Train Student CE Baselines

One baseline per student architecture per dataset (no teacher needed).

```bash
# Aircraft
python -m src.training.train_student_baseline --config configs/aircraft_resnet50.yaml --student student_s
python -m src.training.train_student_baseline --config configs/aircraft_resnet50.yaml --student student_m
python -m src.training.train_student_baseline --config configs/aircraft_resnet50.yaml --student student_l

# Food-101
python -m src.training.train_student_baseline --config configs/food101_resnet50.yaml --student student_s
python -m src.training.train_student_baseline --config configs/food101_resnet50.yaml --student student_m
python -m src.training.train_student_baseline --config configs/food101_resnet50.yaml --student student_l
```

Checkpoints: `checkpoints/students/{dataset}_{student}_baseline.pt`
Training logs: `results/training_logs/{dataset}_{student}_baseline.log`

---

## Phase 2 — Train Distilled Students

`--student` ∈ {`student_s`, `student_m`, `student_l`}
`--target` ∈ {`pregap`, `postgap`}
`--loss` ∈ {`mse`, `mse_ce`}

Example for one config × student × target × loss:

```bash
python -m src.training.train_student_distillation \
    --config configs/aircraft_resnet50.yaml \
    --student student_m --target pregap --loss mse_ce
```

Full set for one config (4 commands per student):

```bash
for STUDENT in student_s student_m student_l; do
  for TARGET in pregap postgap; do
    for LOSS in mse mse_ce; do
      python -m src.training.train_student_distillation \
        --config configs/aircraft_resnet50.yaml \
        --student $STUDENT --target $TARGET --loss $LOSS
    done
  done
done
```

Repeat with: `configs/aircraft_convnext_tiny.yaml`, `configs/food101_resnet50.yaml`, `configs/food101_convnext_tiny.yaml`.

Checkpoints: `checkpoints/students/{dataset}_{teacher}_{student}_{target}_{loss}.pt`
Training logs: `results/training_logs/{dataset}_{teacher}_{student}_{target}_{loss}.log`

## Report Generation

```bash
python -m src.training.training_report \
    --train-dir results/training_logs \
    --output results/reports/training_report.md \
    --figures-dir results/figures/training_report
```

---

## Phase 3 — Evaluate

`evaluate.py` computes accuracy + params + GFLOPs + latency in one pass.

### Teacher

```bash
python -m src.evaluation.evaluate --config configs/aircraft_resnet50.yaml --kind teacher
python -m src.evaluation.evaluate --config configs/aircraft_convnext_tiny.yaml --kind teacher
python -m src.evaluation.evaluate --config configs/food101_resnet50.yaml --kind teacher
python -m src.evaluation.evaluate --config configs/food101_convnext_tiny.yaml --kind teacher
```

### Baseline

```bash
python -m src.evaluation.evaluate --config configs/aircraft_resnet50.yaml --kind baseline --student student_s
# ...repeat for student_m, student_l, and food101 config
```

### Distilled

```bash
python -m src.evaluation.evaluate \
    --config configs/aircraft_resnet50.yaml \
    --kind distilled --student student_m --target pregap --loss mse_ce
```

Add `--skip-costs` to skip GFLOPs/latency computation.

Eval results: `results/evaluation/{filename}_eval.json`

---

## Report Generation

```bash
python -m src.evaluation.report \
    --eval-dir results/evaluation \
    --output results/reports/report.md \
    --figures-dir results/figures/report
```

---

## Dataset Statistics

```bash
python -m src.data.dataset_stats --root data --output-dir results/dataset_stats
```

Outputs:

- `results/dataset_stats/aircraft.json`
- `results/dataset_stats/food101.json`
- `results/reports/dataset_stats.md`

---

## Results Folder Layout

```text
results/
├── training_logs/    ← epoch-level metrics + loguru logs per run
├── evaluation/       ← final test metrics with costs (JSON)
├── reports/          ← generated markdown reports
├── figures/          ← generated plots
├── manifests/        ← planned/completed run lists
└── dataset_stats/    ← dataset info JSONs
```

---

## Config Smoke-Test (fast subset)

Add limits to a config for quick iteration:

```yaml
dataset:
  train_limit: 200
  val_limit: 50
  test_limit: 50
```

Then run any command as normal. Remove the limits before full training.
