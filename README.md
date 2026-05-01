# CARLA Behavioral Cloning Pipeline

Modular imitation-learning pipeline that maps a front camera image to
`(steer, throttle, brake)` using a CARLA expert dataset.

```
project/
├── configs/default.yaml     # all hyperparams / paths (YAML)
├── data/dataset.py          # auto-detecting loader + train/val split by episode
├── models/model.py          # PilotNet + ResNet-ish CNN + bounded output heads
├── training/train.py        # loop: AMP, TB logging, checkpointing, resume, early-stop
├── evaluation/eval.py       # offline eval: MAE/RMSE, steer dir acc, plots
├── utils/                   # config, logging, preprocess, augment, balanced sampler
├── main.py                  # CLI entry point (train / eval / inspect-data)
└── inference.py             # BCController for CARLA integration
```

## Dataset layout (auto-detected)

The loader walks `data.root` and picks up any directory shaped like:

```
<root>/.../<episode>/labels/*.parquet     # or *.csv / *.json
<root>/.../<episode>/images/<camera>/*.jpg
<root>/.../<episode>/metadata.json        # optional
```

It handles the CARLA dump used here:

```
run_0001/worker_<id>/episodes/ep_<hash>/{labels,images,metadata.json}
```

Required label columns: `steer`, `throttle`, `brake`, plus one image-path column
(`img_<camera>` or anything starting with `img_` / `image`). Optional columns
(`speed_kmh`, `reward_total`, `bucket`, `frame_idx`) are picked up automatically.

## Install

```bash
pip install -r requirements.txt
```

The pipeline uses PyTorch + PIL + pandas only — no `torchvision` or
`albumentations` needed.

## Train

```bash
python main.py train --config configs/default.yaml
```

With overrides:

```bash
python main.py train --config configs/default.yaml \
  experiment.name=exp_v2 \
  training.epochs=60 \
  data.loader.batch_size=128 \
  model.arch=deepcnn
```

Resume:

```bash
python main.py train --config configs/default.yaml \
  --resume runs/exp_v2/checkpoints/last.pt
```

TensorBoard:

```bash
tensorboard --logdir runs/
```

## Evaluate

```bash
python main.py eval --config configs/default.yaml \
  --checkpoint runs/exp_v2/checkpoints/best.pt --split val
```

Outputs land in `runs/<exp>/eval_val/`:
`metrics.json`, `predictions.npy`, `targets.npy`, `histograms.png`,
`scatter.png`, `timeseries.png`.

## Inspect the dataset

```bash
python main.py inspect-data --config configs/default.yaml
```

## Use in CARLA

```python
from inference import BCController

ctrl = BCController("runs/exp_v2/checkpoints/best.pt")
steer, throttle, brake = ctrl.act(rgb_uint8_image)
vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
```

## Default dataset path

In `configs/default.yaml`, `data.root` is set to `../../carla_data/dataset`,
which resolves to `/home/T6795AR/Documents/Training/carla_data/dataset` when
you run from `repo/Imitation_learning/project/`. Point it at an absolute
path or different location via CLI:

```bash
python main.py train data.root=/path/to/your/dataset
```

---

## Notes on improving performance

**Data coverage**
- `inspect-data` first. If steering is heavily peaked at 0 (it is in this
  dataset), keep `data.sampler.enabled=true`.
- Try `data.filter.min_speed_kmh: 0.5` to drop frozen-at-red-light frames
  that dilute the signal.
- Mix buckets: `data.filter.buckets: ["expert", "perturbed", "recovery"]`.
  `recovery` is particularly valuable for robustness but hurts if it
  dominates the training set.

**Input**
- The default is 200×88 after cropping 80 px of sky and 20 px of hood.
  If you raise resolution, switch to `model.arch=deepcnn`.
- `normalize=imagenet` mean-subtracts natural images; `minmax` ([-1, 1])
  is a simpler, equally valid baseline.

**Augmentation**
- Start with brightness + small rotation + tiny Gaussian noise.
- `horizontal_flip_prob` flips steering sign — leave it off unless you
  have verified your label signs are symmetric.

**Loss / heads**
- `smoothl1` is kinder to noisy expert labels than pure MSE.
- `training.loss.steer_weight` up-weights the head that actually matters
  for driving. Keeping throttle/brake at 0.5 usually helps stability.

**Regularization**
- Default dropout is 0.2. Push to 0.4 if you see val loss diverging.
- Add `weight_decay=1e-4` if you see overfitting late in training.

**Training dynamics**
- Cosine schedule + 2-epoch warmup works well for BC.
- With AMP on, the default `batch_size=64` fits comfortably on consumer
  GPUs; push to 128–256 on an A100 / H100.

**Next steps (beyond this baseline)**
- Condition on ego speed (concatenate normalized `speed_kmh` with the
  flattened CNN features before the head).
- Use a sequence model (GRU / small transformer) over recent frames to
  handle hesitation and braking intent.
- Predict waypoints rather than raw controls, then follow them with a PID.
- Consider DAgger or policy-gradient fine-tuning on top of BC.
