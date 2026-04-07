# StudyOnWM

This repository is for studying and experimenting with world models.

The project is still being updated, so the codebase mainly serves as a research and experimentation workspace rather than a polished framework.

## Current Focus

- understanding world model ideas
- running small-scale experiments
- trying simple training pipelines on datasets such as `TwoRoom`

## Environment

This project is developed in a local conda environment with Python 3.10.

Useful files:

- [requirements_frozen_v1.txt](/Users/guanchendu/Code/StudyOnWM/requirements_frozen_v1.txt)
- [environment_v1.json](/Users/guanchendu/Code/StudyOnWM/environment_v1.json)

## Data

The current `TwoRoom` dataset is expected at:

```text
./StudyOnWM/data/tworoom.h5
```

## Run

Example training command:

```bash
/opt/anaconda3/envs/wm/bin/python ./Code/StudyOnWM/src/train_tworoom_worldmodel.py
```

## Note

This repository is under active development, so the structure and scripts may continue to change.
