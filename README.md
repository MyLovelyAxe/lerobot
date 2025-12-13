# Lerobot for SmolVLA

This repo is forked from [official lerobot repo](https://github.com/huggingface/lerobot), especially for project [using SmolVLA inference with Isaac Sim](https://github.com/MyLovelyAxe/isaacsim_vla_ws).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Open tasks](#open_tasks)

---

## Requirements

This package is tested on the following environment configuration:

- Ubuntu22.04
- Anaconda3
- RTX3080Ti with driver 575.57.08
- CUDA version 12.9

---

## Installation

#### 1. Create conda env

Create a conda env based on python3.10 to make sure the other dependencies required by lerobot [pyproject.toml](https://github.com/huggingface/lerobot/blob/ce348a34607e2c9120fcd38b9ebfe1b37079c03c/pyproject.toml#L32):

```bash
conda create -n smolvla python=3.10
conda activate smolvla
```

#### 2. Clone and build this repository

The installation steps are almost the same with original lerobot repository, but this forked repo is especially for SmolVLA with Isaac Sim without hardware, the following commands will install from the branch `camera/zmq_socket`:

```bash
cd
git clone git@github.com:MyLovelyAxe/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

## Usage

In a new terminal, start SmolVLA with the following script, which waits for input observation (images and joint states) from Isaac Sim via ZMQ socket, and send out returned actions (absolute target joint states) to Isaac Sim also via ZMQ socket:

```bash
conda activate smolvla
cd ~/lerobot/examples/tutorial/smolvla
python smolvla_zmq.py
```

## Open tasks

For now the perception-action loop with Isaac Sim and VLA model is setup, but only zero-shot SmolVLA is tested, the performance needs to be improved by fine-tuning SmolVLA. Besides, the VLA model would be deployed on Jetson Orin Nano. Therefore the on-going open tasks of this project include:

1. Build a pipeline to generate synthetic dataset which fits lerobot format

2. Fine-tune SmolVLA for some manipulation tasks

3. Deploy fine-tuned SmolVLA on Jetson Orin Nano
