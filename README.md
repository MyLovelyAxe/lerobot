# Lerobot for SmolVLA

This repo is forked from [official lerobot repo](https://github.com/huggingface/lerobot), built especially for project [using SmolVLA inference with Isaac Sim](https://github.com/MyLovelyAxe/isaacsim_vla_ws). Here provides 2 deployment method: on laptop or workstation or [Jetson Orin Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#intro).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Open tasks](#open_tasks)

---

## Requirements

On laptop or workstation:

- Ubuntu22.04
- Anaconda3
- RTX3080Ti with driver 575.57.08
- CUDA version 12.9

On Jetson Orin Nano:

- Jetson Orin Nano [8GB developer kit version]
- Jetpack 6.2.1
- Docker

---

## Installation


<details>
<summary>If run SmolVLA on laptop or workstation, install following these steps:</summary>

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

</details>


<details>
<summary>If run SmolVLA on Jetson Orin Nano, install following these steps:</summary>

#### 1. Install with docker on Jetson

Refer to the `README.md` of repository [lerobot_smolvla_docker](https://github.com/MyLovelyAxe/lerobot_smolvla_docker) to build the image and container for Lerobot SmolVLA.

</details>

---

## Usage

Firstly make sure Isaac Sim and ROS2 work refer to repository [isaacsim_vla_ws](https://github.com/MyLovelyAxe/isaacsim_vla_ws), then start SmolVLA to process. If Isaac Sim doesn't start, this process will suspend and wait for input.

Start SmolVLA with: 

<details>
<summary>on laptop or workstation:</summary>

Make sure repo lerobot is already installed in a conda env, then in a new terminal:

```bash
conda activate smolvla
cd ~/lerobot/examples/tutorial/smolvla
python smolvla_zmq.py
```

</details>


<details>
<summary>on Jetson Orin Nano:</summary>

Make sure the container `smolvla_pytorch27_container` is already created, then in a new terminal:

```bash
docker start -ai smolvla_pytorch27_container
cd /opt/lerobot/examples/tutorial/smolvla
python smolvla_zmq.py
```

</details>

---

## Open tasks

For now the perception-action loop with Isaac Sim and VLA model is setup, but only zero-shot SmolVLA is tested, the performance needs to be improved by fine-tuning SmolVLA. Therefore the on-going open tasks of this project include:

1. Build a pipeline to generate synthetic dataset which fits lerobot format

2. Fine-tune SmolVLA for some manipulation tasks