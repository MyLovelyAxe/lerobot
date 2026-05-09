import argparse
import zmq
import os
import numpy as np
import io
import cv2
import torch
import time
from PIL import Image
from pathlib import Path

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot", 
    type=str, 
    default="so100", 
    choices=["so100", "so101"],
)
args = parser.parse_args()

ROBOT = args.robot


MODEL_ID = "lerobot/smolvla_base"
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20
STORE_IMAGES = False # whether store preprocessed images from socket to /logs
IMAGES_STORE_PATH = Path("logs")

INPUT_OBSERVATION_SOCKET = "tcp://localhost:5556"
OUTPUT_ACTION_SOCKET = "tcp://127.0.0.1:5555"
EMPTY_SIGNAL_SOCKET = "tcp://localhost:5558" # signal to empty action chunk queue

JOINT_ORDER = [
    'shoulder_pan.pos',
    'shoulder_lift.pos',
    'elbow_flex.pos',
    'wrist_flex.pos',
    'wrist_roll.pos',
    'gripper.pos',
]

# define feature structure of output action
# i.e. target joint states
ACTION_FEATURES = {
    'action': {
        'dtype': 'float32',
        'shape': (6,),
        'names': [
            'shoulder_pan.pos', 
            'shoulder_lift.pos', 
            'elbow_flex.pos', 
            'wrist_flex.pos', 
            'wrist_roll.pos', 
            'gripper.pos',
        ],
    },
}

# define feature structure of intput observation
# i.e. images and current joint states
OBS_FEATURES = {
    'observation.state': {
        'dtype': 'float32',
        'shape': (6,),
        'names': [
            'shoulder_pan.pos', 
            'shoulder_lift.pos', 
            'elbow_flex.pos', 
            'wrist_flex.pos', 
            'wrist_roll.pos', 
            'gripper.pos',
        ],
    },
    'observation.images.camera1': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
    'observation.images.camera2': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
}

DATASET_FEATURES = {**ACTION_FEATURES, **OBS_FEATURES}

# radian range of so100 joints in Isaac Sim
SIMULATION_RANGE = {
    # TODO: for now, always use so100 range, no matter so100 or so101
    "so100": {
        'shoulder_pan.pos': {'sim_min': -2.0, 'sim_max': 2.0},
        'shoulder_lift.pos': {'sim_min': 0.0, 'sim_max': 3.5},
        'elbow_flex.pos': {'sim_min': -3.142, 'sim_max': 0.0},
        'wrist_flex.pos': {'sim_min': -2.5, 'sim_max': 1.2},
        'wrist_roll.pos': {'sim_min': -3.142, 'sim_max': 3.142},
        'gripper.pos': {'sim_min': -0.2, 'sim_max': 2.0},
    },
    "so101": {
        'shoulder_pan.pos': {'sim_min': -1.92, 'sim_max': 1.92},
        'shoulder_lift.pos': {'sim_min': -1.75, 'sim_max': 1.75},
        'elbow_flex.pos': {'sim_min': -1.69, 'sim_max': 1.69},
        'wrist_flex.pos': {'sim_min': -1.66, 'sim_max': 1.66},
        'wrist_roll.pos': {'sim_min': -2.74, 'sim_max': 2.84},
        'gripper.pos': {'sim_min': -0.17, 'sim_max': 1.74},
    },
}

# TODO: update rad2pos and pos2rad according to norm mode, since gripper uses a different norm range

def rad2pos(rad: float, joint_name: str, robot: str):
    """Convert radians into motor pos.
    
    Isaac Sim joint state has real radian values, while SO100 robot hardware
    use -100 to 100 as "pos". Original script for SO100 follower, i.e. using_smolvla_example.py,
    use the default value of RobotConfig.use_degrees = False, so it use MotorNormMode.RANGE_M100_100,
    check norm_mode_body of class so100_foller.SO100Follower
    """

    sim_min = SIMULATION_RANGE[robot][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[robot][joint_name]['sim_max']

    # TODO: use this after fine-tuning
    # if joint_name == 'gripper.pos':
    #     # norm = ((bounded_val - min_) / (max_ - min_)) * 100
    #     pos = ((rad - sim_min) / (sim_max - sim_min)) * 100
    # else:
    #     # norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
    #     pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100

    # TODO: this gives a larger gripper joint value, even though it should use MotorNormMode.RANGE_0_100
    pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100
    return pos


def pos2rad(pos: float, joint_name: str, robot: str):
    """Convert the noramlized motor pos into real radian."""

    sim_min = SIMULATION_RANGE[robot][joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[robot][joint_name]['sim_max']

    # TODO: use this after fine-tuning
    # if joint_name == 'gripper.pos':
    #     # unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
    #     rad = (pos / 100) * (sim_max - sim_min) + sim_min
    # else:
    #     # unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
    #     rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    
    # TODO: use correct version for gripper later
    rad = (pos + 100) / 200 * (sim_max - sim_min) + sim_min
    return rad


if __name__ == "__main__":

    # to receive observation as input
    obs_context = zmq.Context()
    obs_socket = obs_context.socket(zmq.SUB)
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")

    # to receive a signal to decide if empty current action chunk
    signal_context = zmq.Context()
    signal_socket = signal_context.socket(zmq.SUB)
    signal_socket.connect(EMPTY_SIGNAL_SOCKET)
    signal_socket.setsockopt(zmq.SUBSCRIBE, b"")

    # to send out the proposed action chunk
    act_context = zmq.Context()
    act_socket = act_context.socket(zmq.PUB)
    act_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5) # Give subscribers a short time to connect

    device = torch.device("cuda")

    model = SmolVLAPolicy.from_pretrained(MODEL_ID)
    # model = model.to(device)
    # model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Robot model: {ROBOT}")

    os.makedirs(IMAGES_STORE_PATH, exist_ok=True)

    # test other configs
    # model.config.n_action_steps = 5

    camera_feature_keys = list(model.config.image_features)
    max_supported_cameras = len(camera_feature_keys)
    if max_supported_cameras == 0:
        raise ValueError(f"Policy {MODEL_ID} exposes no camera inputs.")

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    task = "pick the red block"
    robot_type = "so100_follower"

    print(f"Current task: {task}")

    count = 0
    while True:

        payload = obs_socket.recv()   # one npz blob
        buf = io.BytesIO(payload)
        data = np.load(buf)

        ts = float(data["ts"][0])
        joints = data["joints"].astype(np.float32)

        # image 1: from topic /camera1_rgb
        img1_vec = data["img1"]
        encoded_flag = int(data.get("img1_encoded", np.array([1]))[0])
        # convert JPEG bytes to numpy image
        if encoded_flag == 1:
            buf1 = np.frombuffer(img1_vec.tobytes(), dtype=np.uint8)
            img1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)
        else:
            img1 = img1_vec.reshape((480, 640, 3))
        # convert img into RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # image 2: from topic /camera2_rgb
        img2_vec = data["img2"]
        encoded_flag = int(data.get("img2_encoded", np.array([1]))[0])
        # convert JPEG bytes to numpy image
        if encoded_flag == 1:
            buf2 = np.frombuffer(img2_vec.tobytes(), dtype=np.uint8)
            img2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)
        else:
            img2 = img2_vec.reshape((480, 640, 3))
        # convert img into RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # The expected obs should have this structure:
        # {
        #   "shoulder_pan.pos": joint_value,
        #   "shoulder_lift.pos": joint_value,
        #   "elbow_flex.pos": joint_value,
        #   "wrist_flex.pos": joint_value,
        #   "wrist_roll.pos": joint_value,
        #   "gripper.pos": joint_value,
        #   "camera1": cv2.RGB image with shape h, w, c, no rotation
        #   "camera2": cv2.RGB image with shape h, w, c, no rotation
        # }

        # the joint states in obs should be normalized as relative position,
        # i.e. in range [-100, 100] for body joints or [0, 100] for gripper

        obs = {
            "shoulder_pan.pos": rad2pos(rad=joints[0], joint_name="shoulder_pan.pos", robot=ROBOT),
            "shoulder_lift.pos": rad2pos(rad=joints[1], joint_name="shoulder_lift.pos", robot=ROBOT),
            "elbow_flex.pos": rad2pos(rad=joints[2], joint_name="elbow_flex.pos", robot=ROBOT),
            "wrist_flex.pos": rad2pos(rad=joints[3], joint_name="wrist_flex.pos", robot=ROBOT),
            "wrist_roll.pos": rad2pos(rad=joints[4], joint_name="wrist_roll.pos", robot=ROBOT),
            "gripper.pos": rad2pos(rad=joints[5], joint_name="gripper.pos", robot=ROBOT),
            "camera1": img1,
            "camera2": img2,
        }

        # TODO: check what viewpoint angle and distance to robot (i.e. pose) should the camera have

        # the built obs_frame would have such structure:
        # {
        #   "observation.state": torch.Tensor with shape [1, 6]
        #   "observation.images.camera1": torch.Tensor of image batch with shape [1, c, h, w]
        #   "task": task string same as input,
        #   "robot_type": robot_type string same as input,
        # }
        obs_frame = build_inference_frame(
            observation=obs, 
            ds_features=DATASET_FEATURES, 
            device=device, 
            task=task, 
            robot_type=robot_type,
        )

        obs = preprocess(obs_frame)

        # save images in obs
        if STORE_IMAGES and count % 10 == 0:
            # cam1
            cam1_img = obs['observation.images.camera1'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
            cam1_img_uint8 = (cam1_img * 255).astype(np.uint8)
            cam1_img_uint8_save = Image.fromarray(cam1_img_uint8)  # expects RGB order
            cam1_img_uint8_save.save(IMAGES_STORE_PATH / f"obs_cam1_{count}.png")
            # cam2
            cam2_img = obs['observation.images.camera2'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
            cam2_img_uint8 = (cam2_img * 255).astype(np.uint8)
            cam2_img_uint8_save = Image.fromarray(cam2_img_uint8)  # expects RGB order
            cam2_img_uint8_save.save(IMAGES_STORE_PATH / f"obs_cam2_{count}.png")


        # if receive a stop signal from safety estimator,
        # empty the current action chunk
        # get empty signal
        try:
            empty_msg_bytes = signal_socket.recv(flags=zmq.NOBLOCK)
            empty = np.frombuffer(empty_msg_bytes, dtype=np.float32)[0]
        except zmq.Again:
            empty = 0.0

        if empty:
            model._queues["action"].clear()
            print(f"Too many unsafe actions, clear the current chunk......")

        action = model.select_action(obs)
        action = postprocess(action)

        # the returned action would have such structure:
        # {
        #     'shoulder_pan.pos': abs_target_value,
        #     'shoulder_lift.pos': abs_target_value,
        #     'elbow_flex.pos': abs_target_value,
        #     'wrist_flex.pos': abs_target_value, 
        #     'wrist_roll.pos': abs_target_value, 
        #     'gripper.pos': abs_target_value, 
        # }
        action = make_robot_action(action, DATASET_FEATURES)

        action_array = np.array([
            pos2rad(pos=action["shoulder_pan.pos"], joint_name="shoulder_pan.pos", robot=ROBOT),
            pos2rad(pos=action["shoulder_lift.pos"], joint_name="shoulder_lift.pos", robot=ROBOT),
            pos2rad(pos=action["elbow_flex.pos"], joint_name="elbow_flex.pos", robot=ROBOT),
            pos2rad(pos=action["wrist_flex.pos"], joint_name="wrist_flex.pos", robot=ROBOT),
            pos2rad(pos=action["wrist_roll.pos"], joint_name="wrist_roll.pos", robot=ROBOT),
            pos2rad(pos=action["gripper.pos"], joint_name="gripper.pos", robot=ROBOT),
        ],dtype=np.float32)

        if count % 10 == 0:
            print_action = ', '.join(f"{value:2f}" for value in action_array.tolist())
            print(f"Returned actions: [{print_action}]")

        # send actions to zmq socket
        act_socket.send(action_array.tobytes())

        count += 1
