import zmq
import numpy as np
import io
import cv2
import torch
import time

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


obs_context = zmq.Context()
obs_socket = obs_context.socket(zmq.SUB)
obs_socket.connect("tcp://localhost:5556")
obs_socket.setsockopt(zmq.SUBSCRIBE, b"")


act_context = zmq.Context()
act_socket = act_context.socket(zmq.PUB)
act_socket.bind("tcp://127.0.0.1:5555")

# Give subscribers a short time to connect
time.sleep(0.5)

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

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


SIMULATION_RANGE = {
    'shoulder_pan.pos': {'sim_min': -2.0, 'sim_max': 2.0},
    'shoulder_lift.pos': {'sim_min': 0.0, 'sim_max': 3.5},
    'elbow_flex.pos': {'sim_min': -3.142, 'sim_max': 0.0},
    'wrist_flex.pos': {'sim_min': -2.5, 'sim_max': 1.2},
    'wrist_roll.pos': {'sim_min': -3.142, 'sim_max': 3.142},
    'gripper.pos': {'sim_min': -0.2, 'sim_max': 2.0},
}

def rad2pos(rad: float, joint_name: str):
    """Convert radians into motor pos.
    
    Isaac Sim joint state has real radian values, while SO100 robot hardware
    use -100 to 100 as "pos". Original script for SO100 follower, i.e. using_smolvla_example.py,
    use the default value of RobotConfig.use_degrees = False, so it use MotorNormMode.RANGE_M100_100,
    check norm_mode_body of class so100_foller.SO100Follower
    """
    sim_min = SIMULATION_RANGE[joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[joint_name]['sim_max']
    pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100
    return pos

def pos2rad(pos: float, joint_name: str):
    """Convert the motor pos into real radian."""
    sim_min = SIMULATION_RANGE[joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[joint_name]['sim_max']
    rad = sim_min + (pos + 100) / 200 * (sim_max - sim_min)
    return rad



if __name__ == "__main__":

    device = torch.device("cuda")

    model_id = "lerobot/smolvla_base"
    model = SmolVLAPolicy.from_pretrained(model_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    task = "pick the red block"
    robot_type = "so100_follower"


    for _ in range(MAX_EPISODES):
        # for _ in range(MAX_STEPS_PER_EPISODE):

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

            # obs is dict, which:
            # dict(
            #   "shoulder_pan.pos": joint_value,
            #   "shoulder_lift.pos": joint_value,
            #   "elbow_flex.pos": joint_value,
            #   "wrist_flex.pos": joint_value,
            #   "wrist_roll.pos": joint_value,
            #   "gripper.pos": joint_value,
            #   "camera1": cv2.RGB image with shape h, w, c, no rotation
            #   "camera2": cv2.RGB image with shape h, w, c, no rotation
            # )

            obs = {
                "shoulder_pan.pos": rad2pos(rad=joints[0], joint_name="shoulder_pan.pos"),
                "shoulder_lift.pos": rad2pos(rad=joints[1], joint_name="shoulder_lift.pos"),
                "elbow_flex.pos": rad2pos(rad=joints[2], joint_name="elbow_flex.pos"),
                "wrist_flex.pos": rad2pos(rad=joints[3], joint_name="wrist_flex.pos"),
                "wrist_roll.pos": rad2pos(rad=joints[4], joint_name="wrist_roll.pos"),
                "gripper.pos": rad2pos(rad=joints[5], joint_name="gripper.pos"),
                "camera1": img1,
                "camera2": img2,
            }

            # TODO: check if 2 cameras is the upper limit
            # TODO: check what viewpoint angle and distance to robot (i.e. pose) should the camera have

            # obs_frame = dict(
            #   "observation.state": torch.Tensor with shape [1, 6]
            #   "observation.images.camera1": torch.Tensor of image batch with shape [1, c, h, w]
            #   "task": task string same as input,
            #   "robot_type": robot_type string same as input,
            # )
            obs_frame = build_inference_frame(
                observation=obs, 
                ds_features=DATASET_FEATURES, 
                device=device, 
                task=task, 
                robot_type=robot_type,
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)

            # returned_ction = {
            #     'shoulder_pan.pos': abs_target_value,
            #     'shoulder_lift.pos': abs_target_value,
            #     'elbow_flex.pos': abs_target_value,
            #     'wrist_flex.pos': abs_target_value, 
            #     'wrist_roll.pos': abs_target_value, 
            #     'gripper.pos': abs_target_value, 
            # }
            action = make_robot_action(action, DATASET_FEATURES)

            if count % 10 == 0:
                formatted_action = [
                    f"{action[name]:.3f}" for name in JOINT_ORDER if name in action
                ]
                print(f"returned target states: [{', '.join(formatted_action)}]")

            action_array = np.array([
                pos2rad(pos=action["shoulder_pan.pos"], joint_name="shoulder_pan.pos"),
                pos2rad(pos=action["shoulder_lift.pos"], joint_name="shoulder_lift.pos"),
                pos2rad(pos=action["elbow_flex.pos"], joint_name="elbow_flex.pos"),
                pos2rad(pos=action["wrist_flex.pos"], joint_name="wrist_flex.pos"),
                pos2rad(pos=action["wrist_roll.pos"], joint_name="wrist_roll.pos"),
                pos2rad(pos=action["gripper.pos"], joint_name="gripper.pos"),
            ],dtype=np.float32)

            # In PUB/SUB without topics, just send raw bytes.
            # (If you later want topics, you can use send_multipart([topic, payload]))
            act_socket.send(action_array.tobytes())

            # TODO: the action values seem still have changes in the stuck position, 
            # check if it is because the actions are published too fast, and the robot
            # doesn't have enough time to finish?
            count += 1

        print("Episode finished! Starting new episode...")
