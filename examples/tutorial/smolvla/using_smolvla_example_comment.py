import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def main():
    device = torch.device("cuda")
    model_id = "lerobot/smolvla_base"

    model = SmolVLAPolicy.from_pretrained(model_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # find ports using lerobot-find-port
    follower_port = "/dev/tty.usbmodem58760431631"  # something like "/dev/tty.usbmodem58760431631"

    # the robot ids are used the load the right calibration files
    follower_id = "follower_so100_test"  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        # camera_config = {'camera1': OpenCVCameraConfig(fps=30, width=640, height=480, index_or_path=0, color_mode=<ColorMode.RGB: 'rgb'>, rotation=<Cv2Rotation.NO_ROTATION: 0>, warmup_s=1, fourcc=None)}
        "camera1": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "camera2": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    }
    
    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO100Follower(robot_cfg)
    robot.connect()

    task = "pick the red block"  # something like 
    robot_type = "so100_follower"  # something like "so100_follower" for multi-embodiment datasets

    # This is used to match the raw observation keys to the keys expected by the policy
    
    # robot.action_features = dict(
    #   "shoulder_pan.pos": type<float>,
    #   "shoulder_lift.pos": type<float>,
    #   "elbow_flex.pos": type<float>,
    #   "wrist_flex.pos": type<float>,
    #   "wrist_roll.pos": type<float>,
    #   "gripper.pos": type<float>,
    #   )
    # action_features = dict(
    #   'action': dict(
    #       'dtype': 'float32',
    #       'shape': (6,), # this is a tuple
    #       'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    #       ),
    #   )
    action_features = hw_to_dataset_features(robot.action_features, "action")
    
    # robot.observation_features = robot.action_features + robot._cameras_ft
    # where the robot._cameras_ft = dict(
    #   "camera1": (480, 640, 3), # (h, w, 3)
    # )
    # so robot.observation_features = dict(
    #   "shoulder_pan.pos": type<float>,
    #   "shoulder_lift.pos": type<float>,
    #   "elbow_flex.pos": type<float>,
    #   "wrist_flex.pos": type<float>,
    #   "wrist_roll.pos": type<float>,
    #   "gripper.pos": type<float>,
    #   "camera1": (480, 640, 3), # (h, w, 3)
    #   )
    # obs_features = dict(
    #   'observation.state': dict( # same with action_features['action'] above
    #       'dtype': 'float32',
    #       'shape': (6,), # this is a tuple
    #       'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    #       ),
    #   'observation.images.camera1': dict(
    #       'dtype': 'video',
    #       'shape': (480, 640, 3),
    #       'names': ['height', 'width', 'channels'],
    #       ),
    #   )
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    
    # just combine 2 dict
    # dataset_features = dict(
    #   'action': dict(
    #       'dtype': 'float32',
    #       'shape': (6,), # this is a tuple
    #       'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    #       ),
    #   'observation.state': dict( # same with action_features['action'] above
    #       'dtype': 'float32',
    #       'shape': (6,), # this is a tuple
    #       'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    #       ),
    #   'observation.images.camera1': dict(
    #       'dtype': 'video',
    #       'shape': (480, 640, 3),
    #       'names': ['height', 'width', 'channels'],
    #       ),
    # )
    dataset_features = {**action_features, **obs_features}

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            # obs is dict, which:
            # dict(
            #   "shoulder_pan.pos": joint_value,
            #   "shoulder_lift.pos": joint_value,
            #   "elbow_flex.pos": joint_value,
            #   "wrist_flex.pos": joint_value,
            #   "wrist_roll.pos": joint_value,
            #   "gripper.pos": joint_value,
            #   "camera1": cv2.RGB image with shape h, w, c, no rotation
            # )

            # the above "camera1" image should be processed similarly with:
            #   self.videocapture = cv2.VideoCapture()
            #   ret, frame = self.videocapture.read()
            #   processed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obs = robot.get_observation()

            # obs_frame = dict(
            #   "observation.state": torch.Tensor with shape [1, 6]
            #   "observation.images.camera1": torch.Tensor of image batch with shape [1, c, h, w]
            #   "task": task string same as input,
            #   "robot_type": robot_type string same as input,
            # )
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
