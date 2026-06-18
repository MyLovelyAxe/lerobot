"""
NOTE: This script is simplified lerobot/src/lerobot/scripts/lerobot_record.py

Records a dataset by teleoperating an SO-101 robot.

Example:

$ cd ~/lerobot/isaacsim_sim2real/scripts

Record real dataset with real leader arm and real follower arm:

$ python so101_record.py \
    --num_episodes 2 \
    --dataset_name test_simplified

"""

import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.sim2real.constant import (
    SO101_FOLLOWER_PORT_ID,
    SO101_LEADER_PORT_ID,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_LEADER_NEW_CALIB,
    OUTPUT_ACTION_SOCKET,
    INPUT_OBSERVATION_SOCKET,
    JOINT_ORDER,
    So101Camera,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--username",
        type=str,
        default="hardli",
        help="In order to construct repo_id.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="test_simplified",
        help="In order to construct repo_id.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default="2",
        help="How many episodes to record for this dataset.",
    )
    parser.add_argument(
        "--single_task",
        type=str,
        default="Grab the cube",
        help="Instruction input of the task.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Whether push the result dataset to Hugging Face Hub.",
    )
    # TODO: add option to record real dataset or simulated dataset
    return parser.parse_args()


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@safe_stop_image_writer
def record_loop(
    robot: SO101Follower,
    teleop: SO101Leader,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        act = teleop.get_action()
        act_processed = teleop_action_processor((act, obs))

        # Applies a pipeline to the action, default is IdentityProcessor
        action_values = act_processed
        robot_action_to_send = robot_action_processor((act_processed, obs))

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


# @parser.wrap()
def record(dataset_record_cfg: DatasetRecordConfig) -> LeRobotDataset:
    init_logging()

    # create robots
    teleop = SO101Leader(
        config=SO101LeaderConfig(
            port=SO101_LEADER_PORT_ID, 
            id=SO101_LEADER_NEW_CALIB,
        ),
    )
    robot_camera_config = dict(
        wrist=OpenCVCameraConfig(
            index_or_path=So101Camera.wrist_cam, 
            width=640, 
            height=480, 
            fps=30,
            fourcc="MJPG",
        ),
        side=OpenCVCameraConfig(
            index_or_path=So101Camera.side_cam, 
            width=640, 
            height=480, 
            fps=30,
            fourcc="MJPG",
        ),
    )
    robot = SO101Follower(
        config=SO101FollowerConfig(
            port=SO101_FOLLOWER_PORT_ID, 
            id=SO101_FOLLOWER_NEW_CALIB,
            cameras=robot_camera_config,
        ),
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=dataset_record_cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_record_cfg.video,
        ),
    )

    # Create empty dataset or load existing saved episodes
    dataset = LeRobotDataset.create(
        dataset_record_cfg.repo_id,
        dataset_record_cfg.fps,
        root=dataset_record_cfg.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=dataset_record_cfg.video,
        image_writer_processes=dataset_record_cfg.num_image_writer_processes,
        image_writer_threads=dataset_record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=dataset_record_cfg.video_encoding_batch_size,
    )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < dataset_record_cfg.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}")
            record_loop(
                robot=robot,
                teleop=teleop,
                events=events,
                fps=dataset_record_cfg.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=dataset_record_cfg.episode_time_s,
                single_task=dataset_record_cfg.single_task,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < dataset_record_cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    fps=dataset_record_cfg.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=dataset_record_cfg.reset_time_s,
                    single_task=dataset_record_cfg.single_task,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if dataset_record_cfg.push_to_hub:
        dataset.push_to_hub(tags=dataset_record_cfg.tags, private=dataset_record_cfg.private)

    log_say("Exiting")
    return dataset


def main():

    args = parse_args()

    record(
        dataset_record_cfg=DatasetRecordConfig(
            repo_id=f"{args.username}/{args.dataset_name}",
            num_episodes=args.num_episodes,
            single_task=args.single_task,
            push_to_hub=args.push_to_hub,
        ),
    )


if __name__ == "__main__":

    main()
