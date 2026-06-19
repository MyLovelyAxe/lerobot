"""
NOTE: This script is simplified lerobot/src/lerobot/scripts/lerobot_record.py

Records a dataset by teleoperating an SO-101 robot.

Example:

Record real dataset with real leader arm and real follower arm:

```bash
conda activate smolvla
cd ~/lerobot/isaacsim_sim2real/scripts
python so101_record.py \
    --num_episodes 2 \
    --dataset_name test_simplified
```

"""

import zmq
import time
import argparse
import threading
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
    SO101_NEW_CALIB,
    DEFAULT_HUGGING_FACE_DATASET_ROOT,
    So101Camera,
)
from lerobot.sim2real.thread import (
    reset_sim_robot_worker,
    reset_real_robot_worker,
)
from lerobot.sim2real.utils import (
    joint_state_pos2rad,
    get_obs_from_socket,
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
        default="test_sim2real_data",
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
    parser.add_argument(
        "--real",
        action="store_true",
        default=True,
        help="Whether record real dataset.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        default=True,
        help="Whether record simulated dataset.",
    )
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
    # Record real dataset with real teloperator and real robot
    real_dataset: bool = True
    # Record simulated dataset with real teloperator and simulated robot (via zmq socket)
    sim_dataset: bool = True

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@safe_stop_image_writer
def record_loop(
    robot: SO101Follower,
    teleop: SO101Leader,
    obs_socket: zmq.SyncSocket, # NOTE: for simulation dataset
    action_socket: zmq.SyncSocket, # NOTE: for simulation dataset
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
    real_dataset: LeRobotDataset | None = None,
    sim_dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
):
    
    if real_dataset is not None and real_dataset.fps != fps:
        raise ValueError(f"The real dataset fps should be equal to requested fps ({real_dataset.fps} != {fps}).")
    if sim_dataset is not None and sim_dataset.fps != fps:
        raise ValueError(f"The simulated dataset fps should be equal to requested fps ({sim_dataset.fps} != {fps}).")


    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        # And applies a pipeline to the raw robot observation, default is IdentityProcessor
        if real_dataset is not None:
            real_obs = robot.get_observation()
            real_obs_processed = robot_observation_processor(real_obs)
            real_observation_frame = build_dataset_frame(real_dataset.features, real_obs_processed, prefix=OBS_STR)
        if sim_dataset is not None:
            sim_obs = get_obs_from_socket(
                obs_socket=obs_socket,
                timeout_ms=40, # NOTE: ros2 topics sends around 30Hz, make the timeout slightly higher than this
            )
            # NOTE: in case the obs_socket failed to receive new simulated message
            if sim_obs is None:
                continue
            sim_obs_processed = robot_observation_processor(sim_obs)
            sim_observation_frame = build_dataset_frame(sim_dataset.features, sim_obs_processed, prefix=OBS_STR)

        # Get action from teleop
        if real_dataset is not None or sim_dataset is not None:
            act = teleop.get_action()
            # TODO: use real_obs or sim_obs as options
            act_processed = teleop_action_processor((act, real_obs)) # TODO: why a obs is needed for act_processed from teleop?
            # Applies a pipeline to the action, default is IdentityProcessor
            action_values = act_processed
            robot_action_to_send = robot_action_processor((act_processed, real_obs))
        
        # Send action to real robot
        if real_dataset is not None:
            _sent_action = robot.send_action(robot_action_to_send)
        # Send action to simulated robot
        if sim_dataset is not None:
            sim_action = joint_state_pos2rad(
                pos_joint_state=robot_action_to_send,
                calibration=SO101_NEW_CALIB,
            )
            action_socket.send(sim_action.tobytes())

        # Write to dataset
        if real_dataset is not None:
            action_frame = build_dataset_frame(real_dataset.features, action_values, prefix=ACTION)
            real_frame = {**real_observation_frame, **action_frame, "task": single_task}
            real_dataset.add_frame(real_frame)
        if sim_dataset is not None:
            action_frame = build_dataset_frame(sim_dataset.features, action_values, prefix=ACTION)
            sim_frame = {**sim_observation_frame, **action_frame, "task": single_task}
            sim_dataset.add_frame(sim_frame)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


# @parser.wrap()
def record(dataset_record_cfg: DatasetRecordConfig):
    init_logging()

    # create real teleoperator
    teleop = SO101Leader(
        config=SO101LeaderConfig(
            port=SO101_LEADER_PORT_ID, 
            id=SO101_LEADER_NEW_CALIB,
        ),
    )
    teleop.connect()

    # create real robot
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

    # create simulated robot
    # to receive simulated joint state as input
    obs_context = zmq.Context()
    obs_socket = obs_context.socket(zmq.SUB)
    obs_socket.setsockopt(zmq.CONFLATE, 1) # queue size is 1, new msg overwrites old msg
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)

    # to send out the proposed action chunk
    action_context = zmq.Context()
    action_socket = action_context.socket(zmq.PUB)
    action_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5) # Give subscribers a short time to connect

    # reset robots to the same initial pose
    initial_pose = teleop.get_action()
    reset_sim_thread = threading.Thread(
        target=reset_sim_robot_worker,
        args=(initial_pose, obs_socket, action_socket, JOINT_ORDER),
    )
    reset_real_thread = threading.Thread(
        target=reset_real_robot_worker,
        args=(initial_pose, robot),
    )

    if dataset_record_cfg.real_dataset:
        robot.connect()
        reset_real_thread.start()
    if dataset_record_cfg.sim_dataset:
        reset_sim_thread.start()

    if dataset_record_cfg.real_dataset:
        reset_real_thread.join()
    if dataset_record_cfg.sim_dataset:
        reset_sim_thread.join()

    # processors
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
    real_dataset = LeRobotDataset.create(
        dataset_record_cfg.repo_id,
        dataset_record_cfg.fps,
        root=DEFAULT_HUGGING_FACE_DATASET_ROOT / dataset_record_cfg.repo_id,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=dataset_record_cfg.video,
        image_writer_processes=dataset_record_cfg.num_image_writer_processes,
        image_writer_threads=dataset_record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=dataset_record_cfg.video_encoding_batch_size,
    )
    sim_dataset = LeRobotDataset.create(
        f"{dataset_record_cfg.repo_id}_simulated",
        dataset_record_cfg.fps,
        root=DEFAULT_HUGGING_FACE_DATASET_ROOT / f"{dataset_record_cfg.repo_id}_simulated",
        robot_type=robot.name,
        features=dataset_features,
        use_videos=dataset_record_cfg.video,
        image_writer_processes=dataset_record_cfg.num_image_writer_processes,
        image_writer_threads=dataset_record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=dataset_record_cfg.video_encoding_batch_size,
    )

    listener, events = init_keyboard_listener()

    try:

        with VideoEncodingManager(real_dataset):
            with VideoEncodingManager(sim_dataset):
                recorded_episodes = 0
                while recorded_episodes < dataset_record_cfg.num_episodes and not events["stop_recording"]:
                    log_say(f"Recording episode {real_dataset.num_episodes}")
                    record_loop(
                        robot=robot,
                        teleop=teleop,
                        obs_socket=obs_socket,
                        action_socket=action_socket,
                        events=events,
                        fps=dataset_record_cfg.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        real_dataset=real_dataset,
                        sim_dataset=sim_dataset,
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
                            obs_socket=obs_socket,
                            action_socket=action_socket,
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
                        if dataset_record_cfg.real_dataset:
                            real_dataset.clear_episode_buffer()
                        if dataset_record_cfg.sim_dataset:
                            sim_dataset.clear_episode_buffer()
                        continue

                    if dataset_record_cfg.real_dataset:
                        real_dataset.save_episode()
                    if dataset_record_cfg.sim_dataset:
                        sim_dataset.save_episode()
                    recorded_episodes += 1

        log_say("Stop recording", blocking=True)

        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

        if dataset_record_cfg.push_to_hub:
            if dataset_record_cfg.real_dataset:
                real_dataset.push_to_hub(tags=dataset_record_cfg.tags, private=dataset_record_cfg.private)
            if dataset_record_cfg.sim_dataset:
                sim_dataset.push_to_hub(tags=dataset_record_cfg.tags, private=dataset_record_cfg.private)

        log_say("Exiting")

    except KeyboardInterrupt:

        print("Ctrl + C received")

    finally:

        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


def main():

    args = parse_args()

    record(
        dataset_record_cfg=DatasetRecordConfig(
            repo_id=f"{args.username}/{args.dataset_name}",
            single_task=args.single_task,
            fps=15, # TODO: Note why this should be lower than 30Hz
            num_episodes=args.num_episodes,
            push_to_hub=args.push_to_hub,
            real_dataset=args.real,
            sim_dataset=args.sim,
        ),
    )


if __name__ == "__main__":

    main()
