"""
NOTE: This script is simplified lerobot/src/lerobot/scripts/lerobot_record.py

Records a dataset by teleoperating an SO-101 robot with one control loop:
- teleoperation runs at a higher rate
- dataset recording runs at a lower rate inside the same loop

Example:

```bash
conda activate smolvla
cd ~/lerobot/isaacsim_sim2real/scripts
python so101_record_async.py \
    --num_episodes 2 \
    --dataset_name test_simplified
```
"""

import argparse
import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
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
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.sim2real.constant import (
    CAMERA_CALIB_JSON_PATH,
    DEFAULT_HUGGING_FACE_DATASET_ROOT,
    INPUT_OBSERVATION_SOCKET,
    JOINT_ORDER,
    OUTPUT_ACTION_SOCKET,
    SO101_FOLLOWER_NEW_CALIB,
    SO101_FOLLOWER_PORT_ID,
    SO101_LEADER_NEW_CALIB,
    SO101_LEADER_PORT_ID,
    SO101_NEW_CALIB,
    So101Camera,
)
from lerobot.sim2real.thread import (
    reset_real_robot_worker,
    reset_sim_robot_worker,
)
from lerobot.sim2real.utils import (
    get_obs_from_socket,
    joint_state_pos2rad,
)
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say


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
        default=2,
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
    parser.add_argument(
        "--teleop_fps",
        type=int,
        default=200,
        help="Teleoperation frequency in Hz.",
    )
    parser.add_argument(
        "--record_fps",
        type=int,
        default=20,
        help="Dataset recording frequency in Hz.",
    )
    return parser.parse_args()


# TODO: move this to a record_utils.py
@dataclass
class CameraCalibration:
    """Calibration parameters for real u20cam 720P."""

    width: int | None = 640
    "Width of image"
    
    height: int | None = 480
    "Height of image"
    
    fps: int | None = 30 # TODO: rename to cam_fps
    "The frequency of camera to read frames"
    
    intrinsics: np.ndarray | None = None
    "Intrinsics parameters including focal length, principal points, shape (3, 3)"
    
    dist_coeffs: np.ndarray | None = None
    "Distortion coefficients, shape (5, )"
    
    dist_model: str = "opencv_pinhole"
    "Model of camera distortion"


# TODO: move this to a record_utils.py
@dataclass
class DatasetRecordConfig:
    """The config to setup recording of dataset"""

    repo_id: str
    "Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`)."
    
    single_task: str
    "A short but accurate description of the task performed during the recording (e.g. 'Pick the Lego block and drop it in the box on the right.')"
    
    root: str | Path | None = None
    "Root directory where the dataset will be stored (e.g. 'dataset/path')."
    
    fps: int = 15 # TODO: rename to record_fps
    "Limit the frames per second to record."
    
    teleop_fps: int = 100
    "Limit the frames per second to teleoperate."
    
    episode_time_s: int | float = 60
    "Number of seconds for data recording for each episode."
    
    reset_time_s: int | float = 60
    "Number of seconds for resetting the environment after each episode."
    
    num_episodes: int = 50
    "Number of episodes to record."
    
    video: bool = True
    "Encode frames in the dataset into video"
    
    push_to_hub: bool = True
    "Upload dataset to Hugging Face hub."
    
    private: bool = False
    "Add tags to your dataset on the hub."
    
    tags: list[str] | None = None
    "Add tags to your dataset on the hub."
    
    num_image_writer_processes: int = 0
    "Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only; "
    "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
    "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
    "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."

    num_image_writer_threads_per_camera: int = 4
    "Number of threads writing the frames as png images on disk, per camera."
    "Too many threads might cause unstable teleoperation fps due to main thread being blocked."
    "Not enough threads might cause low camera fps."

    video_encoding_batch_size: int = 1
    "Number of episodes to record before batch encoding videos"
    "Set to 1 for immediate encoding (default behavior), or higher for batched encoding"

    rename_map: dict[str, str] = field(default_factory=dict)
    "Rename map for the observation to override the image and state keys"
    
    record_real: bool = True
    "Whether record real dataset with real teloperator and real robot"
    
    record_sim: bool = True
    "Whether record simulated dataset with real teloperator and simulated robot (via zmq socket)"
    
    cam_calib: CameraCalibration | None = None
    "Configuration for real camera, if record real dataset."

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@safe_stop_image_writer
def record_loop(
    robot: SO101Follower | None,
    teleop: SO101Leader,
    obs_socket: zmq.SyncSocket | None,
    action_socket: zmq.SyncSocket | None,
    events: dict,
    fps: int,
    record_fps: int | None,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    real_dataset: LeRobotDataset | None = None,
    sim_dataset: LeRobotDataset | None = None,
    control_time_s: int | float | None = None,
    single_task: str | None = None,
):
    if real_dataset is not None and record_fps is not None and real_dataset.fps != record_fps:
        raise ValueError(f"The real dataset fps should be equal to requested fps ({real_dataset.fps} != {record_fps}).")
    if sim_dataset is not None and record_fps is not None and sim_dataset.fps != record_fps:
        raise ValueError(f"The simulated dataset fps should be equal to requested fps ({sim_dataset.fps} != {record_fps}).")

    record_period_s = None if record_fps is None else 1.0 / record_fps
    record_accumulated_s = 0.0
    latest_sim_obs = None

    start_episode_t = time.perf_counter()
    last_loop_t = start_episode_t
    timestamp = 0.0

    while timestamp < control_time_s:
        loop_start_t = time.perf_counter()
        elapsed_since_last_loop = loop_start_t - last_loop_t
        last_loop_t = loop_start_t
        if record_period_s is not None:
            record_accumulated_s += elapsed_since_last_loop

        if events["exit_early"]:
            events["exit_early"] = False
            break

        real_obs = None
        if robot is not None:
            real_obs = robot.get_observation()

        if obs_socket is not None:
            sim_obs = get_obs_from_socket(
                obs_socket=obs_socket,
                timeout_ms=1,
            )
            if sim_obs is not None:
                latest_sim_obs = sim_obs

        act = teleop.get_action()
        act_processed = teleop_action_processor((act, real_obs))
        action_values = act_processed
        robot_action_to_send = robot_action_processor((act_processed, real_obs))

        if robot is not None:
            robot.send_action(robot_action_to_send)

        if action_socket is not None:
            sim_action = joint_state_pos2rad(
                pos_joint_state=robot_action_to_send,
                calibration=SO101_NEW_CALIB,
            )
            action_socket.send(sim_action.tobytes())

        should_record = False
        if record_period_s is not None and record_accumulated_s >= record_period_s:
            should_record = True
            record_accumulated_s = 0.0

        if should_record:
            if sim_dataset is not None and latest_sim_obs is None:
                pass
            else:
                if real_dataset is not None:
                    real_obs_processed = robot_observation_processor(real_obs)
                    real_observation_frame = build_dataset_frame(real_dataset.features, real_obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(real_dataset.features, action_values, prefix=ACTION)
                    real_frame = {**real_observation_frame, **action_frame, "task": single_task}
                    real_dataset.add_frame(real_frame)

                if sim_dataset is not None:
                    sim_obs_processed = robot_observation_processor(latest_sim_obs)
                    sim_observation_frame = build_dataset_frame(sim_dataset.features, sim_obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(sim_dataset.features, action_values, prefix=ACTION)
                    sim_frame = {**sim_observation_frame, **action_frame, "task": single_task}
                    sim_dataset.add_frame(sim_frame)

        dt_s = time.perf_counter() - loop_start_t
        precise_sleep(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


def create_datasets(
    dataset_record_cfg: DatasetRecordConfig,
    robot: SO101Follower,
    dataset_features: dict,
) -> tuple[LeRobotDataset | None, LeRobotDataset | None]:
    real_dataset = None
    sim_dataset = None

    if dataset_record_cfg.record_real:
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

    if dataset_record_cfg.record_sim:
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

    return real_dataset, sim_dataset


def record(dataset_record_cfg: DatasetRecordConfig):
    init_logging()

    teleop = SO101Leader(
        config=SO101LeaderConfig(
            port=SO101_LEADER_PORT_ID,
            id=SO101_LEADER_NEW_CALIB,
        ),
    )
    teleop.connect()

    robot_camera_config = dict(
        wrist=OpenCVCameraConfig(
            index_or_path=So101Camera.wrist_cam,
            width=dataset_record_cfg.cam_calib.width,
            height=dataset_record_cfg.cam_calib.height,
            fps=dataset_record_cfg.cam_calib.fps,
            intrinsics=dataset_record_cfg.cam_calib.intrinsics,
            dist_coeffs=dataset_record_cfg.cam_calib.dist_coeffs,
            undistort=True,
            fourcc="MJPG",
        ),
        side=OpenCVCameraConfig(
            index_or_path=So101Camera.side_cam,
            width=dataset_record_cfg.cam_calib.width,
            height=dataset_record_cfg.cam_calib.height,
            fps=dataset_record_cfg.cam_calib.fps,
            intrinsics=dataset_record_cfg.cam_calib.intrinsics,
            dist_coeffs=dataset_record_cfg.cam_calib.dist_coeffs,
            undistort=True,
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

    obs_context = zmq.Context()
    obs_socket = obs_context.socket(zmq.SUB)
    obs_socket.setsockopt(zmq.CONFLATE, 1)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)

    action_context = zmq.Context()
    action_socket = action_context.socket(zmq.PUB)
    action_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5)

    initial_pose = teleop.get_action()
    reset_sim_thread = threading.Thread(
        target=reset_sim_robot_worker,
        args=(initial_pose, obs_socket, action_socket, JOINT_ORDER),
    )
    reset_real_thread = threading.Thread(
        target=reset_real_robot_worker,
        args=(initial_pose, robot),
    )

    if dataset_record_cfg.record_real:
        robot.connect()
        reset_real_thread.start()
    if dataset_record_cfg.record_sim:
        reset_sim_thread.start()

    if dataset_record_cfg.record_real:
        reset_real_thread.join()
    if dataset_record_cfg.record_sim:
        reset_sim_thread.join()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=dataset_record_cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_record_cfg.video,
        ),
    )

    real_dataset, sim_dataset = create_datasets(
        dataset_record_cfg=dataset_record_cfg,
        robot=robot,
        dataset_features=dataset_features,
    )

    listener, events = init_keyboard_listener()

    try:
        if real_dataset is not None:
            real_dataset_cm = VideoEncodingManager(real_dataset)
        else:
            real_dataset_cm = None
        if sim_dataset is not None:
            sim_dataset_cm = VideoEncodingManager(sim_dataset)
        else:
            sim_dataset_cm = None

        class NullContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with (real_dataset_cm or NullContext()):
            with (sim_dataset_cm or NullContext()):
                recorded_episodes = 0
                while recorded_episodes < dataset_record_cfg.num_episodes and not events["stop_recording"]:
                    episode_idx = real_dataset.num_episodes if real_dataset is not None else recorded_episodes
                    log_say(f"Recording episode {episode_idx}")
                    record_loop(
                        # TODO: test if it can record only on real follower or only sim follower
                        robot=robot if dataset_record_cfg.record_real else None,
                        teleop=teleop,
                        obs_socket=obs_socket if dataset_record_cfg.record_sim else None,
                        action_socket=action_socket if dataset_record_cfg.record_sim else None,
                        events=events,
                        fps=dataset_record_cfg.teleop_fps,
                        record_fps=dataset_record_cfg.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        real_dataset=real_dataset,
                        sim_dataset=sim_dataset,
                        control_time_s=dataset_record_cfg.episode_time_s,
                        single_task=dataset_record_cfg.single_task,
                    )

                    if not events["stop_recording"] and (
                        (recorded_episodes < dataset_record_cfg.num_episodes - 1) or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment")
                        record_loop(
                            robot=robot if dataset_record_cfg.record_real else None,
                            teleop=teleop,
                            obs_socket=obs_socket if dataset_record_cfg.record_sim else None,
                            action_socket=action_socket if dataset_record_cfg.record_sim else None,
                            events=events,
                            fps=dataset_record_cfg.teleop_fps,
                            record_fps=None,
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
                        if real_dataset is not None:
                            real_dataset.clear_episode_buffer()
                        if sim_dataset is not None:
                            sim_dataset.clear_episode_buffer()
                        continue

                    if real_dataset is not None:
                        real_dataset.save_episode()
                    if sim_dataset is not None:
                        sim_dataset.save_episode()
                    recorded_episodes += 1

        log_say("Stop recording", blocking=True)

        robot.disconnect()
        teleop.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

        if dataset_record_cfg.push_to_hub:
            if real_dataset is not None:
                real_dataset.push_to_hub(tags=dataset_record_cfg.tags, private=dataset_record_cfg.private)
            if sim_dataset is not None:
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

    with open(CAMERA_CALIB_JSON_PATH) as f:
        camera_calib = json.load(f)

    cam_calib = CameraCalibration(
        width=camera_calib["image_width"],
        height=camera_calib["image_height"],
        fps=30,
        intrinsics=np.array(
            camera_calib["intrinsics"]["camera_matrix"],
            dtype=np.float32,
        ).reshape(3, 3),
        dist_coeffs=np.array(
            camera_calib["distortion"]["coefficients"],
            dtype=np.float32,
        ).reshape(5,),
        dist_model=camera_calib["distortion"]["model"],
    )

    record(
        dataset_record_cfg=DatasetRecordConfig(
            repo_id=f"{args.username}/{args.dataset_name}",
            single_task=args.single_task,
            fps=args.record_fps,
            teleop_fps=args.teleop_fps,
            num_episodes=args.num_episodes,
            push_to_hub=args.push_to_hub,
            record_real=args.real,
            record_sim=args.sim,
            cam_calib=cam_calib,
        ),
    )


if __name__ == "__main__":
    main()
