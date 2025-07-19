import os
from datetime import datetime

import cv2
import numpy as np
from ale_py import ALEInterface
from cv2 import VideoWriter
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack

from rl_explainer import saliency_maps
from rl_explainer.saliency_maps import SaliencyMethod

ale = ALEInterface()

ENV_NAME = "BreakoutNoFrameskip-v4"
MODEL_PATH = "rl_explainer/models/breakout_ec2_2nd_try/best_model.zip"
SEED = 165934
N_TIMESTEPS = 2500
FRAME_SIZE = (160, 210)  # original Atari

ACTION_NAMES = {
    0: "",
    1: "",
    2: "RIGHT",
    3: "LEFT",
}  # 0 is NOOP, 1 is FIRE, for clarity we don't show them

set_random_seed(SEED)


def record_agent(
    saliency_method: SaliencyMethod,
    render: bool = False,
    overlay_actions: bool = True,
    fps: int = 30,
) -> None:
    """
    Record DQN Agent with superimposed saliency maps.
    """
    dqn_model = DQN.load(MODEL_PATH)
    target_layer = dqn_model.q_net.features_extractor.cnn[4]
    saliency = saliency_maps.initialize_saliency_method(
        saliency_method, dqn_model, target_layer
    )

    env = make_vec_env(ENV_NAME, n_envs=1, wrapper_class=AtariWrapper)
    env = VecFrameStack(env, n_stack=4)
    env.seed(SEED)

    # Create videos directory if it doesn't exist
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_video_writer = get_video_writer(
        os.path.join(videos_dir, f"{timestamp}_agent_raw.mp4"),
        fps=fps,
    )
    saliency_video_writer = get_video_writer(
        os.path.join(videos_dir, f"{timestamp}_agent_{saliency_method}.mp4"),
        fps=fps,
    )

    obs = env.reset()
    lives_lost = 0

    while lives_lost < 5:
        action, _ = dqn_model.predict(obs, deterministic=True)
        action_value = int(action[0])
        action_name = ACTION_NAMES.get(action_value, "UNKNOWN")

        raw_frame = env.render()

        saliency_map = saliency.generate_superimposed_map(obs, action)
        resized_map = cv2.resize(
            saliency_map, FRAME_SIZE, interpolation=cv2.INTER_NEAREST
        )

        if overlay_actions:
            resized_map = add_action_overlay(resized_map, action_name)
            raw_frame = add_action_overlay(raw_frame, action_name)

        saliency_video_writer.write(cv2.cvtColor(resized_map, cv2.COLOR_RGB2BGR))
        raw_video_writer.write(cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))

        obs, _, done, _ = env.step(action)

        if render:
            env.render("human")

        if done[0]:
            lives_lost += 1
            obs = env.reset()

    env.close()
    raw_video_writer.release()
    saliency_video_writer.release()

    print(f"Videos saved to {videos_dir}/ directory:")
    print(f"  - {timestamp}_agent_raw.mp4")
    print(f"  - {timestamp}_agent_{saliency_method}.mp4")


def add_action_overlay(frame: np.ndarray, action_name: str) -> np.ndarray:
    """
    Add action name overlay to the frame.

    Args:
        frame: The video frame (numpy array)
        action_name: Name of the action being taken

    Returns:
        Frame with text overlay
    """
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (5, 5), (150, 30), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)

    cv2.putText(
        frame_bgr,
        f"Action: {action_name}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def get_video_writer(video_filename: str, fps: int = 30) -> VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return VideoWriter(video_filename, fourcc, fps, FRAME_SIZE)


if __name__ == "__main__":
    record_agent(SaliencyMethod.GRAD_CAM, render=True, overlay_actions=True, fps=30)
