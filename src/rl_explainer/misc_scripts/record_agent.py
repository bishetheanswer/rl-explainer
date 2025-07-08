import cv2
from ale_py import ALEInterface
from cv2 import VideoWriter
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

from rl_explainer import saliency_maps
from rl_explainer.saliency_maps import SaliencyMethod

ale = ALEInterface()

ENV_NAME = "BreakoutNoFrameskip-v4"
MODEL_PATH = "rl_explainer/models/breakout_ec2_2nd_try/best_model.zip"
SEED = 165934
N_TIMESTEPS = 2500
FRAME_SIZE = (160, 210)  # original Atari

set_random_seed(SEED)


def record_agent(saliency_method: SaliencyMethod, render: bool = False) -> None:
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

    env = VecVideoRecorder(
        env,
        ".",
        record_video_trigger=lambda x: x == 0,
        video_length=N_TIMESTEPS,
        name_prefix="raw_agent",
    )

    video_writer = get_video_writer(f"agent_{saliency_method}.mp4")

    obs = env.reset()
    done = [False]
    for _ in range(N_TIMESTEPS):
        action, _ = dqn_model.predict(obs, deterministic=True)

        saliency_map = saliency.generate_superimposed_map(obs, action)
        resized_map = cv2.resize(
            saliency_map, FRAME_SIZE, interpolation=cv2.INTER_NEAREST
        )
        video_writer.write(cv2.cvtColor(resized_map, cv2.COLOR_RGB2BGR))

        obs, _, done, _ = env.step(action)

        if render:
            env.render("human")

        if done[0]:
            obs = env.reset()

    env.close()
    video_writer.release()


def get_video_writer(video_filename: str, fps: int = 20) -> VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return VideoWriter(video_filename, fourcc, fps, FRAME_SIZE)


if __name__ == "__main__":
    record_agent(SaliencyMethod.GRAD_CAM, render=True)
