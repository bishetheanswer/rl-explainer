import time

from ale_py import ALEInterface
from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecVideoRecorder,
)

ale = ALEInterface()

ENV_NAME = "BreakoutNoFrameskip-v4"
FILENAME = "trial.mp4"
MODEL_FILENAME = "dqn-BreakoutNoFrameskip-v4.zip"

checkpoint = load_from_hub(
    repo_id="sb3/dqn-BreakoutNoFrameskip-v4",
    filename=MODEL_FILENAME,
)
print(checkpoint)
print(type(checkpoint))

model = DQN.load(checkpoint)


eval_env = make_vec_env(ENV_NAME, n_envs=1, wrapper_class=AtariWrapper)
eval_env = VecFrameStack(eval_env, n_stack=4)

video_length = 1000
eval_env = VecVideoRecorder(
    eval_env,
    "agent_videos",
    record_video_trigger=lambda x: x == 0,
    video_length=video_length,
    name_prefix=FILENAME,
)

obs = eval_env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = eval_env.step(action)
    eval_env.render("human")
    time.sleep(0.05)

eval_env.close()
