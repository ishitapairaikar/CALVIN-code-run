# ========================
# RL Training on CALVIN
# Task: move_slider_left
# Using Debug Dataset
# ========================

import os
import hydra
import numpy as np
import gymnasium as gym
from gym import spaces
from stable_baselines3 import SAC
from calvin_env.envs.play_table_env import PlayTableSimEnv
from hydra import initialize, compose
import calvin_env

# ------------------------
# Step 0: Set CALVIN root and debug dataset path
# ------------------------
calvin_root = "/home/calvin"  # Change if needed
debug_dataset_path = os.path.join(calvin_root, "dataset/calvin_debug_dataset")
assert os.path.exists(debug_dataset_path), "Debug dataset path does not exist!"

# ------------------------
# Step 1: Load CALVIN config
# ------------------------
with initialize(config_path="calvin_env/conf/"):
    cfg = compose(
        config_name="config_data_collection.yaml",
        overrides=[
            "cameras=static_and_gripper",
            f"+datamodule.root_data_dir={debug_dataset_path}"
        ]
    )
    # Disable GUI/VR for RL training
    cfg.env["use_egl"] = False
    cfg.env["show_gui"] = False
    cfg.env["use_vr"] = False
    cfg.env["use_scene_info"] = True

# ------------------------
# Step 2: Create custom RL environment
# ------------------------
class SlideEnv(PlayTableSimEnv):
    def __init__(self, tasks=None, **kwargs):
        super().__init__(**kwargs)
        # 7D continuous action: position (x,y,z), orientation (rx,ry,rz), gripper
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        # Observation: end-effector pose + gripper (7D)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.tasks = hydra.utils.instantiate(tasks)

    def reset(self):
        super().reset()
        self.start_info = self.get_info()
        return self.get_obs()

    def get_obs(self):
        robot_obs, _ = self.robot.get_observation()
        return robot_obs[:7]

    def _success(self):
        current_info = self.get_info()
        task_filter = ["move_slider_left"]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return 'move_slider_left' in task_info

    def _reward(self):
        return int(self._success()) * 10, {}

    def _termination(self):
        done = self._success()
        return done, {'success': done}

    def step(self, action):
        # Convert gripper action to discrete (-1 close, 1 open)
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1

        # Optional: joint space action
        if len(env_action) == 8:
            env_action = {"action": env_action, "type": "joint_rel"}

        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)

        obs = self.get_obs()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info = {**r_info, **d_info}
        return obs, reward, done, info

# ------------------------
# Step 3: Instantiate environment
# ------------------------
new_env_cfg = {**cfg.env}
new_env_cfg["tasks"] = cfg.tasks
new_env_cfg.pop('_target_', None)
new_env_cfg.pop('_recursive_', None)

env = SlideEnv(**new_env_cfg)

# ------------------------
# Step 4: Train SAC agent
# ------------------------
model = SAC("MlpPolicy", env, verbose=1)

# Run a small number of timesteps for debug / proof-of-concept
model.learn(total_timesteps=5000, log_interval=4)

# Save the model
model.save("sac_slide_debug")

print("RL training completed on move_slider_left task!")
