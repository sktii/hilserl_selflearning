import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    # KeyBoardIntervention2
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

# from experiments.config import DefaultTrainingConfig
# from experiments.ram_insertion.wrapper import RAMEnv
from examples.experiments.config import DefaultTrainingConfig
# from examples.experiments.ram_insertion.wrapper import RAMEnv # Commented out

from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "left": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "right": {
            "serial_number": "none",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "left": lambda img: img,
        "wrist": lambda img: img,
        "right": lambda img: img,
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.045, 0.18, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 260
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left", "wrist", "right"]
    classifier_keys = ["left", "wrist", "right"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"
    replay_buffer_capacity = 10000

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human"):
        # env = RAMEnv(
        #     fake_env=fake_env,
        #     save_video=save_video,
        #     config=EnvConfig(),
        # )
        env = UR5eStackCubeGymEnv(render_mode=render_mode, image_obs=True, hz=8, config=EnvConfig())
        classifier=False
        # fake_env=True
        # env = GripperCloseEnv(env)
        if not fake_env:
            # env = SpacemouseIntervention(env)
            env = KeyBoardIntervention2(env)
            pass
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env



import glfw
import gymnasium as gym # Changed from import gym to gymnasium
class KeyBoardIntervention2(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.left, self.right = False, False
        self.action_indices = action_indices

        self.gripper_state = 'open'
        self.intervened = False
        self.action_length = 0.3 
        self.current_action = np.array([0, 0, 0, 0, 0, 0])
        self.flag = False
        self.key_states = {
            'w': False, 'a': False, 's': False, 'd': False,
            'j': False, 'k': False, 'l': False, ';': False,
        }

        if self.env.render_mode == "human" and hasattr(self.env, "_viewer") and self.env._viewer:
             if hasattr(self.env._viewer, "viewer") and self.env._viewer.viewer:
                  if hasattr(self.env._viewer.viewer, "window") and self.env._viewer.viewer.window:
                       glfw.set_key_callback(self.env._viewer.viewer.window, self.glfw_on_key)

    def glfw_on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_W: self.key_states['w'] = True
            elif key == glfw.KEY_A: self.key_states['a'] = True
            elif key == glfw.KEY_S: self.key_states['s'] = True
            elif key == glfw.KEY_D: self.key_states['d'] = True
            elif key == glfw.KEY_J: self.key_states['j'] = True
            elif key == glfw.KEY_K: self.key_states['k'] = True
            elif key == glfw.KEY_L: 
                self.key_states['l'] = True
                self.flag = True
            elif key == glfw.KEY_SEMICOLON:
                self.intervened = not self.intervened
                self.env.intervened = self.intervened
                print(f"Intervention toggled: {self.intervened}")

        elif action == glfw.RELEASE:
            if key == glfw.KEY_W: self.key_states['w'] = False
            elif key == glfw.KEY_A: self.key_states['a'] = False
            elif key == glfw.KEY_S: self.key_states['s'] = False
            elif key == glfw.KEY_D: self.key_states['d'] = False
            elif key == glfw.KEY_J: self.key_states['j'] = False
            elif key == glfw.KEY_K: self.key_states['k'] = False
            elif key == glfw.KEY_L: self.key_states['l'] = False

        self.current_action = [
            int(self.key_states['w']) - int(self.key_states['s']), # x
            int(self.key_states['a']) - int(self.key_states['d']), # y
            int(self.key_states['j']) - int(self.key_states['k']), # z
        ]
        self.current_action = np.array(self.current_action, dtype=np.float64)
        self.current_action *= self.action_length

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self.current_action.copy()

        if self.gripper_enabled:
            if self.flag:
                if self.gripper_state == 'open':
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
                self.flag = False
            
            if self.gripper_state == 'close':
                gripper_action = np.random.uniform(-1, -0.9, size=(1,)) 
            else:
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if self.intervened:
            return expert_a, True
        else:
            if self.gripper_enabled:
                if action[-1] < 0:
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
            
            return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        return obs, info
