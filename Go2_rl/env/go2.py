from pathlib import Path
import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class Go2Env(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file=os.path.join(Path(__file__).parents[2], "resources/robots/go2/scene.xml"),
        ctrl_cost_weight=0.05, ###### edited 0.5
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.195, 0.75), ###### edited 0.2, 1.0
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 35 ###### edited 27
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            25, ## 5
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        
        ###### begin editing
        self.feet_indices = [2, 5, 8, 11]  # 根据 mujoco 模型中脚的索引调整
        self.feet_num = len(self.feet_indices)
        self.swing_target_height = 0.08
        self.elapsed_steps = 0
        ###### end editing

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    
    ###### begin editing
    @property
    def reward_contact(self):
        res = 0.0
        period = 0.8
        offset = 0.5
        step_time = self.elapsed_steps * self.dt
        phase = (step_time % period) / period
        leg_phase = [
            phase,
            (phase + offset) % 1,
            phase,
            (phase + offset) % 1
        ]  # 假设对称步态
        for i, foot_idx in enumerate(self.feet_indices):
            is_stance = leg_phase[i] < 0.55
            contact = self.contact_forces[foot_idx, 2] > 1
            res += float(not (contact ^ is_stance))  # contact 与 stance phase 一致
        return res

    @property
    def reward_feet_swing_height(self):
        reward = 0.0
        for i, foot_idx in enumerate(self.feet_indices):
            contact = np.linalg.norm(self.contact_forces[foot_idx, :3]) > 1.
            foot_pos = self.data.xipos[foot_idx][2]  # Z 方向高度
            if not contact:
                reward -= (foot_pos - self.swing_target_height) ** 2
        return reward

    @property
    def reward_contact_no_vel(self):
        penalty = 0.0
        for i in self.feet_indices:
            contact = np.linalg.norm(self.contact_forces[i, :3]) > 1.
            if contact:
                vel = self.data.cvel[i][:3]  # local frame velocity
                penalty += np.sum(np.square(vel))
        return -penalty

    @property
    def reward_hip_pos(self):
        # 假设 hip joint 是第1、2、7、8位（你需要根据 XML 验证）
        hip_dofs = [0, 3, 6, 9]
        pos = self.data.qpos.flat[hip_dofs]
        return -np.sum(np.square(pos))
    ###### end editing


    def step(self, action):
        xy_position_before = self.get_body_com("base")[:2].copy() ###### edited torso
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("base")[:2].copy() ###### edited torso
        
        ###### begin editing
        self.elapsed_steps += 1
        ###### end editing

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward
        ###### begin editing
        contact_reward = self.reward_contact
        swing_reward = self.reward_feet_swing_height
        contact_vel_penalty = self.reward_contact_no_vel
        hip_penalty = self.reward_hip_pos
        ###### end editing

        rewards = forward_reward + healthy_reward + contact_reward + swing_reward ###### rewards = forward_reward + healthy_reward
        costs = ctrl_cost = self.control_cost(action) + contact_vel_penalty + hip_penalty ######costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            ###### begin editing
            "reward_contact": contact_reward,
            "reward_swing": swing_reward,
            "penalty_contact_vel": contact_vel_penalty,
            "penalty_hip": hip_penalty,
            ###### end editing
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        ###### begin editing
        self.elapsed_steps = 0
        ###### end editing

        return observation
