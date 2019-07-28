import gym
from gym import spaces
import numpy as np


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)
        self.simulator = env;
        self.previous_angle = None

    def reset(self, **kwargs):
        self.previous_angle = None
        print('reset reward function')
        return self.simulator.reset(**kwargs)

    def reward(self, reward):

        # get_agent_info()
        
        speed = self.simulator.speed
        current_position = self.simulator.cur_pos
        current_angle = self.simulator.cur_angle

        print(f'speed: {speed} position: {current_position} angle: {current_angle}')
        # check if the bot is in the lane.
        try:
            lane_position = self.simulator.get_lane_pos2(current_position, current_angle)
        except NotInLane:
            # the further you are away the bigger the penalty
            lane_reward = -20 * np.abs(lane_position.dist) * speed
        else:
            # the closer you are the centre the greater the reward
            lane_reward = 20 / np.abs(lane_position.dist) * speed

        # check if the bot is wobbling, waddling or shaking
        # we want a smooth driving experience
        if self.previous_angle == None: 
            angle_difference = 0
        else:
            angle_difference = 180 * np.abs(self.previous_angle - current_angle)
        # If the angle is greater than 20 degrees
        if angle_difference > 20:
            angle_reward = -10 * angle_difference * speed
        else:
            angle_reward = 10 * speed

        # Compute the collision avoidance penalty
        is_too_close_to_obstacle = self.simulator._proximity_penalty2(current_position, current_angle)

        # high speed crashes are to be discouraged
        has_collision_penalty = (is_too_close_to_obstacle > 0)
        if has_collision_penalty:
            colission_penalty = (-10 + collision_penalty) * speed
        else: 
            colission_penalty = 10 * speed

        # Compute the reward
        reward = lane_reward + angle_reward + colission_penalty
        self.previous_angle = current_angle;
        print(f'toal current reward: {reward}')
        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self,
                 env,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102
                 ):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist

    def action(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels
