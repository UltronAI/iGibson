import gym
import numpy as np

from gibson2.robots.robot_locomotor import LocomotorRobot


class Locobot(LocomotorRobot):
    """
    Locobot robot
    Reference: https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
    Uses differentiable_drive / twist command control
    """

    def __init__(self, config):
        self.config = config
        # https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
        # Maximum translational velocity: 70 cm/s
        # Maximum rotational velocity: 180 deg/s (>110 deg/s gyro performance will degrade)
        self.linear_velocity = config.get('linear_velocity', 0.5)
        self.angular_velocity = config.get('angular_velocity', np.pi / 2.0)
        self.wheel_dim = 2
        self.wheel_axle_half = 0.115  # half of the distance between the wheels
        self.wheel_radius = 0.038  # radius of the wheels
        LocomotorRobot.__init__(self,
                                "locobot/locobot.urdf",
                                base_name="base_link",
                                action_dim=self.wheel_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="differential_drive")

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_high = np.zeros(self.wheel_dim)
        self.action_high[0] = self.linear_velocity
        self.action_high[1] = self.angular_velocity
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        # action list:
        # 0: move ahead 0.2m (0.4 m/s * 0.5s)
        # 1: move back 0.2m (-0.4 m/s * 0.5s)
        # 2: turn left 45 degree (pi/2 rad/s * 0.5s)
        # 3: turn right 45 degree (-pi/2 rad/s * 0.5s)
        self.action_list = [
            np.array([0.4, 0.0]), 
            np.array([-0.4, 0.0]), 
            np.array([0.0, np.pi / 2]), 
            np.array([0.0, -np.pi / 2])
        ]

        self.action_space = gym.spaces.Discrete(len(self.action_list))

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts['gripper_link'].get_position()
