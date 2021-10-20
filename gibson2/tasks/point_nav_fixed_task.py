from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.reward_functions.potential_reward import PotentialReward
from gibson2.reward_functions.collision_reward import CollisionReward
from gibson2.reward_functions.point_goal_reward import PointGoalReward
from gibson2.reward_functions.slack_reward import SlackReward

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker

import numpy as np


class PointNavFixedTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(PointNavFixedTask, self).__init__(env)
        self.reward_type = self.config.get('reward_type', 'l2')
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
            SlackReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.goal_format = self.config.get('goal_format', 'polar')
        self.dist_tol = self.termination_conditions[-1].dist_tol

        self.robot_traj = [self.initial_pos[:2]]
        self.trav_map = env.scene.floor_map[0] / 255.
        self.valid_area = self.get_valid_area()

        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', True
        )
        self.target_visual_object_visible_to_agent = self.config.get(
            'target_visual_object_visible_to_agent', False
        )
        self.floor_num = 0

        self.load_visualization(env)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])
        self.target_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])

        if self.target_visual_object_visible_to_agent:
            env.simulator.import_object(self.initial_pos_vis_obj)
            env.simulator.import_object(self.target_pos_vis_obj)
        else:
            self.initial_pos_vis_obj.load()
            self.target_pos_vis_obj.load()

        if env.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=[0, 1, 0, 0.3],
                radius=0.1,
                length=cyl_length,
                initial_offset=[0, 0, cyl_length / 2.0])
                for _ in range(self.num_waypoints_vis)]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path(env)
        return geodesic_dist

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        return l2_distance(env.robots[0].get_position()[:2],
                           self.target_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        if self.reward_type == 'l2':
            return self.get_l2_potential(env)
        elif self.reward_type == 'geodesic':
            return self.get_geodesic_potential(env)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        self.robot_traj = [self.robot_pos]
        self.geodesic_dist = self.get_geodesic_potential(env)
        self.shortest_path, _ = self.get_shortest_path(env, True, True)
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(PointNavFixedTask, self).get_termination(
            env, collision_links, action, info)

        info['path_length'] = self.path_length
        info['distance_to_goal'] = self.geodesic_dist
        if done:
            info['spl'] = float(info['success']) * \
                min(1.0, self.geodesic_dist / self.path_length)
        else:
            info['spl'] = 0.0

        return done, info

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(),
                                *env.robots[0].get_rpy())

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == 'polar':
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(
            env.robots[0].get_linear_velocity(),
            *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(
            env.robots[0].get_angular_velocity(),
            *env.robots[0].get_rpy())[2]
        task_obs = np.append(
            task_obs, [linear_velocity, angular_velocity])

        return task_obs

    def get_shortest_path(self,
                          env,
                          from_initial_pos=False,
                          entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return env.scene.get_shortest_path(
            self.floor_num, source, target, entire_path=entire_path)

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)

        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([shortest_path[i][0],
                                  shortest_path[i][1],
                                  floor_height]))
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(
                    pos=np.array([0.0, 0.0, 100.0]))

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
        self.robot_traj.append(self.robot_pos)

    def get_global_infos(self, env):
        occupancy_map = np.copy(self.trav_map)
        valid_area = np.copy(self.valid_area)

        goal_pos_map = np.zeros_like(self.trav_map)
        map_xy = env.scene.world_to_map(self.target_pos[:2])
        # print("goal_pos", self.target_pos, "->", map_xy)
        goal_pos_map[map_xy[0], map_xy[1]] = 1.0

        robot_pos_map = np.zeros_like(self.trav_map)
        map_xy = env.scene.world_to_map(self.robot_pos[:2])
        # print("robot_pos", self.robot_pos, "->", map_xy)
        robot_pos_map[map_xy[0], map_xy[1]] = 1.0

        robot_traj_map = np.zeros_like(self.trav_map)
        for i, pos_t in enumerate(self.robot_traj):
            map_xy = env.scene.world_to_map(pos_t)
            # print("pos_t", f"{i}", pos_t, "->", map_xy)
            robot_traj_map[map_xy[0], map_xy[1]] = (i + 1.0) / len(self.robot_traj)

        shortest_path_map = np.zeros_like(self.trav_map)
        shorest_path, _ = self.get_shortest_path(env, entire_path=True)
        for i, pos_s in enumerate(shorest_path):
            # print("pos_s", f"{i}", pos_s)
            map_xy = env.scene.world_to_map(pos_s[:2])
            # print("pos_s", f"{i}", pos_s, "->", map_xy)
            shortest_path_map[map_xy[0], map_xy[1]] = 1.0 - i / len(shorest_path)

        global_infos = {
            "occupancy": self.crop_map(occupancy_map),
            "robot_pos": self.crop_map(robot_pos_map),
            "robot_traj": self.crop_map(robot_traj_map),
            "shortest_path": self.crop_map(shortest_path_map),
            "goal_pos": self.crop_map(goal_pos_map),
        }

        return global_infos

    def get_valid_area(self):
        nonzero_y, nonzero_x = self.trav_map.nonzero()
        x_lower_bound, x_upper_bound = nonzero_x.min(), nonzero_x.max()
        y_lower_bound, y_upper_bound = nonzero_y.min(), nonzero_y.max()

        x_size = x_upper_bound - x_lower_bound
        y_size = y_upper_bound - y_lower_bound

        valid_size = 224 #max(x_size, y_size) + 6

        x_lower_padding = (valid_size - x_size) // 2
        x_upper_padding = (valid_size - x_size) - x_lower_padding
        y_lower_padding = (valid_size - y_size) // 2
        y_upper_padding = (valid_size - y_size) - y_lower_padding

        # mask = np.zeros_like(self.trav_map).astype(np.bool)
        # mask[
        #     (y_lower_bound - y_padding) : (y_upper_bound + y_padding),
        #     (x_lower_bound - x_padding) : (x_upper_bound + x_padding),
        # ] = True

        return np.array([
            y_lower_bound - y_lower_padding,
            y_upper_bound + y_upper_padding,
            x_lower_bound - x_lower_padding,
            x_upper_bound + x_upper_padding,
        ])

    def crop_map(self, original_map):
        row_min, row_max, col_min, col_max = self.valid_area.copy()
        padding_row_b = abs(min(row_min, 0))
        padding_row_u = abs(max(row_max - original_map.shape[0], 0))
        padding_col_b = abs(min(col_min, 0))
        padding_col_u = abs(max(col_max - original_map.shape[1], 0))
        new_map = np.pad(
            original_map,
            ((padding_row_b, padding_row_u), (padding_col_b, padding_col_u))
        )
        return new_map[
            row_min + padding_row_b: row_max + padding_row_b,
            col_min + padding_col_b: col_max + padding_col_b,]

    @property
    def num_global_infos(self):
        return 5