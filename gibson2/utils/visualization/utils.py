import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import cv2
import imageio
import gibson2
import scipy

AGENT_SPRITE = imageio.imread(
    os.path.join(
        str(gibson2.__path__[0]),
        "utils",
        "arrow.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))

# with open(os.path.join(
#     str(gibson2.__path__[0]),
#     "utils/visualization/assets",
#     'seg_colors.txt'
# ), 'r') as f:
#     SEG_COLOR = []
#     for line in f.readlines():
#         SEG_COLOR.append([int(t) for t in line.strip().split()])

def draw_agent(
    top_down_map,
    pos, angle,
    agent_radius_px
):
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, angle * 180 / np.pi - 90
    )
    rotated_agent = np.flip(rotated_agent, axis=1)
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    paste_overlapping_image(
        top_down_map, resized_agent, tuple(pos))
    return top_down_map

def draw_circle(
    top_down_map, point, color, radius
):
    cv2.circle(
        top_down_map, 
        tuple(point[::-1]),
        radius=radius,
        color=tuple(color),
        thickness=-1
    )
    return top_down_map

def draw_path(
    top_down_map,
    path_points,
    color,
    thickness: int = 2,
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
    Args:
        top_down_map: A colored version of the map.
        color: color code of the path, from TOP_DOWN_MAP_COLORS.
        path_points: list of points that specify the path to be drawn
        thickness: thickness of the path.
    """
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            tuple(prev_pt[::-1]),
            tuple(next_pt[::-1]),
            color,
            thickness=thickness,
        )
    return top_down_map

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        (min_pad[0]) : (foreground.shape[0] - max_pad[0]),
        (min_pad[1]) : (foreground.shape[1] - max_pad[1]),
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background

def get_top_down_map(env, traj=[]):
    top_down_map = np.copy(env.scene.original_trav_map)
    top_down_map = np.stack([top_down_map] * 3, axis=2)
    init_pos = env.scene.world_to_map(env.task.initial_pos[:2])
    goal_pos = env.scene.world_to_map(env.task.target_pos[:2])
    shortest_path = [env.scene.world_to_map(p) for p in env.task.shortest_path]

    pedestrian_positions = []
    pedestrian_waypoints = []
    if hasattr(env.task, "pedestrians"):
        pedestrian_positions = env.task.get_pedestrians_pos()
        pedestrian_positions = [env.scene.world_to_map(np.asarray(ped_pos[:2])) for ped_pos in pedestrian_positions]
    if hasattr(env.task, "pedestrian_waypoints"):
        for waypoints in env.task.pedestrian_waypoints:
            pedestrian_waypoints.append([env.scene.world_to_map(p) for p in waypoints])

    if len(traj) > 0:
        traj = [env.scene.world_to_map(p[:2]) for p in traj]
    current_pos = env.scene.world_to_map(env.robots[0].get_position()[:2])
    current_angle = env.robots[0].get_rpy()[2]

    if len(pedestrian_positions) > 0:
        for ped_pos in pedestrian_positions:
            top_down_map = draw_circle(top_down_map, ped_pos, (255, 0, 255), 5)
    if len(pedestrian_waypoints) > 0:
        for waypoints in pedestrian_waypoints:
            top_down_map = draw_path(top_down_map, waypoints, (255, 0, 255), 1)

    top_down_map = draw_path(top_down_map, shortest_path, (0, 0, 255), 1)
    if len(traj) > 0:
        top_down_map = draw_path(top_down_map, traj, (255, 0, 0), 1)
    top_down_map = draw_circle(top_down_map, init_pos, (0, 0, 255), 1)
    top_down_map = draw_circle(top_down_map, goal_pos, (0, 255, 0), 1)

    top_down_map = draw_agent(
        top_down_map,
        current_pos,
        current_angle,
        agent_radius_px=4
    )
    return top_down_map

def get_video_frame(obs, env, traj):
    frame = []
    if 'rgb' in obs:
        rgb = (obs['rgb'] * 255.0).astype(np.uint8)
        frame.append(rgb.copy())
    if 'depth' in obs:
        depth = (np.tile(obs['depth'], [1, 1, 3]) * 255.0).astype(np.uint8)
        frame.append(depth.copy())

    video_frame = np.concatenate(frame, axis=0)
    # cv2.imshow("obs", video_frame[..., ::-1])
    top_down_map = get_top_down_map(env, traj)
    top_down_map = cv2.resize(
        top_down_map, 
        (video_frame.shape[0], video_frame.shape[0]), 
        interpolation=cv2.INTER_LINEAR
    )
    frame = np.concatenate([video_frame, top_down_map], axis=1)
    return frame
    

    