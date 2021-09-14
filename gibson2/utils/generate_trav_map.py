#!/usr/bin/env python

import os
import sys
import time
import random
import gibson2
import argparse

from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.map_utils import gen_trav_map
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene,SCENE_SOURCE
from gibson2.utils.assets_utils import get_ig_scene_path,get_cubicasa_scene_path,get_3dfront_scene_path, get_ig_category_path

"""
script to generate all traversability maps:

for file in ../../gibson2/ig_dataset/scenes/*
  python generate_trav_map.py $(basename $file)

to generate traversability maps for cubicasa5k or 3dfront:
pass in additional flag --source CUBICASA or --source THREEDFRONT

"""

def generate_trav_map(scene_name, scene_source, load_full_scene=True):
    if scene_source not in SCENE_SOURCE:
        raise ValueError(
            'Unsupported scene source: {}'.format(scene_source))
    if scene_source == "IG":
        scene_dir = get_ig_scene_path(scene_name)
    elif scene_source == "CUBICASA":
        scene_dir = get_cubicasa_scene_path(scene_name)
    else:
        scene_dir = get_3dfront_scene_path(scene_name)
    random.seed(0)
    scene = InteractiveIndoorScene(scene_name, 
                                   build_graph=False,
                                   texture_randomization=False,
                                   scene_source=scene_source,
                                   not_load_object_categories=['door', 'carpet'],
                                   should_open_all_doors=False)
    if not load_full_scene:
        scene._set_first_n_objects(3)
    s = Simulator(mode='headless', image_width=512,
                  image_height=512, device_idx=0)
    s.import_ig_scene(scene)
    
    if load_full_scene:
        scene.open_all_doors()

    for i in range(20):
        s.step()

    vertices_info, faces_info = s.renderer.dump()
    s.disconnect()

    if load_full_scene:
        trav_map_filename_format = 'floor_trav_{}_new.png'
        obstacle_map_filename_format = 'floor_{}_new.png'
    else:
        trav_map_filename_format = 'floor_trav_no_obj_{}_new.png'
        obstacle_map_filename_format = 'floor_no_obj_{}_new.png'

    gen_trav_map(vertices_info, faces_info, 
                 output_folder=os.path.join(scene_dir, 'layout'),
        trav_map_filename_format = trav_map_filename_format,
        obstacle_map_filename_format =obstacle_map_filename_format)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Traversability Map')
    parser.add_argument('scene_names', metavar='s', type=str,
                        nargs='+', help='The name of the scene to process')
    parser.add_argument('--source', default="IG", dest='source', help='Source of the scene, should be among [CUBICASA, IG, THREEDFRONT]')

    args = parser.parse_args()
    for scene_name in args.scene_names:
        if "int" not in scene_name:
            pass
        generate_trav_map(scene_name, args.source, False)
        generate_trav_map(scene_name, args.source, True)

if __name__ == "__main__":
    main()
