from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict
import os
import table_config.object_spec as object_spec
import pdb
import numpy as np
import json
import random


def get_single_tidy_config(episode="single"):

    base_path = f"./exps/{episode}"
    os.makedirs(base_path, exist_ok = True)
    obs_counter = 0

    object_list, grid = object_spec.get_tidy_config(num_cup1=0, num_cup2=6, num_uten=8, horizontal=True, stick_to_left=True, num_napkin=0, num_plate=4)
    np.savetxt(f'{base_path}/grid.out', grid)
                        
    pdb.set_trace()
    num_objs = len(object_list)

    sim = Tabletop_Sim(
        width=640,
        height=640,
        indivisual_loading=True,
    )

    for i in range(num_objs):

        obj = object_list[i]
        obj_name = obj["name"]
        obj_type = obj["type"]

        if obj["color"] is None:
            pdb.set_trace()

        yes = sim.load_object(
            name = obj_name,
            type_name = obj_dict.filename[obj_type],
            mesh_name = obj_dict.type[obj_type],
            baseMass = 1,
            position = obj["pos"],
            angle = obj["angle"],
            size = obj_dict.default_size[obj_type],
            rgb= obj["color"],
            scale_factor= obj_dict.default_scale_factor[obj_type],
            material = None,
            texture=True
        )
        if yes is False:
            print(f"Collison caused by obj {obj_name}!")
            pdb.set_trace()
        else:
            sim.get_observation_nvisii(f"{base_path}/{obs_counter}")
            sim.get_observation_nvisii_cliport(f"{base_path}/{obs_counter}/clip/")
            obs_counter += 1

def batch_tidy_config(episode="batch_init", render=True, with_napkin=False):
  
    base_path = f"./exps/{episode}/with_napkin_{with_napkin}"
    os.makedirs(base_path, exist_ok = True)

    batch_tidy_config = object_spec.batch_initialize_tidy_config(napkin=with_napkin)
    with open(f'{base_path}/config.json', 'w') as fout:
        json.dump(batch_tidy_config, fout)
                        
    num_configs = len(batch_tidy_config)
    print(f"{num_configs} initial tidy configurations" )

    if render is False:
        return 

    sim = Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
        )

    for conf in batch_tidy_config:
        
        object_list, grid, spec = conf

        sim = init_config(sim, object_list)
        if with_napkin:
            curr_path = f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_napkin_{spec[3]}_hori_{spec[4]}_left_{spec[5]}"
        else:
            curr_path = f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_hori_{spec[3]}_left_{spec[4]}"
        sim.get_observation_nvisii(curr_path)
        sim.get_observation_nvisii_cliport(f"{curr_path}/clip/")

def batch_random_walk(episode="batch_traj", with_napkin=False):
    root_path = f"./exps/{episode}/with_napkin_{with_napkin}"
    os.makedirs(root_path, exist_ok = True)

    with open(f'./exps/batch_init//with_napkin_{with_napkin}/config.json', 'r') as f:
        batch_tidy_config = json.load(f)
                        
    pdb.set_trace()
    num_configs = len(batch_tidy_config)
    print(f"{num_configs} initial tidy configurations" )

    sim = Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
    )

    for conf in batch_tidy_config:
        object_list, grid, spec = conf
        grid = np.array(grid)
        if with_napkin:
            base_path = f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_napkin_{spec[3]}_hori_{spec[4]}_left_{spec[5]}"
        else:
            base_path = f"{root_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_hori_{spec[3]}_left_{spec[4]}"
        os.makedirs(base_path, exist_ok = True)

        # initialize the tidy configuration
        sim = init_config(sim, object_list)
        sim.get_observation_nvisii(f"{base_path}/0.png")

        sim.get_observation_nvisii(f"{base_path}/0")
        sim.get_observation_nvisii_cliport(f"{base_path}/0/clip/")

        random.shuffle(object_list)

        for t in range(len(object_list)):
            obj = object_list[t] 
            obj, grid = object_spec.random_walk_one_step(obj, grid)
            object_list[t] = obj
            sim.reset_obj_pose(name=obj["name"], 
                size=obj_dict.default_size[obj["type"]], 
                position=obj["pos"], 
                baseOrientationAngle=0)
        
            sim.get_observation_nvisii(f"{base_path}/{t+1}")
            sim.get_observation_nvisii_cliport(f"{base_path}/{t+1}/clip/")
        # pdb.set_trace()

               
def init_config(sim, object_list):

    sim.reset()
    num_objs = len(object_list)
    for i in range(num_objs):

        obj = object_list[i]
        obj_name = obj["name"]
        obj_type = obj["type"]

        yes = sim.load_object(
                name = obj_name,
                type_name = obj_dict.filename[obj_type],
                mesh_name = obj_dict.type[obj_type],
                baseMass = 1,
                position = obj["pos"],
                angle = obj["angle"],
                size = obj_dict.default_size[obj_type],
                rgb= obj["color"],
                scale_factor= obj_dict.default_scale_factor[obj_type],
                material = None,
                texture=True
        )
        if yes is False:
            print(f"Collison!")
    return sim

get_single_tidy_config()
# batch_tidy_config(render=False)
# batch_random_walk()
