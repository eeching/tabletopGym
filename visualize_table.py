from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict
import os
import table_config.object_spec as object_spec
import pdb
import numpy as np
import json
import random


def create_single_config(base_path, num_cup1=9, num_cup2=10, num_uten=5, horizontal=True, stick_to_left=True, num_napkin=2, on_napkin=True):

    object_list, grid = object_spec.get_tidy_config(num_cup1=num_cup1, num_cup2=num_cup2, num_uten=num_uten, horizontal=horizontal, stick_to_left=stick_to_left, num_napkin=num_napkin, on_napkin=on_napkin)

    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/tidy_grid.npy', 'wb') as f:
        np.save(f, grid)


    with open(f'{base_path}/tidy_object.json', "w") as fp:
        json.dump(object_list, fp) 

    object_list, grid = get_untidy_config(object_list, grid)

    with open(f'{base_path}/untidy_grid.npy', 'wb') as f:
        np.save(f, grid)


    with open(f'{base_path}/untidy_object.json', "w") as fp:
        json.dump(object_list, fp) 


def get_untidy_config(object_list, grid):

    for t in range(len(object_list)):
        obj = object_list[t+1] 
        obj, grid = object_spec.random_walk_one_step(obj, grid)
        object_list[t+1] = obj

    return object_list, grid


def load_single_config(base_path, tidy):

    if tidy:
        grid_path = f'{base_path}/tidy_grid.npy'
        object_path = f'{base_path}/tidy_object.json'
    else:
        grid_path = f'{base_path}/untidy_grid.npy'
        object_path = f'{base_path}/untidy_object.json'

        
    with open(grid_path, 'rb') as f:
        grid = np.load(f)

    with open(object_path, "r") as fp:
        raw_data = json.load(fp)

        object_list = {}

        for k in raw_data.keys():
            object_list[int(k)] = raw_data[k]

    return object_list, grid


def get_single_tidy_config(episode="single", with_napkin=False, on_napkin=True):

    base_path = f"./exps/{episode}/tidy_config_demo"
    os.makedirs(base_path, exist_ok = True)
    obs_counter = 0

    if with_napkin:
        num_napkin = 3

    object_list, grid = object_spec.get_tidy_config(num_cup1=9, num_cup2=10, num_uten=9, horizontal=True, stick_to_left=True, num_napkin=num_napkin, on_napkin=on_napkin)

    with open(f'{base_path}/grid.out', 'w') as outfile:
        for i in range(grid.shape[2]):
            np.savetxt(outfile, grid[:,:,i], fmt='%i')

    with open(f'{base_path}/object.json', "w") as fp:
        json.dump(object_list, fp) 
                       
    # pdb.set_trace()
    num_objs = len(object_list)

    sim = Tabletop_Sim(
        width=640,
        height=640,
        indivisual_loading=True,
    )

    for i in range(num_objs):

        obj = object_list[i]
        obj_id = obj["id"]
        obj_type = obj["type"]

        # if obj["color"] is None:
        #     pdb.set_trace()

        yes = sim.load_object(
            name = obj_id,
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
            print(f"Collison caused by obj {obj_id}!")
            # pdb.set_trace()
        else:
            sim.get_observation_nvisii(f"{base_path}/{obs_counter}")
            sim.get_observation_nvisii_cliport(f"{base_path}/{obs_counter}/clip/")
            obs_counter += 1

def single_random_walk(episode="single", with_napkin=True, on_napkin=True):
    
    num_cup1, num_cup2, num_uten, num_napkin, horizontal, stick_to_left = 9, 6, 5, 2, True, True
    object_list, grid = object_spec.get_tidy_config(num_cup1=num_cup1, num_cup2=num_cup2, num_uten=num_uten, horizontal=horizontal, stick_to_left=stick_to_left, num_napkin=num_napkin, on_napkin=on_napkin)
    base_path = f"./exps/{episode}/with_napkin_{with_napkin}/cup1_{num_cup1}_cup2_{num_cup2}_fork_{num_uten}_napkin_{num_napkin}_hori_{horizontal}_left_{stick_to_left}_on_napkin_{on_napkin}"
    os.makedirs(base_path, exist_ok = True)

    sim = Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
    )

    sim = init_config(sim, object_list)
    sim.get_observation_nvisii(f"{base_path}/0")
    sim.get_observation_nvisii_cliport(f"{base_path}/0/clip/")

    random.shuffle(object_list)

    for t in range(len(object_list)):
        obj = object_list[t] 
        obj, grid = object_spec.random_walk_one_step(obj, grid)
        object_list[t] = obj
        sim.reset_obj_pose(name=obj["id"], 
            size=obj_dict.default_size[obj["type"]], 
            position=obj["pos"], 
            baseOrientationAngle=0)
        
        sim.get_observation_nvisii(f"{base_path}/{t+1}")
        sim.get_observation_nvisii_cliport(f"{base_path}/{t+1}/clip/")
        with open(f'{base_path}/{t+1}/grid.out', 'w') as outfile:
            for i in range(grid.shape[2]):
                np.savetxt(outfile, grid[:,:,i], fmt='%i')



def batch_tidy_config(episode="batch_init", render=True, with_napkin=False, on_napkin=False):
    
    
    if with_napkin:
        base_path = f"./exps/{episode}/with_napkin_{with_napkin}/on_napkin_{on_napkin}"
    else:
        base_path = f"./exps/{episode}/with_napkin_{with_napkin}"

    os.makedirs(base_path, exist_ok = True)

    batch_tidy_config = object_spec.batch_initialize_tidy_config(napkin=with_napkin, on_napkin=on_napkin)

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
            curr_path = f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_napkin_0_hori_{spec[3]}_left_{spec[4]}"
        sim.get_observation_nvisii(curr_path)
        sim.get_observation_nvisii_cliport(f"{curr_path}/clip/")

def batch_random_walk(episode="batch_traj", with_napkin=False, on_napkin=False):

    if with_napkin:
        base_path = f"./exps/{episode}/with_napkin_{with_napkin}/on_napkin_{on_napkin}"
        config_path = f'./exps/batch_init/with_napkin_{with_napkin}/on_napkin_{on_napkin}/config.json'
    else:
        base_path = f"./exps/{episode}/with_napkin_{with_napkin}"
        config_path = f'./exps/batch_init/with_napkin_{with_napkin}/config.json'
    os.makedirs(base_path, exist_ok = True)

    with open(config_path, 'r') as f:
        batch_tidy_config = json.load(f)
                        
    num_configs = len(batch_tidy_config)
    print(f"{num_configs} initial tidy configurations" )
    batch_tidy_config_sub_idx = np.random.choice(len(batch_tidy_config), len(batch_tidy_config), replace=False)
    # pdb.set_trace()
    
    sim = Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
    )

    for idx in batch_tidy_config_sub_idx:
        conf = batch_tidy_config[idx]
        object_list, grid, spec = conf
        grid = np.array(grid)
        if with_napkin:
            curr_path = f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_napkin_{spec[3]}_hori_{spec[4]}_left_{spec[5]}"
        else:
            curr_path = f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_napkin_0_hori_{spec[3]}_left_{spec[4]}"
        os.makedirs(base_path, exist_ok = True)

        # initialize the tidy configuration
        sim = init_config(sim, object_list)

        sim.get_observation_nvisii(f"{curr_path}/0")
        sim.get_observation_nvisii_cliport(f"{curr_path}/0/clip/")

        random.shuffle(object_list)

        for t in range(len(object_list)):
            obj = object_list[t] 
            obj, grid = object_spec.random_walk_one_step(obj, grid)
            object_list[t] = obj
            sim.reset_obj_pose(name=obj["id"], 
                size=obj_dict.default_size[obj["type"]], 
                position=obj["pos"], 
                baseOrientationAngle=0)
        
            sim.get_observation_nvisii(f"{curr_path}/{t+1}")
            sim.get_observation_nvisii_cliport(f"{curr_path}/{t+1}/clip/")
        # pdb.set_trace()

               
def init_config(sim, object_list):

    sim.reset()
    num_objs = len(object_list)
    for i in range(1, num_objs+1):

        obj = object_list[i]
        obj_id = obj["id"]
        obj_type = obj["type"]

        yes = sim.load_object(
                name = obj_id,
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

# get_single_tidy_config(with_napkin=True)
# single_random_walk(with_napkin=True, on_napkin=False)
episode = "linfeng-on-napkin"
# batch_tidy_config(episode="batch_init", render=True, with_napkin=False, on_napkin=False)
# batch_random_walk(episode=episode, with_napkin=False)
# batch_random_walk(episode=episode, with_napkin=True, on_napkin=False)
# batch_random_walk(episode=episode, with_napkin=True, on_napkin=True)




