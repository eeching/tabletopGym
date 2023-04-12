from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict
import pdb
import numpy as np
import random
from visualize_table import init_config, create_single_config, load_single_config
import math
import os
from llm_utils import config_to_llmquery, get_llm_response

root_dir = "./exps/llm_env/with_napkin_True"
conf_dir = "cup1_4_cup2_0_fork_0_napkin_0_hori_True_left_True_on_napkin_True"

class LLM_tabletopGym:

    def __init__(self, object_list, grid, log_dir, render=True):

        self.object_list = object_list
        self.grid = grid
        self.log_dir = log_dir

        sim =  Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
        )

        self.sim =  init_config(sim, object_list)
        self.render = render
        self.timestep = 0

        if render:
            self.sim.get_observation_nvisii(f"{self.log_dir}/0")
            self.sim.get_observation_nvisii_cliport(f"{self.log_dir}/0/clip/")
        
    
    def step(self, action):

        # action = (object_id, end_coordinate)
        obj_id, end_pos = action
        obj = self.object_list[obj_id]

        collision, obj = self.check_collison(obj, end_pos)
        if collision is False:
            self.object_list[obj_id] = obj
            self.sim.reset_obj_pose(name=obj["id"], 
                    size=obj_dict.default_size[obj["type"]], 
                    position=obj["pos"], 
                    baseOrientationAngle=0)
            self.timestep += 1

            if self.render:
            
                self.sim.get_observation_nvisii(f"{self.log_dir}/{self.timestep}")
                self.sim.get_observation_nvisii_cliport(f"{self.log_dir}/{self.timestep}/clip/")

            return True, self.object_list
        else:
            return False, self.object_list

    def check_collison(self, obj, end_pos):

        x0, y0, z0 = obj["pos"]
        w, l, h = obj["shape"]
        id = obj["id"]
        x, y, z = end_pos

        x_s, x_e, y_s, y_e, z_s, z_e = x, x+w, y, y+l, math.ceil(z), math.ceil(z)+h

        if self.is_valid_coord(x_s, y_s, x_e, y_e, z_s, z_e) and np.sum(self.grid[x_s:x_e, y_s:y_e, z_s:z_e]) == 0:
            self.grid[x0:x0+w, y0:y0+l, math.ceil(z0):math.ceil(z0)+h] = 0
            self.grid[x_s:x_e, y_s:y_e, z_s:z_e] = id
            obj["pos"] = (x, y, z)
            return False, obj

        return True, {}

    def is_valid_coord(self, x_s, y_s, x_e, y_e, z_s, z_e):
        return x_s >= 0 and y_s >= 0 and z_s >= 0 and x_e >= 0 and y_e >= 0 and z_e >= 0 and x_s >= 0 and y_s >= 0 and z_s <= 32 and x_e <= 32 and y_e <= 32 and z_e <= 2 

    def parse_llm_action(self, action):
        # action = (object_id_pick, near, object_id_place)

        obj_pick_id, relative_pos, obj_recep_id = action

        obj_pick = self.object_list[obj_pick_id]
        obj_recep = self.object_list[obj_recep_id]

        x0, y0, z0 = obj_recep["pos"]
        w_target, l_target, h_target  = obj_recep["shape"]
        w_pick, l_pick, h_pick = obj_pick["shape"]

        end_pos = None

        if relative_pos == "on top of":
            x_s, y_s, x_e, y_e, z_s, z_e = x0, y0, x0+w_pick, y0+l_pick, 1, 1+h_pick
            
            if z0 == 0 and h_target == 1 and self.is_valid_coord(x_s, y_s, x_e, y_e, z_s, z_e) and np.sum(self.grid[x_s: x_e, y_s: y_e, 0]) == w_pick*l_pick and np.sum(self.grid[x_s:x_e, y_s:y_e, z_s:z_e]) == 0:
                end_pos = (x_s, y_s, 0.1)
        
        else: 
            possible_pos = ["top", "right", "bottom", "left"]
            random.shuffle(possible_pos)

            for pos in possible_pos:
                if pos == "top":
                    x_s, y_s, x_e, y_e, z_s, z_e = x0, y0-l_pick, x0+w_pick, y0, 0, 0+h_pick
                    if self.is_valid_coord(x_s, y_s, x_e, y_e, z_s, z_e) and np.sum(self.grid[x_s:x_e, y_s:y_e, z_s:z_e]) == 0:
                        end_pos = (x_s, y_s, 0)
                        break
                elif pos == "right":
                    x_s, y_s, x_e, y_e, z_s, z_e = x0 + w_target, y0, x0 + w_target + w_pick, y0 + l_pick, 0, 0 + h_pick
                    if self.is_valid_coord(x_s, y_s, x_e, y_e, z_s, z_e) and np.sum(self.grid[x_s:x_e, y_s:y_e, z_s:z_e]) == 0:
                        end_pos = (x_s, y_s, 0)
                        break
                elif pos == "bottom":
                    x_s, y_s, x_e, y_e, z_s, z_e = x0, y0 + l_target, x0 + w_pick, y0 + l_target + l_pick, 0, 0 + h_pick
                    if self.is_valid_coord(x_s, y_s, x_e, y_e, z_s, z_e) and np.sum(self.grid[x_s:x_e, y_s:y_e, z_s:z_e]) == 0:
                        end_pos = (x_s, y_s, 0)
                        break
                elif pos == "left":
                    x_s, y_s, x_e, y_e, z_s, z_e = x0 - w_pick, y0, x0, y0 + l_pick, 0, 0 + h_pick
                    if self.is_valid_coord(x_s, y_s, x_e, y_e, z_s, z_e) and np.sum(self.grid[x_s:x_e, y_s:y_e, z_s:z_e]) == 0:
                        end_pos = (x_s, y_s, 0)
                        break

            
        if end_pos is not None:
            return True, obj_pick["id"], end_pos
        else:
            return False, {}, {}

def init_tabletop_env(zero_shot, action_type, log_dir=None, loading=False):
    
    if loading is False:
        num_cup1, num_cup2, num_uten, num_napkin, horizontal, stick_to_left, with_napkin, on_napkin = 9, 0, 0, 0, True, True, True, True
        log_dir = f"./exps/llm_env/with_napkin_{with_napkin}/cup1_{num_cup1}_cup2_{num_cup2}_fork_{num_uten}_napkin_{num_napkin}_hori_{horizontal}_left_{stick_to_left}_on_napkin_{on_napkin}"
        create_single_config(log_dir, num_cup1=num_cup1, num_cup2=num_cup2, num_uten=num_uten, num_napkin=num_napkin, horizontal=horizontal, stick_to_left=stick_to_left, on_napkin=on_napkin)    
    
    object_list, grid = load_single_config(log_dir, tidy=False)
    llm_env = LLM_tabletopGym(object_list, grid, f"{log_dir}/{zero_shot}/{action_type}")

    return llm_env, object_list, grid, f"{log_dir}/{zero_shot}/{action_type}"


def test_action_parsing():

    log_dir = "./exps/llm_env/with_napkin_True/cup1_4_cup2_0_fork_0_napkin_0_hori_True_left_True_on_napkin_True"

    llm_env, object_list, grid = init_tabletop_env(log_dir)

    # Move 2 on the left of 1
    # action_1 = (1, (11, 2, 0))
    language_action_1 = (1, "near", 4)
    is_valid, id, end_pos = llm_env.parse_llm_action(language_action_1)
    print(is_valid, id, end_pos)
    if is_valid:
        action = (id, end_pos)
    llm_env.step(action)

    # Move 2 below 1
    # action_2 = (2, (11, 6, 0))
    language_action_2 = (2, "near", 1)
    is_valid, id, end_pos = llm_env.parse_llm_action(language_action_2)
    print(is_valid, id, end_pos)
    if is_valid:
        action = (id, end_pos)
    llm_env.step(action)

    # Move 3 below 4
    # action_3 = (3, (15, 6, 0))
    language_action_3 = (3, "near", 4)
    is_valid, id, end_pos = llm_env.parse_llm_action(language_action_3)
    print(is_valid, id, end_pos)
    if is_valid:
        action = (id, end_pos)
    llm_env.step(action)


def get_response_from_llm(zero_shot, action_type, object_list, output_dir=None):
    
    prompt_dir = f"prompts/action_proposal/{zero_shot}_{action_type}.yaml"
    llm_scene_config = config_to_llmquery(object_list)  
    action_list = get_llm_response(prompt_dir, llm_scene_config, action_type, output_dir=output_dir)

    return action_list

def env_execute_llm_action(zero_shot, action_type):
    
    llm_env, object_list, _, base_dir = init_tabletop_env(zero_shot, action_type, log_dir = f"{root_dir}/{conf_dir}", loading=False)
    action_list = get_response_from_llm(zero_shot, action_type, list(object_list.values()), output_dir=f"{base_dir}/{zero_shot}_{action_type}_ap.yaml")

    for action in action_list:
        obj_1, pos, obj_2 = action
        assert obj_1["type"] == object_list[obj_1["id"]]["type"]
        assert obj_2["type"] == object_list[obj_2["id"]]["type"]
        action = (obj_1["id"], pos, obj_2["id"])

        is_valid, id, end_pos = llm_env.parse_llm_action(action)
        print(is_valid, id, end_pos)
        if is_valid:
            action = (id, end_pos)
        llm_env.step(action)



env_execute_llm_action("zeroshot", "relative")










        