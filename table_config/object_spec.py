import tabletop_gym.envs.object as obj_dict
import pdb
import numpy as np
import random
import math
import itertools
import math


object_types = [
    'cup', 'knife', 'spoon', 'fork', 'plate', 'napkin'
]

object_dict = {'cup': ['cup1', 'cup2', 'cup3', 'cup4', 'cup5'], 
    'knife': ["knife1"],
    'spoon': ["spoon1"],
    'fork': ["fork1"],
    'plate': ["plate1", "plate2", "plate3"],
    'napkin': ["napkin"]}

colors = {
    "bronze" : [80., 47., 32., 255],
    "gold" : [156.,124.,56., 255],
    "silver" : [140., 140., 143, 255],
    "blue" : [53, 104, 195, 255],
    "yellow" : [244, 214, 60, 255] ,
    "green" : [50, 205, 50, 255] ,
    "red" : [255, 88, 71, 255] ,
    "purple" : [100, 104, 200, 255],
    "cyan": [94, 185, 255, 255],
    "orange": [255, 165, 0, 255] ,
    "brown": [165, 42, 42, 255] ,
    "pink": [240, 128, 128, 255] ,
    "dark_green": [0, 128, 0, 255] ,
    "sky_blue": [135, 206, 250, 255] ,
    "white": [200, 200, 180, 255] ,
    "pure_white": [240, 240, 240, 255] ,
    "lemon_yellow": [255,250,155, 255],
    "lavender": [255,210,215, 255],
}

material = ['plastic', 'metallic', 'cloth']

def visualize_all():
    object_list = []

    pos_count = 0
    obj_idx = 0
    for obj_type_ in object_dict.keys():
        for obj_type in object_dict[obj_type_]:
            if obj_type == "cup1":
                for c in colors.keys():
                    color = colors[c]
                    x, y = pos_count%8*4, pos_count//8*4
                    angle = pos_count%4

                    o = {"type": obj_type, "name": len(object_list), "color": color, "pos": (x, y), "angle": angle}
                    object_list.append(o)
                    pos_count += 1
            else:
                default_color = obj_dict.texture_color[obj_type]
                if default_color is None:
                    color = colors["white"]
                else:
                    color = colors[default_color]
                x, y = pos_count%8*4, pos_count//8*4
                angle = pos_count%4
                o = {"type": obj_type, "name": len(object_list), "color": color, "pos": (x, y), "angle": angle}
                object_list.append(o)
                pos_count += 1
    pdb.set_trace()
    return object_list    

def get_tidy_config(num_cup1=25, num_cup2=10, num_uten=12, horizontal=True, stick_to_left=True, num_napkin=0, on_napkin=True):

    grid = np.zeros((32, 32, 2), dtype=np.int8)

    type_list = ["cup1", "cup2", "fork1", "napkin"]
    cluster_size = {}

    cluster_size["cup1"] = (int(np.sqrt(num_cup1))*4, int(np.sqrt(num_cup1))*4)
    cluster_size["fork1"] = (num_uten, 8)
    cluster_size["napkin"] = (num_napkin*3, 8)
    cluster_size["plate1"] = (4, 4)
    if on_napkin:
        cluster_size["fork_napkin"] = np.max((cluster_size["fork1"], cluster_size["napkin"]), axis=0).tolist()
    num_plate = 0
    if horizontal:
        cluster_size["cup2"] = (min(num_cup2, 5)*3, math.ceil(num_cup2/5)*3)        
    else: 
        cluster_size["cup2"] = (math.ceil(num_cup2/5)*3, min(num_cup2, 5)*3)
        # cluster_size["fork1"] = (10, num_uten*4)

    left_start_idx = (0, 0)
    right_start_idx = (32, 0)
    
    if on_napkin:
        type_list = ["cup1", "cup2", "fork_napkin"]
        random.shuffle(type_list)
    else:
        type_list = ["cup1", "cup2", "fork1", "napkin"]
        random.shuffle(type_list)

    obj_cluster_pos = {}
    for idx, obj in enumerate(type_list):
        if idx == 0:
            x0, y0 = 0, 0
            x1, y1 = cluster_size[obj]
            if horizontal: 
                left_start_idx = (0, y1)
            else:
                left_start_idx = (x1, 0)
            print(obj, x0, y0, x1, y1)
        elif idx == 1:
            _x, _y = cluster_size[obj]
            x0, y0 = (32 - _x, 0)
            x1, y1 = (32, _y)
            if horizontal:
                right_start_idx = (32, y1)
            else:
                right_start_idx = (x0, 0)
            print(obj, x0, y0, x1, y1)
        elif idx == 2:
            if stick_to_left:
                x0, y0 = left_start_idx
                x1, y1 = x0 + cluster_size[obj][0], y0 + cluster_size[obj][1]
                if horizontal: 
                    left_start_idx = (0, y1)
                else:
                    left_start_idx = (x1, 0)
            else:
                _x, _y = cluster_size[obj]
                x0, y0 = right_start_idx[0] - _x, right_start_idx[1]
                x1, y1 = right_start_idx[0], _y + right_start_idx[1]
                if horizontal:
                    right_start_idx = (32, y1)
                else:
                    right_start_idx = (x0, 0)
            print(obj, x0, y0, x1, y1)
        elif idx == 3: 
            if stick_to_left:
                _x, _y = cluster_size[obj]
                x0, y0 = right_start_idx[0] - _x, right_start_idx[1]
                x1, y1 = right_start_idx[0], _y + right_start_idx[1]
            else:
                x0, y0 = left_start_idx
                x1, y1 = x0 + cluster_size[obj][0], y0 + cluster_size[obj][1]
            print(obj, x0, y0, x1, y1)
            
        
        obj_cluster_pos[obj] = (x0, y0, x1, y1) 

    type_list = ["cup1", "cup2", "napkin", "fork1"]

    obj_list = []
    for _type in type_list:
        if _type == "cup1":
            color_list = np.random.choice(list(colors.keys()), num_cup1)
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_type]
            k = int(np.sqrt(num_cup1))
            for i in range(num_cup1):
                color = colors[color_list[i]]
                x, y = x_0 + i%k*4, y_0 + i//k*4
                angle = 0
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y, 0), "shape": (4, 4), "angle": angle}
                obj_list.append(o)
                grid[x:x+4, y:y+4, :] = len(obj_list)
        elif _type == "cup2": 
            color = colors["white"]
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_type]
            for i in range(num_cup2):
                if horizontal:
                    x, y = x_0 + i%5*3, y_0 + i//5*3
                else:
                    x, y = x_0 + i//5*3, y_0 + i%5*3
                angle = 0
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y, 0), "shape": (3, 3), "angle": angle}
                obj_list.append(o)
                grid[x:x+3, y:y+3, :] = len(obj_list)
        elif _type == "fork1":
            color = colors["silver"]
            if on_napkin:
                _joint_type = "fork_napkin"
                z = 1
            else:
                _joint_type = _type
                z = 0
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_joint_type]
            for i in range(num_uten):
                x, y = x_0 + i, y_0
                angle = 0
                grid[x:x+1, y:y+8, z] = len(obj_list)
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y, z*0.1), "shape": (1, 8), "angle": angle}
                obj_list.append(o)
        elif _type == "napkin":
            color = colors["white"]
            if on_napkin:
                _joint_type = "fork_napkin"
            else:
                _joint_type = _type
                
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_joint_type]
            for i in range(num_napkin):
                x, y = x_0 + i*3, y_0
                angle = 0
                grid[x:x+3, y:y+8, 0] = len(obj_list)
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y, 0), "shape": (3, 8), "angle": angle}
                obj_list.append(o)
        elif _type == "plate1":
            color = colors["white"]
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_type]
            for i in range(num_plate):
                x, y = x_0, y_0
                angle = 0
                grid[x:x+4, y:y+4].append(len(obj_list))
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y, 0), "shape": (4, 4), "angle": angle}
                obj_list.append(o)

        else:
            print("No such object!")

    print(grid)

    return obj_list, grid

def batch_initialize_tidy_config(napkin=False, on_napkin=False):

    horizontal_list = [True, False]
    stick_to_left_list = [True, False]
    batch_tidy_config = []
    if napkin:
        num_cup1_list = [4, 9]
        num_cup2_list = np.arange(4, 10).tolist()
        num_utensil_list = np.arange(3, 12).tolist()
        num_napkin_list = np.arange(3, 6).tolist()
        cart_prod = itertools.product(num_cup1_list, num_cup2_list, num_utensil_list, num_napkin_list, horizontal_list, stick_to_left_list)

        for element in cart_prod:
            num_cup1, num_cup2, num_uten, num_napkin, horizontal, stick_to_left = element
            if on_napkin and num_napkin*3 < num_uten:
                continue
            obj_list, grid = get_tidy_config(num_cup1, num_cup2, num_uten, horizontal, stick_to_left, num_napkin, on_napkin=on_napkin)
            batch_tidy_config.append((obj_list, grid.tolist(), element))
    else:
        num_cup1_list = [4, 9, 16]
        num_cup2_list = np.arange(4, 15).tolist()
        num_utensil_list = np.arange(4, 10).tolist()
        cart_prod = itertools.product(num_cup1_list, num_cup2_list, num_utensil_list, horizontal_list, stick_to_left_list)

        for element in cart_prod:
            num_cup1, num_cup2, num_uten, horizontal, stick_to_left = element
            obj_list, grid = get_tidy_config(num_cup1, num_cup2, num_uten, horizontal, stick_to_left)
            batch_tidy_config.append((obj_list, grid.tolist(), element))

    return batch_tidy_config


def random_walk_one_step(obj, grid):
    w, h = obj["shape"]
    x0, y0, z0 = obj["pos"]
    _type = obj['type']
    print(grid)
    while True:
        x, y = random.sample(range(32), 2)
        if np.sum(grid[x:x+w, y:y+h, :]) == 0:

            # print(f"type {_type} with idx {obj['name']} orginal location {obj['pos']}")
            # print(f"New location ({x} - {x+w}, {y} - {y+h}) ")
            # pdb.set_trace()
            if _type == "fork1" or _type == "napkin":
                grid[x:x+w, y:y+h, 0] = obj["name"]
            else:
                grid[x:x+w, y:y+h, :] = obj["name"]
            grid[x0:x0+w, y0:y0+h, math.ceil(z0)] = 0
            obj["pos"] = (x, y, 0)
            break 
            
        elif np.sum(grid[x:x+w, y:y+h, 0]) != 0 and np.sum(grid[x:x+w, y:y+h, 1]) == 0 and (_type == "fork1" or _type == "napkin"):
            
            # print(f"type {_type} with idx {obj['name']} orginal location {obj['pos']}")
            # print(f"New location ({x} - {x+w}, {y} - {y+h} ")

            # pdb.set_trace()

            grid[x:x+w, y:y+h, 1] = obj["name"]
            grid[x0:x0+w, y0:y0+h, math.ceil(z0)] = 0
            obj["pos"] = (x, y, 0.1)
        
            break 

    return obj, grid

    


    

