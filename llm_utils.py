import openai
import json
import yaml
import webcolors
import os
import pdb
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def config_to_llmquery(config, conf_logdir=None, output_logdir=None):

    if conf_logdir is not None:
        with open(conf_logdir) as fp:
            config = json.load(fp)

    description_of_table = """
        There is table divided into 32 x 32 grids.
        Each object can occupy one or more grid cells.
        The position of each object is represented by it's position with respect to top left corner (x,y,z) and it's size as length, width, and height of the grids it occupies (l,w,h)."""
    objects = []
    positions = []

    for obj in config:
        _, obj_color = get_colour_name(obj['color'][:3])
        obj_type = f"{obj['type']}_{obj_color}"
        id = f"{obj_type}_{obj['id']}"
        size = (obj['shape'][0], obj['shape'][1], obj['shape'][2])
        pos = (obj['pos'][0], obj['pos'][1], obj['pos'][2])

        objects.append(id)
        positions.append(f"{id} at {pos} size {size}")
    
    llmquery = {
        "description_of_table": description_of_table,
        "objects_on_table": objects,
        "position_of_objects": positions
    }

    if output_logdir is not None:
        with open(output_logdir, "w") as fp:
            yaml.dump(llmquery, fp)

    return llmquery

def load_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def load_message(raw_message):

    if "content" in raw_message:
        content = raw_message["content"]
    elif "content_yaml" in raw_message:
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), raw_message["content_yaml"])
        content = load_yaml(file_path)

    message = {
        "role" : raw_message["role"], 
        "content" : json.dumps(content), 
    }
    return message

def load_prompt(file_path):
        
    raw_prompt = load_yaml(file_path)
    messages = [load_message(m) for m in raw_prompt["messages"]]   
    prompt = {"model": raw_prompt["model"], "messages": messages, "temperature": raw_prompt["temperature"]}

    return prompt


def get_llm_response(prompt_dir, scene, action_type, scene_dir=None, output_dir=None):
    # Load Open API Key
    prompt = load_prompt(prompt_dir)

    if scene_dir is not None:
        scene = load_yaml(scene_dir)

    prompt["messages"].append({
            "role": "user",
            "content": json.dumps(scene)
        })

    response = openai.ChatCompletion.create(
            model = prompt["model"],
            messages = prompt["messages"],
            temperature = prompt["temperature"],
    )
    pdb.set_trace()
    usage = response['usage']['total_tokens']
    action_proposal = response['choices'][0]['message']['content']

    action_proposal = yaml.safe_load(action_proposal)
    # Print or save response
    if output_dir is not None:
        with open(output_dir, "w") as f:
            yaml.dump(action_proposal, f)
    else:
        print(response)
        print(f"Total Usage Tokens  {usage}")

    action_list = parse_response(action_type, action_proposal)

    return action_list

def parse_response(action_type, response):

    response = response["instructions"]
    
    action_list = []
    for step in response:
        action_list.append(step["action"])

    action_list = parse_action(action_type, action_list)

    return action_list

    
def parse_action(action_type, action_list):

    if action_type == "relative":
        regex = r'\b(?:near|on top of|\w+_\w+_\d+)\b'
        action_list = [re.findall(regex, action) for action in action_list]
        action_list = [(parse_object(action[0]), action[1], parse_object(action[2])) for action in action_list]
        
    else:
        regex = r'\b(?:near|on top of|\w+_\w+_\d+)\b|\(\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\)'
        action_list = [re.findall(regex, action) for action in action_list]
        action_list = [(parse_object(action[0]), action[1], parse_object(action[2]), tuple(map(int, action[3][1:-1].split(',')))) for action in action_list]

    return action_list


def parse_object(obj_str):
    regex = r'(\w+)_(\w+)_(\d+)'
    result = re.findall(regex, obj_str)[0]
    obj = {"type": result[0], "id": int(result[2])}
    return obj
    


# prompt_dir = "lang/data/prompts/action_proposal/zeroshot_absolute.yaml"
# scene_dir = "lang/data/scenes/simple_untidy_001.yaml"
# output_dir = "lang/data/actions_proposals/absolute_actions/zeroshot_simple_001.yaml"

# get_llm_response(prompt_dir, scene_dir, output_dir)


    


    