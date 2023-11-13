import json




def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

