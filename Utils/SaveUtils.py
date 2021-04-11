from enum import Enum

def get_json_dict(t):
    json_dict ={}
    for k, v in  vars(t).items():
        if not k.endswith("__"):
            if type(v) == type:
                json_dict[k] = get_json_dict(v)
            elif isinstance(v, Enum):
                json_dict[k] = str(v)
            else:
                json_dict[k] = v 
    return json_dict