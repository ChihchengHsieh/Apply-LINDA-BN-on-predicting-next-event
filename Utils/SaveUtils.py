from enum import Enum


def get_json_dict(t):
    if isinstance(t,  Enum):
        return t.value

    if isinstance(t, str):
        return t

    if isinstance(t, int) | isinstance(t, float):
        return t

    json_dict = {}

    # If it's dict
    for k, v in vars(t).items():
        if not k.endswith("__"):
            if type(v) == type:
                json_dict[k] = get_json_dict(v)
            elif isinstance(v, Enum):
                json_dict[k] = v.value
            elif isinstance(v, list):
                json_dict[k] = [get_json_dict(v_i) for v_i in v]
            else:
                json_dict[k] = v
    return json_dict
