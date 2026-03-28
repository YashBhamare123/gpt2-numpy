import numpy as np

# path should have .npz only
def create_state_dict(path : str) -> dict:
    state_dict = {}
    data = np.load(path, allow_pickle= True)
    for key in data.keys():
        state_dict[key] = {
            'params' : data[key].item()['params'],
            'shape' : data[key].item()['shape']
        }
    return state_dict
