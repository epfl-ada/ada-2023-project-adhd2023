# ######################
# Some useful utilities.
# ######################


import json, os, pickle


def save_json(objects, path):
    """
    Save a list of objects as JSON (.txt).
    """

    # Remove the file if it exists
    if os.path.exists(path):
        os.remove(path)
    for obj in objects:
        # 'a' stands for 'append' to the end of the file
        # '+' to create the file if it doesn't exist
        with open(path, 'a+') as f:
            f.write(json.dumps(obj))
            f.write('\n')


def load_json(path):
    """
    Read a JSON from a text file. Expect a list of objects.
    """

    with open(path) as f:
        lines = f.readlines()
    return [json.loads(s) for s in lines]


def save_pkl(obj, path):
    """
    Save an object to path.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    """
    Load a pickle from path.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
