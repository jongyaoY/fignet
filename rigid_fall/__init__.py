import os
assets_root = os.path.join(os.path.dirname(__file__), "assets")

def asset_path_completion(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(assets_root, path)


