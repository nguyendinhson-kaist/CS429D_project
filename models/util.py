import importlib
from einops import rearrange

def s2c(data, h=2, w=2, d=2):
    """
    Convert spatial data to channel data to reduce memory usage.
        - Embedding a cube of size (h, w, d) into a single voxel
    """
    return rearrange(data, 'b c (H h) (W w) (D d) -> b (c h w d) H W D', h=h, w=w, d=d )

def c2s(data, h=2, w=2, d=2):
    """
    Convert channel data to spatial data.
        - Transform a single voxel back into a cube of size (h, w, d)
    """
    return rearrange(data, 'b (c h w d) H W D -> b c (H h) (W w) (D d)', h=h, w=w, d=d)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)