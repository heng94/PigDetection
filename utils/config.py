from omegaconf import OmegaConf


def load_config(cfg_file):
    
    cfg = OmegaConf.load(cfg_file)
    
    return cfg