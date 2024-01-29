import kedro
from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import _create_hook_manager

class OtherHooks():
    pass

class NodeHooks():
    def __init__(self):
        config_loader = OmegaConfigLoader(conf_source="./conf/", env="base")
        parameters = config_loader.get("parameters")
        self.skip_preprocessing = parameters["skip_preprocessing"]
        self.skip_training = parameters["skip_training"]
        print(f"Skip preprocessing: {self.skip_preprocessing}")
        print(f"Skip training: {self.skip_training}")

    @hook_impl
    def before_node_run(self, node):
        print(node.tags)
        if "preprocessing" in node.tags and self.skip_preprocessing:
            print(f"Skipping {node.name}")
            return None
        if "training" in node.tags and self.skip_training:
            print(f"Skipping {node.name}")
            return None
        