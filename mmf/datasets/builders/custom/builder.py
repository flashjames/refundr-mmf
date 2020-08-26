
from mmf.common.registry import registry
from mmf.datasets.builders.custom.dataset import (
    MMFCustomDataset,
)
from mmf.datasets.builders.vqa2.builder import VQA2Builder
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder

@registry.register_builder("mmbt_custom")
class MMFCustomBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("mmbt_custom")
        self.dataset_name = "mmbt_custom"
        self.dataset_class =  MMFCustomDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/custom/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        
        config = config

        if config.use_features:
            self.dataset_class = MMIMDbFeaturesDataset

        obj = self.dataset_class(config, dataset_type)
        self.dataset = obj

        return self.dataset



    def build(self, config, dataset_type="train", *args, **kwargs):
        pass
