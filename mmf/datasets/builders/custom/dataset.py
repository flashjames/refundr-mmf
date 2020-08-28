import copy
import json
import numpy as np

import torch
import csv
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset import MMFDataset
from pathlib import Path


class MMFCustomDataset(BaseDataset):
    def __init__(self, config, dataset_type, *args, **kwargs):
        self.dataset_name =  "mmf_custom"
        super().__init__(
            "mmf_custom", config, dataset_type, *args, **kwargs
        )
        import pdb;pdb.set_trace()
        # self.dataset_type = dataset_type
        file_ = f"./datasets/prisjakt/product_pages_ml__{dataset_type}.csv"
        fname = f"/home/prox/refundr/backend-web/scripts/{file}"
        my_file = Path(fname)
        if not my_file.is_file():
            fname = f"../{file}"

        with open(fname, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            dct_rows = {}
            dct_meta_ids = {}
            idx_to_index_target = {}
            index_target = -1
            max_ = 0
            for i, row in enumerate(csv_reader):

                dct_rows[i]  = row
                unique_meta_id = row["meta_product_page__id"]
                unique_meta_id_int = int(unique_meta_id)
                row["target"] = unique_meta_id_int
                if unique_meta_id_int > max_:
                    max_ = unique_meta_id_int

                if unique_meta_id not in dct_meta_ids:
                    dct_meta_ids[unique_meta_id] = [row]
                    index_target += 1
                else:
                    dct_meta_ids[unique_meta_id].append(row)
                idx_to_index_target[i] = index_target

        self.idx_to_index_target = idx_to_index_target
        self.dct_meta_ids = dct_meta_ids   
        self.num_labels = len(dct_meta_ids.keys())
        self.unique_labels = dct_meta_ids 
        self.annotation_db = dct_rows

    def __len__(self):
        return len(self.annotation_db)

    def meta_id_to_target_index(self, meta_id):
        lst = self.dct_meta_ids.keys()
        return 
        
        
    def labels(self, obj, idx):
        target = obj["target"]
        return target, torch.tensor(target)


    def __getitem__(self, idx):
        
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        #plot = sample_info["plotr"]
        #if isinstance(plot, list):
        #   plot = plot[0]
        raw_text = sample_info["name"]
        processed_sentence = self.text_processor({"text": raw_text})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        """
        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)
        """
        current_sample.answers, current_sample.targets = self.labels(sample_info, idx)
        #import pdb;pdb.set_trace()
        print(idx)
        current_sample.id = sample_info["idx"]
        
        #import pdb;pdb.set_trace()
        
        return current_sample

