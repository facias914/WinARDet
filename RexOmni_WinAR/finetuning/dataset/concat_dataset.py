import copy
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from finetuning.engine.registry import BUILDER
from torch.utils.data import ConcatDataset as TorchConcatDataset


class ConcatDataset(TorchConcatDataset):

    def __init__(self, datasets, tokenizer=None, data_args=None):
        datasets_instance = []
        for cfg in datasets:
            # 处理LazyObject，需要先解析
            if hasattr(cfg, "_module") and hasattr(cfg, "_imported"):
                # 这是一个LazyObject，需要解析
                cfg = cfg.build()

            if isinstance(cfg, dict):
                # 创建字典副本，避免修改原始配置
                cfg_copy = copy.deepcopy(cfg)
                cfg_copy["data_args"] = data_args
                cfg_copy["tokenizer"] = tokenizer
                datasets_instance.append(BUILDER.build(cfg_copy))
            elif isinstance(cfg, list):
                for item in cfg:
                    # 创建字典副本，避免修改原始配置
                    item_copy = copy.deepcopy(item)
                    item_copy["data_args"] = data_args
                    item_copy["tokenizer"] = tokenizer
                    datasets_instance.append(BUILDER.build(item_copy))
            else:
                raise ValueError(f"Invalid dataset configuration: {cfg}")
        super().__init__(datasets=datasets_instance)

    def __repr__(self):
        main_str = "Dataset as a concatenation of multiple datasets. \n"
        main_str += ",\n".join([f"{repr(dataset)}" for dataset in self.datasets])
        return main_str

    # 解藕实验 
    @staticmethod
    def _infer_task_type(dataset) -> str:
        name = dataset.__class__.__name__.lower()
        if "grounding" in name or "pointing" in name:
            return "det"
        if "qa" in name:
            return "vqa"
        return "unknown"

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if isinstance(item, dict):
            dataset_name = getattr(self.datasets[0], "dataset_name", None)
            task_type = None
            try:
                # TorchConcatDataset exposes `cumulative_sizes`; mimic its dataset selection
                import bisect

                dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
                dataset = self.datasets[dataset_idx]
                dataset_name = getattr(dataset, "dataset_name", dataset_name)
                task_type = self._infer_task_type(dataset)
            except Exception:
                task_type = "unknown"

            item.setdefault("dataset_name", dataset_name)
            item.setdefault("task_type", task_type)
        return item
