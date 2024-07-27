from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    source: str
    
    prompt: str

    targets: List[List[int]] = None

    output_dir_root: str = "/content/"

    # def __post_init__(self):
    #     self.output_dir_root.mkdir(exist_ok=True, parents=True)
