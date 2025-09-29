import numpy as np
from typing import List, Dict

# Forward reference to MetaDB to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hpo.meta.meta_db import MetaDB

class WarmStarter:
    def __init__(self, meta_db: 'MetaDB'):
        self.meta_db = meta_db

    def seed_from_similar(self, dataset_meta_vector: List[float], top_k: int = 5, top_configs_per_study: int = 3) -> List[Dict]:
        sims = self.meta_db.find_similar_studies(dataset_meta_vector, top_k=top_k)
        seeds=[]
        for study_id,_score in sims:
            # query best configs for study
            # here we assume meta_db can return top configs
            top = self.meta_db.get_top_configs(study_id, top_configs_per_study)
            seeds.extend(top)
        return seeds