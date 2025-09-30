import numpy as np
from typing import List, Dict, Any, Optional
from hpo.database import ExperimentDatabase
from hpo.configuration import ConfigurationSpace
from hpo.meta.meta_learner import MetaLearner

class WarmStartManager:
    """
    Manages warm-starting by:
    1. Finding similar past studies from the DB based on meta-features.
    2. Training a MetaLearner on their results.
    3. Using the MetaLearner to rank candidate configs for a new study.
    """

    def __init__(self, db: ExperimentDatabase, config_space: ConfigurationSpace):
        self.db = db
        self.cs = config_space
        self.meta_learner = MetaLearner()

    def find_and_train(self, current_meta_features: Dict[str, Any], meta_key: str):
        # 1. Find similar studies
        similar_study_ids = self.db.find_similar_studies(meta_key, current_meta_features[meta_key])
        if not similar_study_ids:
            return

        # 2. Collect data
        all_configs, all_scores = [], []
        for study_id in similar_study_ids:
            trials = self.db.get_study_trials(study_id)
            for trial in trials:
                if trial['status'] == 'completed':
                    all_configs.append(trial['parameters'])
                    # Assuming 'loss' is the metric, and we want to minimize it
                    all_scores.append(trial['metrics'].get('loss', 1e6))

        if not all_configs:
            return

        # 3. Train MetaLearner
        config_vectors = self.cs.to_array(all_configs)
        # For simplicity, we'll use a constant meta-vector since we filtered by one key.
        # A more complex system would use a vector of all meta-features.
        meta_vectors = np.array([[current_meta_features[meta_key]]] * len(all_configs))
        self.meta_learner.fit(meta_vectors, config_vectors, np.array(all_scores))

    def rank_candidates(self, candidates: List[Dict[str, Any]], current_meta_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.meta_learner.trained:
            return candidates  # No warm-start info, return as is

        candidate_vectors = self.cs.to_array(candidates)
        # Use a constant meta-vector for prediction
        meta_vector = np.array([[list(current_meta_features.values())[0]]])

        predicted_scores = self.meta_learner.predict(meta_vector, candidate_vectors)

        # Sort candidates by predicted score (lower is better)
        sorted_indices = np.argsort(predicted_scores)
        return [candidates[i] for i in sorted_indices]