import pytest
from hpo.pruners import MedianPruner, ASHAPruner

# --- Tests for MedianPruner ---

def test_median_pruner_no_pruning_on_startup():
    """
    Tests that the MedianPruner does not prune during the startup phase.
    """
    pruner = MedianPruner(n_startup_trials=5)
    # Only 3 completed trials, so pruning should be inactive
    completed_trials = [None] * 3
    should_prune = pruner.should_prune(trial_id=4, step=1, value=10, completed_trials=completed_trials)
    assert not should_prune

def test_median_pruner_no_pruning_on_warmup():
    """
    Tests that the MedianPruner does not prune during the warmup phase.
    """
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=10)
    completed_trials = [None] * 2
    # Step 5 is less than n_warmup_steps
    should_prune = pruner.should_prune(trial_id=3, step=5, value=10, completed_trials=completed_trials)
    assert not should_prune

def test_median_pruner_prunes_below_median():
    """
    Tests that the pruner correctly prunes a trial with a value below the median.
    """
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=1)
    completed_trials = [None] * 2

    # Report some values for step 1
    pruner.should_prune(trial_id=0, step=1, value=100, completed_trials=completed_trials)
    pruner.should_prune(trial_id=1, step=1, value=110, completed_trials=completed_trials)

    # The median at step 1 is 105. A new trial with value 90 should be pruned.
    should_prune = pruner.should_prune(trial_id=2, step=1, value=90, completed_trials=completed_trials)
    assert should_prune

def test_median_pruner_does_not_prune_above_median():
    """
    Tests that the pruner does not prune a trial with a value above the median.
    """
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=1)
    completed_trials = [None] * 2

    pruner.should_prune(trial_id=0, step=1, value=100, completed_trials=completed_trials)
    pruner.should_prune(trial_id=1, step=1, value=110, completed_trials=completed_trials)

    # A new trial with value 120 should NOT be pruned.
    should_prune = pruner.should_prune(trial_id=2, step=1, value=120, completed_trials=completed_trials)
    assert not should_prune

# --- Tests for ASHAPruner ---

@pytest.fixture
def asha_pruner():
    """A fixture for an ASHAPruner instance."""
    # Rungs will be at 1, 3, 9
    return ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)

def test_asha_pruner_rung_calculation(asha_pruner):
    """
    Tests that the rungs are calculated correctly.
    """
    assert asha_pruner.rungs == [1, 3, 9]

def test_asha_pruner_no_pruning_before_rung_is_full(asha_pruner):
    """
    Tests that ASHA does not prune until a rung has enough trials.
    """
    # Rung 1 needs 3 trials before pruning
    asha_pruner.should_prune(trial_id=0, step=1, value=10)
    should_prune = asha_pruner.should_prune(trial_id=1, step=1, value=5) # Should not prune yet
    assert not should_prune

def test_asha_pruner_prunes_worst_performer(asha_pruner):
    """
    Tests that ASHA prunes the worst-performing trial once a rung is full.
    """
    # Report three trials to fill the first rung (size=3)
    asha_pruner.should_prune(trial_id=0, step=1, value=100)
    asha_pruner.should_prune(trial_id=1, step=1, value=120)

    # The third trial has the worst performance and should be pruned.
    # The top performer is 120. With reduction_factor=3, the threshold is 120.
    # Let's re-evaluate with a clearer example.
    # Trials: 100, 120. Next is 90. Rung values: [100, 120, 90].
    # Sorted: [120, 100, 90]. Threshold index = 3 // 3 = 1. Threshold is sorted[0] = 120.
    # This seems wrong. Let's trace the logic.
    # threshold_index = len(rung_values) // reduction_factor = 3 // 3 = 1.
    # threshold = sorted_values[threshold_index - 1] = sorted_values[0] = 120.
    # `value < threshold` -> `90 < 120` -> True. Correct.

    # What if the new value is in the middle?
    # Rung values: [100, 120]. New trial is 110.
    asha_pruner_2 = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
    asha_pruner_2.should_prune(trial_id=0, step=1, value=100)
    asha_pruner_2.should_prune(trial_id=1, step=1, value=120)
    should_prune_middle = asha_pruner_2.should_prune(trial_id=2, step=1, value=110)
    # Rung values: [100, 120, 110]. Sorted: [120, 110, 100]. Threshold: 120.
    # `110 < 120` -> True. Correct.
    assert should_prune_middle

def test_asha_pruner_does_not_prune_top_performer(asha_pruner):
    """
    Tests that ASHA does not prune a top-performing trial.
    """
    asha_pruner.should_prune(trial_id=0, step=1, value=100)
    asha_pruner.should_prune(trial_id=1, step=1, value=90)

    # This trial should be the new best performer and not be pruned.
    should_prune = asha_pruner.should_prune(trial_id=2, step=1, value=110)
    # Rung values: [100, 90, 110]. Sorted: [110, 100, 90]. Threshold: 110.
    # `110 < 110` -> False. Correct.
    assert not should_prune