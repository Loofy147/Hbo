import os
import time
from hpo.core.study import Study
from hpo.core.parameter import SearchSpace

# Define a simple objective function
def objective(trial):
    x = trial.params['x']
    y = trial.params['y']
    return (x - 2) ** 2 + (y - 3) ** 2

# Define the search space
search_space = SearchSpace()
search_space.add_uniform('x', -5, 5)
search_space.add_uniform('y', -5, 5)

# Define the database file path
db_file = "hpo_resume_test.db"

# Clean up previous runs if the file exists
if os.path.exists(db_file):
    os.remove(db_file)

print("="*60)
print("🚀 PART 1: Running initial trials")
print("="*60)

# Create a new study
study = Study(
    study_name="resume-test",
    storage=f"sqlite:///{db_file}",
    direction="minimize",
    verbose=True
)

# Run the first batch of trials
study.optimize(objective, n_trials=5, search_space=search_space)

print("\n📋 Initial study summary:")
study.print_summary()
best_trial_part1 = study.best_trial

print(f"\n✅ Best trial from Part 1 is #{best_trial_part1.trial_id} with value {best_trial_part1.value:.4f}")

print("\n... Simulating a crash and restart ...\n")
time.sleep(2)

print("="*60)
print("🚀 PART 2: Resuming the study")
print("="*60)

# Create a new Study object connected to the same database
# This should automatically load the previous state
resumed_study = Study(
    study_name="resume-test",
    storage=f"sqlite:///{db_file}",
    direction="minimize", # Direction will be loaded from the existing study
    verbose=True
)

# Verify that the previous trials were loaded
trials_df = resumed_study.get_trials_dataframe()
print(f"✅ Resumed study. Found {len(trials_df)} existing trials.")

# Assert that the number of trials is correct
assert len(trials_df) == 5, "Should have loaded the 5 trials from the first run."

print("\n🚀 Continuing optimization...")
resumed_study.optimize(objective, n_trials=5, search_space=search_space)

print("\n📋 Final study summary:")
resumed_study.print_summary()
best_trial_part2 = resumed_study.best_trial

print(f"\n✅ Best trial from Part 2 is #{best_trial_part2.trial_id} with value {best_trial_part2.value:.4f}")

# Final validation
final_df = resumed_study.get_trials_dataframe()
print(f"\n📊 Final validation: Total trials in DB = {len(final_df)}")
assert len(final_df) == 10, "Should have a total of 10 trials after resuming."

print("\n🎉 Persistence and resume functionality verified successfully!")

# Clean up the test database
os.remove(db_file)