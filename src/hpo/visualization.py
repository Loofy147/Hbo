import matplotlib.pyplot as plt

def plot_optimization_history(study, save_path=None):
    """
    Plots the optimization history of a study.

    This function generates a set of plots to visualize the optimization process,
    including the progression of trial values, the distribution of values, and
    trial durations.

    Args:
        study (Study): The completed study object to visualize.
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    complete_trials = [t for t in study.trials if t.state == 'COMPLETE']
    if len(complete_trials) < 2:
        print("Not enough completed trials to generate plots.")
        return

    values = [t.value for t in complete_trials]
    trial_numbers = list(range(len(values)))

    # Calculate cumulative best value
    best_values = []
    current_best = values[0]
    for value in values:
        if study.direction == 'maximize':
            current_best = max(current_best, value)
        else: # minimize
            current_best = min(current_best, value)
        best_values.append(current_best)

    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Optimization History for '{study.study_name}'", fontsize=16)

    # Plot 1: Optimization History
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(trial_numbers, values, 'o', alpha=0.5, markersize=4, label='Trial Values')
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2.5, label='Best Value')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # Plot 2: Value Distribution
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(values, bins=min(15, len(values)//2 + 1), alpha=0.75, edgecolor='black')
    best_val = study.best_trial.value
    ax2.axvline(best_val, color='red', linestyle='--', linewidth=2, label=f'Best Value: {best_val:.4f}')
    ax2.set_xlabel('Objective Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Value Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    # Plot 3: Trial Durations
    ax3 = plt.subplot(2, 2, 3)
    durations = [t.duration for t in complete_trials]
    ax3.plot(trial_numbers, durations, 'o-', alpha=0.7, color='green')
    ax3.set_xlabel('Trial Number')
    ax3.set_ylabel('Duration (seconds)')
    ax3.set_title('Trial Durations')
    ax3.grid(True, alpha=0.4)

    # Plot 4: Parameter Importance (if applicable)
    # This is a placeholder for a more advanced plot.
    # For now, we'll plot a simple scatter of the first two params vs. value.
    ax4 = plt.subplot(2, 2, 4)
    if len(study.search_space.params) >= 2:
        param_names = list(study.search_space.params.keys())
        p1_name, p2_name = param_names[0], param_names[1]
        p1_values = [t.params[p1_name] for t in complete_trials]
        p2_values = [t.params[p2_name] for t in complete_trials]

        sc = ax4.scatter(p1_values, p2_values, c=values, cmap='viridis', alpha=0.8)
        plt.colorbar(sc, ax=ax4, label='Objective Value')
        ax4.set_xlabel(p1_name)
        ax4.set_ylabel(p2_name)
        ax4.set_title(f'Parameter Relationship: {p1_name} vs {p2_name}')
    else:
        ax4.text(0.5, 0.5, 'Not enough parameters to plot relationships.',
                 horizontalalignment='center', verticalalignment='center')
        ax4.set_title('Parameter Relationship')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()