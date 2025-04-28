import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches


parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

num_seeds = 10
num_eval = 55
nn_architecture = "smart"

fronts_list = []
hv_list = np.zeros((num_seeds, num_eval))
crashes_list = np.zeros((num_seeds, num_eval))

for seed in range(1, 11):
    # Load the object from the file
    with open(f"results/aim_fair_{seed}_continuous_{nn_architecture}_front.pkl", "rb") as f:
        fronts_list.append(pickle.load(f))
    with open(f"results/aim_fair_{seed}_continuous_{nn_architecture}_hv.pkl", "rb") as f:
        hv_list[seed - 1] = pickle.load(f)
    with open(f"results/aim_fair_{seed}_continuous_{nn_architecture}_crashes.pkl", "rb") as f:
        crashes_list[seed - 1] = pickle.load(f)


def compute_and_plot_ema(data, alpha=0.2):
    """
    Computes the exponential moving average (EMA) of a given array and optionally plots it.

    Parameters:
        data (array-like): The input data to compute EMA on.
        alpha (float): The smoothing factor (0 < alpha ≤ 1). Smaller alpha means more smoothing.

    Returns:
        np.ndarray: The computed EMA array.
    """
    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be in the range (0, 1].")

    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]  # Initialize EMA with the first data point

    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema


# HYPERVOLUME TRAINING CURVE
mean_hv = np.mean(hv_list, axis=0)
std_hv = np.std(hv_list, axis=0)
hv_list_ema = compute_and_plot_ema(mean_hv)
# Plot the mean reward
plt.figure(figsize=(10, 6))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],  # Computer Modern Roman
    "mathtext.fontset": "cm",  # Computer Modern for math expressions
    'axes.unicode_minus' : False
})
plt.plot(np.arange(num_eval), hv_list_ema, color="#3b528b", linewidth=2)
# Plot the confidence interval (mean ± std deviation)
plt.fill_between(np.arange(num_eval), hv_list_ema - std_hv*1.96, hv_list_ema + std_hv*1.96,
                 color="#3b528b", alpha=0.1)
plt.xlabel('Evaluation run', fontsize=20)
plt.ylabel('Hypervolume', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
plt.tight_layout()
if args.save:
    plt.savefig(f"plots/hypervolume_{nn_architecture}.pdf", format='pdf')
    plt.savefig(f"plots/hypervolume_{nn_architecture}.png", format='png')
else:
    plt.show()
plt.close()

# NUMBER OF CRASHES TRAINING CURVE
mean_crashes = np.mean(crashes_list, axis=0)
std_crashes = np.std(crashes_list, axis=0)
crashes_list_ema = compute_and_plot_ema(mean_crashes)
# Plot the mean reward
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_eval), crashes_list_ema, color="#3b528b", linewidth=2)
# Plot the confidence interval (mean ± std deviation)
plt.fill_between(np.arange(num_eval), max((crashes_list_ema - std_crashes*1.96).all(), 0), crashes_list_ema +
                 std_crashes*1.96,
                 color="#3b528b", alpha=0.1)
plt.xlabel('Evaluation run', fontsize=20)
plt.ylabel('Number of crashes', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
plt.tight_layout()
if args.save:
    plt.savefig(f"plots/crashes_{nn_architecture}.pdf", format='pdf')
    plt.savefig(f"plots/crashes_{nn_architecture}.png", format='png')
else:
    plt.show()
plt.close()

# PARETO FRONT PLUS K-MEANS VELOCITY VS EMISSIONS
for seed in range(9, 10):
    best_num_sol = 0
    best_index = 0
    for i in range(len(fronts_list[seed])):
        if len(fronts_list[seed][i]) > best_num_sol and crashes_list[seed, i] <= 50:
            best_num_sol = len(fronts_list[seed][i])
            best_index = i

    pareto_front = np.array(fronts_list[seed][best_index])

    n_clusters = 3  # Number of categories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pareto_front)
    ordered_clusters = [clusters[len(clusters)//2], clusters[-1], clusters[0]]
    if ordered_clusters[1] + ordered_clusters[2] == 1:
        ordered_clusters[0] = 2
    elif ordered_clusters[1] + ordered_clusters[2] == 2:
        ordered_clusters[0] = 1
    else:
        ordered_clusters[0] = 0
    last, mid = 0, 0
    for i in range(len(clusters)):
        if clusters[i] == clusters[0]:
            last += 1
        elif clusters[i] == clusters[-1]:
            mid += 1

    pareto_vel, pareto_em = [], []
    for i in range(pareto_front.shape[0]):
        pareto_vel.append(pareto_front[i][0])
        pareto_em.append(pareto_front[i][1])

    for idx, em in enumerate(pareto_em):
        pareto_em[idx] = em * 50

    x_staircase = np.concatenate([[pareto_vel[0]], pareto_vel])
    y_staircase = np.concatenate([pareto_em, [pareto_em[-1]]])

    colors = ['#31688e', '#440154', '#35b779']
    # Create the figure and axis
    plt.figure(figsize=(8, 6))
    plt.step(x_staircase[-mid-1:], y_staircase[-mid-1:], where='post', color='#440154')
    plt.step(x_staircase[last:-mid], y_staircase[last:-mid], where='post', color='#31688e')
    plt.step(x_staircase[:last+1], y_staircase[:last+1], where='post', color='#35b779')
    for idx, i in enumerate(ordered_clusters):
        plt.scatter(
            pareto_front[clusters == i, 0], pareto_front[clusters == i, 1]*50, color=colors[idx]
        )

    blue_patch = mpatches.Patch(color='#440154', label='Emission saving solutions')
    red_patch = mpatches.Patch(color='#31688e', label='Balanced solutions')
    green_patch = mpatches.Patch(color='#35b779', label='Performance based solutions')
    plt.legend(handles=[blue_patch, red_patch, green_patch], fontsize=18)

    # Set plot labels and title
    plt.xlabel('Velocity [m/s]', fontsize=20)
    plt.ylabel(r'-CO$_2$ emissions [g/s]', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    plt.tight_layout()
    if args.save:
        plt.savefig(f"plots/pareto_{nn_architecture}.pdf", format='pdf')
        plt.savefig(f"plots/pareto_{nn_architecture}.png", format='png')
    else:
        plt.show()
    plt.close()

    # BOXPLOTS + FAIRNESS DISCUSSION
    with open(f"results/boxplot_speeds_{nn_architecture}_{seed+1}.pkl", "rb") as f:
        boxplot_speeds = pickle.load(f)
    with open(f"results/boxplot_emissions_{nn_architecture}_{seed+1}.pkl", "rb") as f:
        boxplot_emissions = pickle.load(f)
    with open(f"results/delta_fairness_{nn_architecture}_{seed+1}.pkl", "rb") as f:
        delta_fairness = pickle.load(f)

    indexed_bs = list(enumerate(boxplot_speeds))
    sorted_indexed_bs = sorted(indexed_bs, key=lambda x: np.median(x[1]))
    original_indexes_bs = [idx for idx, _ in sorted_indexed_bs]

    new_boxplot_emissions = []
    new_boxplot_speeds = []
    for i in original_indexes_bs:
        new_boxplot_emissions.append(boxplot_emissions[i])
        new_boxplot_speeds.append(boxplot_speeds[i])

    for i, b in enumerate(new_boxplot_emissions):
        for j, e in enumerate(b):
            new_boxplot_emissions[i][j] = e * 50

    new_delta_fairness_a = []
    for i in original_indexes_bs:
        new_delta_fairness_a.append(delta_fairness[i])
    x = np.arange(len(delta_fairness))
    y = new_delta_fairness_a

    coefficients = np.polyfit(x, y, deg=1)
    line = np.poly1d(coefficients)
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = line(x_smooth)
    y_fitted = line(x)
    residuals = y - y_fitted
    std_error = np.sqrt(np.sum(residuals ** 2) / (len(y) - len(coefficients)))
    confidence_interval = std_error
    y_upper = y_smooth + confidence_interval
    y_lower = y_smooth - confidence_interval
    fair_color = 'darkgreen'
    plt.figure(figsize=(8, 6))
    for i in range(len(y)):
        plt.scatter(x[i], y[i], label=f"Cluster {i + 1}", c=fair_color)
    plt.plot(x_smooth, y_smooth, ':', color=fair_color, alpha=0.7)
    plt.fill_between(
        x_smooth, y_lower, y_upper, color=fair_color, alpha=0.05
    )
    plt.xticks(range(len(x)), labels=[""]*len(x))
    plt.tick_params(labelsize=18)
    plt.xlabel(r"Pareto efficient solutions parametrized by $\omega$", fontsize=20)
    plt.ylabel(r"$\Delta_F$ [s]", fontsize=20)
    plt.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    plt.tight_layout()
    if args.save:
        plt.savefig(f"plots/fairness_{nn_architecture}.pdf", format='pdf')
        plt.savefig(f"plots/fairness_{nn_architecture}.png", format='png')
    else:
        plt.show()
    plt.close()

    _, ax = plt.subplots(figsize=(8, 6), dpi=100)
    bplot = ax.boxplot(new_boxplot_speeds, vert=True, patch_artist=True)
    # fill with colors
    colors = ['#440154'] * 10 + ['#31688e'] * 8 + ['#35b779'] * 7
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set_color('#fde725')
    ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    ax.set_xlabel(r"Pareto efficient solutions parametrized by $\omega$", fontsize=20)
    ax.set_ylabel("Velocity [m/s]", fontsize=20)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=18)
    blue_patch = mpatches.Patch(color='#440154', label='Emission saving solutions')
    red_patch = mpatches.Patch(color='#31688e', label=r'Balanced solutions')
    green_patch = mpatches.Patch(color='#35b779', label=r'Performance based solutions')
    plt.legend(handles=[blue_patch, red_patch, green_patch], fontsize=16)
    if args.save:
        plt.savefig(f"plots/boxplot_vel_{nn_architecture}.pdf", format='pdf')
        plt.savefig(f"plots/boxplot_vel_{nn_architecture}.png", format='png')
    else:
        plt.show()
    plt.close()

    _, ax = plt.subplots(figsize=(8, 6), dpi=100)
    bplot = ax.boxplot(new_boxplot_emissions, vert=True, patch_artist=True)
    # fill with colors
    colors = ['#440154'] * 10 + ['#31688e'] * 8 + ['#35b779'] * 7
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set_color('#fde725')
    ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    ax.set_xlabel(r"Pareto efficient solutions parametrized by $\omega$", fontsize=20)
    ax.set_ylabel(r"CO$_2$ emissions [g/s]", fontsize=20)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=18)
    blue_patch = mpatches.Patch(color='#440154', label='Emission saving solutions')
    red_patch = mpatches.Patch(color='#31688e', label=r'Balanced solutions')
    green_patch = mpatches.Patch(color='#35b779', label=r'Performance based solutions')
    plt.legend(handles=[blue_patch, red_patch, green_patch], fontsize=16)
    if args.save:
        plt.savefig(f"plots/boxplot_em_{nn_architecture}.pdf", format='pdf')
        plt.savefig(f"plots/boxplot_em_{nn_architecture}.png", format='png')
    else:
        plt.show()
    plt.close()
