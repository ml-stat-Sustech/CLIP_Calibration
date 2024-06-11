import os
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_reliability_diagram(preds, confs, labels, n_bins = 15, title = None, save_dir=None):


    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(confs, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(n_bins):

        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            accuracy = np.mean(labels[in_bin] == preds[in_bin])
            mean_confidence = np.mean(confs[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)


    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)


    weights = np.histogram(confs, bins)[0] / len(confs)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))

    #plot
    delta = 1.0/n_bins
    x = np.arange(0,1,delta)
    mid = np.linspace(delta/2,1-delta/2,n_bins)
    error = np.abs(np.subtract(mid,bin_acc))

    plt.rcParams["font.family"] = "serif"
    #size and axis limits
    plt.figure(figsize=(6,6))
    plt.xlim(0,1)
    plt.ylim(0,1)
    #plot grid
    plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
    #plot bars and identity line
    plt.bar(x, bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
    plt.bar(x, error, bottom=np.minimum(bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
    ident = [0.0, 1.0]
    plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
    #labels and legend
    plt.ylabel('Accuracy',fontsize=13)
    plt.xlabel('Confidence',fontsize=13)
    plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
    plt.text(0.025, 0.85, f'ECE: {ece*100:.2f}%',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round, pad=0.5', facecolor='wheat', edgecolor='orange'))


    if title is not None:
        plt.title(title,fontsize=16)
    plt.tight_layout()

    # save ece fig
    plt.savefig(save_dir)


    return plt


def plot_proximity_conf(proximity, conf, save_dir, sort_by_conf=True):
    """
    Plots proximity and conf lists as line plots. If sort_by_conf is True, the plot is based on sorted conf values.
    Otherwise, it uses the original order.

    :param proximity: List of proximity values
    :param conf: List of conf values
    :param save_dir: Directory to save the plot
    :param sort_by_conf: Whether to sort by conf values or not
    """

    # Check if the lists have the same length
    if len(proximity) != len(conf):
        raise ValueError("proximity and conf lists must have the same length!")

    if sort_by_conf:
        # Sort by conf and get sorted indices
        sorted_indices = sorted(range(len(proximity)), key=lambda k: proximity[k])

        # Rearrange proximity and conf based on sorted indices
        proximity = [proximity[i] for i in sorted_indices]
        conf = [conf[i] for i in sorted_indices]

    # Check if save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(proximity, label='Proximity', color='blue')
    plt.plot(conf, label='Conf', color='red')
    plt.legend()
    title = 'Proximity and Conf Plot (Sorted by Conf)' if sort_by_conf else 'Proximity and Conf Plot'
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Save plot
    filename = 'proximity_conf_plot_sorted.png' if sort_by_conf else 'proximity_conf_plot.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()





def compute_ece(probs, true_labels, num_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(probs >= bin_lower, probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_acc = accuracy_score(true_labels[in_bin], np.round(probs[in_bin]))
            bin_conf = np.mean(probs[in_bin])
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin

    return ece

def plot_proximity_acc_ece(proximity, pred, label, conf, save_dir):
    # Ensure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Sort by proximity
    sorted_indices = np.argsort(proximity)
    pred = pred[sorted_indices]
    label = label[sorted_indices]
    conf = conf[sorted_indices]
    proximity = proximity[sorted_indices]

    # Divide data into 10 bins
    num_samples = len(proximity)
    bin_size = num_samples // 10

    metrics = defaultdict(list)
    average_proximities = []

    for i in range(10):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i != 9 else num_samples

        bin_pred = pred[start_idx:end_idx]
        bin_label = label[start_idx:end_idx]
        bin_conf = conf[start_idx:end_idx]
        bin_proximity = proximity[start_idx:end_idx]

        # Calculate accuracy and ECE for this bin
        acc = accuracy_score(bin_label, bin_pred)
        plt, ece = reliability_diagram(bin_pred, bin_conf, bin_label)
        avg_proximity = np.mean(bin_proximity)

        metrics['accuracy'].append(acc)
        metrics['ece'].append(ece)
        metrics['conf'].append(np.mean(bin_conf))
        average_proximities.append(avg_proximity)
        plt.savefig(os.path.join(save_dir, str(i) + '_.png'))

    # Calculate average values for accuracy and ECE
    avg_acc = np.mean(metrics['accuracy'])
    avg_ece = np.mean(metrics['ece'])
    avg_conf = np.mean(conf)

    # Initialize the subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Accuracy on the first subplot
    axs[0].plot(average_proximities, [x*100 for x in metrics['accuracy']], marker='o', label='Accuracy', color='b')
    # axs[0].set_xlabel('Average Proximity')
    axs[0].set_xlabel('Average Proximity')
    axs[0].set_ylabel('Accuracy (%)')  # Updated label to show percentage
    axs[0].set_title('Accuracy')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xticks(average_proximities)
    axs[0].set_xticklabels([f"{avg:.2f}" for avg in average_proximities])
    axs[0].text(0.05, 0.95, f'Avg: {avg_acc*100:.2f}%', transform=axs[0].transAxes, verticalalignment='top')  # Display average value

    # Plot Confidence on the second subplot (swap with ECE)
    axs[1].plot(average_proximities, [x*100 for x in metrics['conf']], marker='x', label='Confidence', color='r')  # Swap label and color
    # axs[1].set_xlabel('Average Proximity')
    axs[1].set_xlabel('Average Proximity')
    axs[1].set_ylabel('Confidence  (%)')  # Updated label to show percentage
    axs[1].set_title('Confidence')  # Updated title
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xticks(average_proximities)
    axs[1].set_xticklabels([f"{avg:.2f}" for avg in average_proximities])
    axs[1].text(0.05, 0.95, f'Avg: {avg_conf*100:.2f}%', transform=axs[1].transAxes, verticalalignment='top')  # Display average value

    # Plot ECE on the third subplot (swap with Confidence)
    axs[2].plot(average_proximities, [x*100 for x in metrics['ece']], marker='x', label='ECE', color='g')  # Swap label and color
    # axs[2].set_xlabel('Average Proximity')
    axs[2].set_xlabel('Average Proximity')
    axs[2].set_ylabel('ECE (%)')  # Updated label to show percentage
    axs[2].set_title('ECE')  # Updated title
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xticks(average_proximities)
    axs[2].set_xticklabels([f"{avg:.2f}" for avg in average_proximities])
    axs[2].text(0.05, 0.95, f'Avg: {avg_ece*100:.2f}%', transform=axs[2].transAxes, verticalalignment='top')  # Display average value
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'proximity_acc_ece.png'))




def reliability_diagram(preds, confs, labels, n_bins = 10, title = None):

    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(confs, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(n_bins):

        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:
            accuracy = np.mean(labels[in_bin] == preds[in_bin])
            mean_confidence = np.mean(confs[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)

    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)

    weights = np.histogram(confs, bins)[0] / len(confs)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))

    delta = 1.0/n_bins
    x = np.arange(0,1,delta)
    mid = np.linspace(delta/2,1-delta/2,n_bins)
    error = np.abs(np.subtract(mid,bin_acc))

    plt.rcParams["font.family"] = "serif"
    #size and axis limits
    plt.figure(figsize=(6,6))
    plt.xlim(0,1)
    plt.ylim(0,1)
    #plot grid
    plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
    #plot bars and identity line
    plt.bar(x, bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
    plt.bar(x, error, bottom=np.minimum(bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
    ident = [0.0, 1.0]
    plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
    #labels and legend
    plt.ylabel('Accuracy',fontsize=13)
    plt.xlabel('Confidence',fontsize=13)
    plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
    plt.text(0.025, 0.85, f'ECE: {ece*100:.2f}%',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round, pad=0.5', facecolor='wheat', edgecolor='orange'))


    if title is not None:
        plt.title(title,fontsize=16)
    plt.tight_layout()
    # plt.show()

    return plt, ece

