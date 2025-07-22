import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
from math import atan2, degrees
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


RED = (200, 50, 50)
GREEN = (0, 200, 100)
BLUE = (35, 100, 255)
GRAY = (150, 150, 150)
ORANGE = (255, 165, 30)
BLACK = (0, 0, 0)

def bold(text):
    return "\033[1m{}\033[0m".format(text)

def underline(text):
    return "\033[4m{}\033[0m".format(text)

def colored(color, text):
    return "\033[38;2;{};{};{}m{}\033[0m".format(color[0], color[1], color[2], bold(text))


def print_color_line(STR, probs, green_list_i, blue_item=""):
    str_to_print = STR
    green_list = np.array([1 if i in green_list_i else 0 for i in range(len(probs))])
    max_index_search_arr = probs.copy()
    max_index_search_arr[green_list != 0] = -1
    max_index = np.argmax(max_index_search_arr)
    del max_index_search_arr
    for i,item in enumerate(probs):
        if i == blue_item:
            str_to_print += colored(BLUE, '{val:>5} '.format(val=round(item, 3)))
        elif i == max_index:
            str_to_print += colored(ORANGE, '{val:>5} '.format(val=round(item, 3)))
        elif green_list[i]!=0:
            str_to_print += colored(GREEN, '{val:>5} '.format(val=round(item, 3)))
        else:
            str_to_print += '{val:>5} '.format(val=round(item, 3))
    print(str_to_print)

def print_inter_mat(R, removed_interactions):
    for i, row in enumerate(R):
        # Print each element in the row
        print(bold(f" {i}\t| "), end=" ")
        for j, val in enumerate(row):
            if j == removed_interactions[i]:
                print(colored(BLUE, val), end=" ")
            elif val == 1:
                print(colored(GREEN, val), end=" ")
            else:
                print(val, end=" ")
        # Print the removed item index in blue at the end of the row
        print()


def plot_list(list, xlabel, ylabel, title):
    plt.plot(list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip - 1] + (ydata[ip] - ydata[ip - 1]) * (x - xdata[ip - 1]) / (xdata[ip] - xdata[ip - 1])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip - 1]
        dy = ydata[ip] - ydata[ip - 1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, fontweight='bold', rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):
    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines) + 2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)





def plot_HRK_simple(hrk_list, top_K = 5, save_plot_dir = ""):
    # Create a figure and axis
    max_y_val = 0
    plt.figure(figsize=(8, 6))

    color_dict = {"RAND": "lightgray", "POP": "orange", "MF": "#6A4C93", "QRS": "#4A90E2", "QRS_init": "darkgray", "QRS_no_ent": "brown"}
    for HRK, title, interesting_plt in hrk_list:
        if not interesting_plt: continue
        if 'QRS' in title: title='QRS'

        if max_y_val < max(HRK[:top_K+1]): max_y_val = max(HRK[:top_K+1])
        plt.plot(HRK[:top_K+1], label=title, color=color_dict[title], linewidth=7)

    plt.yticks(np.arange(0, max_y_val + 0.1, 0.1))  # From 0 to 1.0, with a step of 0.1

    # Add labels and a title
    plt.xlabel('K Value',fontsize=20)
    plt.ylabel('HR@K',fontsize=20)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[2], handles[3], handles[1], handles[0]], ['MF', 'VQRS', 'POP', 'RAND'],fontsize=20)
    # plt.legend([handles[2], handles[0], handles[1]], ['VQRS', 'RAND', 'VQRS no entanglers'],fontsize=20)
    # plt.legend(fontsize=20)

    plt.autoscale(axis='y', tight=True)
    plt.yticks(np.arange(0, max_y_val+0.1, 0.1),fontsize=20)  # From 0 to 1.0, with a step of 0.1
    plt.xlim([0, top_K])
    plt.xticks(range(0, top_K+1),fontsize=20)
    plt.grid(alpha=0.5)
    if save_plot_dir != "": plt.savefig(save_plot_dir)  # Change the path and filename as needed
    plt.show()



def plot_HRK_shoots_simple(hrk_list, top_K = 5, x_line_log=-1, x_line_sqrt=-1, save_plot_dir = ""):
    max_y_val = 0
    plt.figure(figsize=(10, 6))

    color_dict = {"RAND": "mediumseagreen", "POP": "orange", "MF": "brown", "QRS": ["lightsteelblue", "cornflowerblue", "royalblue", "slategrey"]}
    for HRK, name, interesting_plt in hrk_list:
        if not interesting_plt: continue
        if name == 'QRS_inf': label ='QRS infinte shots'
        if name == 'QRS_items_num': label = 'QRS M shots'
        if name == 'QRS_sqrt': label = 'QRS \"sqrt(M)\" shots'
        if name == 'QRS_log2': label = 'QRS \"log2(M)\" shots'
        if 'QRS' in name: color = color_dict['QRS'].pop()
        else:
            color = color_dict[name]
            label = name

        if max_y_val < max(HRK[:top_K+1]): max_y_val = max(HRK[:top_K+1])
        plt.plot(HRK[:top_K+1], label=label, color=color)

    plt.yticks(np.arange(0, max_y_val + 0.1, 0.1))  # From 0 to 1.0, with a step of 0.1

    if x_line_sqrt!= -1: plt.axvline(x=x_line_log, color='#DDA0DD', linestyle='--', label='QRS \"log2\" HR lim', linewidth=2)
    if x_line_sqrt!= -1: plt.axvline(x=x_line_sqrt, color='#9400D3', linestyle='--', label='QRS \"sqrt\" HR lim', linewidth=2)
    # Add labels and a title
    plt.xlabel('K Value')
    plt.ylabel('Hit Ratio')

    # Show the legend
    plt.legend()
    plt.autoscale(axis='y', tight=True)
    plt.yticks(np.arange(0, max_y_val+0.1, 0.1))  # From 0 to 1.0, with a step of 0.1
    plt.xlim([0, top_K])
    plt.xticks(range(0,top_K+1))
    plt.grid(alpha=0.5)
    if save_plot_dir != "": plt.savefig(save_plot_dir)  # Change the path and filename as needed
    plt.show()



def plot_HRK_shots_bars(hrk_list, top_K = 5, save_plot_dir = ""):
    hr10_shots_value = []
    for HR, name, interesting_plot in hrk_list:
        if not interesting_plot: continue
        if name == 'QRS_inf': name ='QRS infinte shots'
        if name == 'QRS_items_num': name = 'QRS M shots'
        if name == 'QRS_sqrt': name = 'QRS sqrt(M) shots'
        if name == 'QRS_log2': name = 'QRS log2(M) shots'
        hr10_shots_value.append((HR[top_K], name))


    # Extract the values and labels
    values = [item[0] for item in hr10_shots_value]
    labels = [item[1] for item in hr10_shots_value]

    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Create the bar plot
    bars = plt.bar(labels, values, color='royalblue')

    # Set the x and y limits to start from 0
    plt.xlim(-0.5, len(labels) - 0.5)  # Ensure bars don't go outside the figure
    plt.ylim(0, max(values) + 0.1)  # Slightly increase y limit for better spacing

    # Add labels and a title
    plt.ylabel('HR@10 score', fontsize=12)

    # Position x-axis labels at the center of the bars
    plt.xticks(range(len(labels)), labels, ha='center', fontsize=10)

    for bar in bars:
        yval = bar.get_height()  # Get the height of the bar
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,  # Positioning the text slightly above the bar
                 f'{yval:.4f}', ha='center', va='bottom', fontsize=10, color='black')

    # Show the grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()  # To prevent clipping of labels
    if save_plot_dir != "": plt.savefig(save_plot_dir)  # Change the path and filename as needed
    plt.show()


def plot_HRK(hrk_list, plot_title, save_plot_dir = ""):
    max_y_val = 0
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot()
    ax.set_xlabel('K value', fontsize=14)
    ax.set_ylabel('HR@K', fontsize=14)

    going_up_reds = ["salmon", "lightcoral", "indianred", "brown"]
    going_up_blues = ["lightblue", "lightsteelblue", "cornflowerblue", "royalblue", "slategrey"]
    color_dict = {"RAND": ["mediumseagreen"], "POP": ["orange"], "MF": ["brown"], "QRS": ["royalblue", "cornflowerblue", "slategrey", "steelblue", "lightsteelblue", "royalblue", "cornflowerblue", "slategrey", "steelblue", "lightsteelblue"]}
    # color_dict = {"RAND": ["mediumseagreen"], "POP": ["orange"], "MF": ["brown"],
    #               "QRS":  going_up_blues+going_up_reds}


    first = 1
    xvals = []
    # printing not interesting plots
    color, linestyle, linewidth = 'lightgrey', '--', 1
    for HRK, title, interesting_plt in hrk_list:
        if interesting_plt: continue
        if first:
            xvals.append(0.5)
            first = 0
        else:
            xvals.append(xvals[-1]+0.8)
        if max_y_val < max(HRK): max_y_val = max(HRK)
        ax.plot(HRK, label=title, color=color, linestyle=linestyle, linewidth=linewidth)

    # second plot the interesting plots
    first = 1
    for HRK, title, interesting_plt in hrk_list:
        if not interesting_plt: continue
        if first:
            xvals.append(9.5)
            first = 0
        else:
            xvals.append(xvals[-1] - 1)
        for key in color_dict.keys():
            if title.startswith(key):
                color = color_dict[key].pop(0)
                linestyle = '-'
                linewidth = '2'
        if max_y_val < max(HRK): max_y_val = max(HRK)
        ax.plot(HRK, label=title, color=color, linestyle=linestyle, linewidth=linewidth)

    labelLines(ax.get_lines(), xvals=xvals, align=False, fontsize=14)

    plt.autoscale(axis='y', tight=True)
    plt.yticks(np.arange(0, max_y_val+0.1, 0.1))  # From 0 to 1.0, with a step of 0.1
    plt.xlim([0, 10])
    plt.xticks(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    plt.grid(alpha=0.5)

    if save_plot_dir != "": plt.savefig(save_plot_dir)  # Change the path and filename as needed

    plt.show()


def plot_heatmap(matrix_list_of_lists, plot_title, x_labels, y_labels, save_plot_dir = ""):
    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(matrix_list_of_lists, columns=x_labels, index=y_labels)

    # Create the heatmap
    plt.figure(figsize=(15, 10))
    heatmap = sns.heatmap(df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)

    # Add titles and labels
    plt.title(plot_title)
    plt.xlabel('Epoch Num')
    plt.ylabel('Layers Num')

    # Show the plot
    if save_plot_dir != "": plt.savefig(save_plot_dir)  # Change the path and filename as needed

    plt.show()


def calculate_norms(vectors):
    num_vectors = len(vectors)
    norms = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            norms[i, j] = np.sum(np.abs(vectors[i] - vectors[j])**2)
    norms = (norms/np.max(norms))*2
    print_ascending_indices(norms)
    return norms

def print_ascending_indices(norms):
    for user, sublist in enumerate(norms):
        indices = np.argsort(sublist)
        print(f"user {user} - {indices}")


def apply_pca_and_plot(lists_of_vectors, lists_of_titles):
    colors = [('red', 60, 'x'), ('green', 60, 'x'), ('orange', 60, 'x'), ('purple', 45, 'x'), ('purple', 45, 'x'), ('purple', 45, 'x'), ('purple', 45, 'x'), ('purple', 45, 'x'), ('purple', 45, 'x')]

    fig, axs = plt.subplots(1,len(lists_of_vectors)-1, figsize=((len(lists_of_vectors)-1)*6, 6))
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing here
    pca = PCA(n_components=2)

    src_pca = pca.fit_transform(np.array(lists_of_vectors[0]))
    src_x, src_y = src_pca[:, 0], src_pca[:, 1]
    src_x, src_y = src_x - min(src_x), src_y - min(src_y)
    src_x, src_y = src_x / max(src_x), src_y / max(src_y)

    for list_num, (vecs_list, list_title) in enumerate(zip(lists_of_vectors[1:],lists_of_titles[1:])):

        data = np.array(vecs_list)
        reduced_data = pca.fit_transform(data)

        x,y = reduced_data[:,0], reduced_data[:,1]
        x,y = x - min(x), y - min(y)
        x,y = x/max(x), y/max(y)

        label = "Users' latent vectors" if list_num == 0 else ""
        if len(lists_of_vectors)-1 > 1:
            axs[list_num].scatter(src_x, src_y, c='blue', s=45, marker='o', label=label)
            axs[list_num].scatter(x,y, c=colors[list_num][0], s=colors[list_num][1], marker=colors[list_num][2], label='')
            axs[list_num].set_xlabel(list_title, fontsize=14)
            for i, txt in enumerate(range(len(x))):
                axs[list_num].plot([src_x[i], x[i]], [src_y[i], y[i]], color='lightblue', linestyle='--',
                                   alpha=0.7)
            fig.legend(loc='upper left', bbox_to_anchor=(0.00, 0.89), fontsize=14)
        else:
            axs.scatter(src_x, src_y, c='blue', s=80, marker='o', label=label, linewidths=3)
            axs.scatter(x,y, c=colors[list_num][0], s=colors[list_num][1], marker=colors[list_num][2], label=list_title, linewidths=3)
            axs.set_xlabel('PCA component 1', fontsize=14)
            axs.set_ylabel('PCA component 2', fontsize=14)
            plt.legend(loc='upper right', fontsize=14)
            for i, txt in enumerate(range(len(x))):
                axs.plot([src_x[i], x[i]], [src_y[i], y[i]], color='lightblue', linestyle='--',
                                   alpha=0.7)

    plt.show()


def create_R_from_all_interactions(all_interaction, users_num, items_num):
    R = np.zeros((users_num, items_num))
    for user, item in enumerate(all_interaction):
        R[user, item] = 1  # or another value to indicate interaction
    return np.array(R)
