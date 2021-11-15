"""Script for plotting data.

On error bars:
https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.07-Error-Bars/
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser('plotting statst')
    parser.add_argument('read_path')
    parser.add_argument('write_path')
    parser.add_argument('--x_label', required=True)
    parser.add_argument('--y_label', required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.read_path)

    # Plotting elements
    labels = data['n']
    x_pos = np.arange(len(labels))
    errs = data['std_error']
    means = data['mean']

    # Plotting
    fig, ax = plt.subplots()

    ax.bar(x_pos, means,
           yerr=errs,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_xlabel(args.x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(args.y_label)
    ax.set_title(args.title)
    ax.yaxis.grid(True)

    fig.savefig(args.write_path, bbox_inches='tight')
