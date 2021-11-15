"""Script for computing mean and std. error"""

import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='mean and std error script')
    parser.add_argument(
        'read_path',
        help='path to data.',
        type=str)
    parser.add_argument(
        'write_path',
        help='path to write the csv.',
        type=str)
    parser.add_argument(
        'n_lst',
        help='list of testing size samples n.',
        nargs='+',
        type=int)
    args = parser.parse_args()

    # Read the file
    print('Reading file...')
    with open(args.read_path, 'r') as fobj:
        lines = [float(line.strip()) for line in fobj.readlines()]

    # Instantiate data frame
    df = pd.DataFrame({'n': args.n_lst, 'mean': np.empty_like(
        args.n_lst), 'std_error': np.empty_like(args.n_lst)})

    # Get length of nargs for range increment reasons
    len_nargs = len(args.n_lst)

    # Compute incrementer
    incrementer = len(lines) // len_nargs

    # Get the appropriate partitions and calculate summary stats
    print('Building df...')

    mean_lst = []
    std_err_lst = []

    n_ix = 0
    print(len(lines))
    for partition in range(0, len(lines), incrementer):

        # log increment
        print(partition, partition + incrementer)

        # Get the line interval
        interval = lines[partition: partition + incrementer]

        # Compute stats
        mean_ = np.mean(interval)
        std_err = 1.96*(np.std(interval)/np.sqrt(100))

        # Update the dataframe
        mean_lst.append(round(mean_, 5))
        std_err_lst.append(round(std_err, 3))

    # Change values
    print(len(mean_lst))
    print(len(std_err_lst))
    print(len(args.n_lst))
    df['mean'] = mean_lst
    df['std_error'] = std_err_lst

    # Write the dataframe to file
    if not args.write_path.endswith('csv'):
        args.write_path = args.write_path + '.csv'

    # Writing file
    print('Writing file...')
    df.to_csv(args.write_path, index=False)
