"""
Author: Jared Frazier
Project: OLA 3
File: id3.py
Class: CSCI 4350
Instructor: Dr. Joshua Phillips
Description: Module for ID3 Decision Tree.

Ch. 18 AIMA 3ed: Learning from Examples
"""

import argparse
import numpy as np
import math


class Node:
    def __init__(
            self, attr_ixs,
            value=None,
            left_child=None, right_child=None, decision=None):
        """Define state for Node.

        :param attr_ix: The index corresponding to the attribute
            for a particular dataset. For example,
            if a row in a dataset is [0.2 0.12 0.5 0]
            where the last column is the label, then there are four
            potential attribute indices here. If the attribute index
            is -1, then the value corresponds to the decision
        :param value: Corresponds to the value passed to a node or
            the decision associated with a node.
        :param left_child: Pointer to left Node.
        :param right_child: Pointer to right Node.
        :param decision: Class label given a particular 
        """

        # Save args
        self.attr_ixs = attr_ixs
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.decision = decision


class ID3DecisionTree:
    """ID3 Decision Tree."""

    def __init__(self):
        """Define state for ID3DecisionTree.

        :param data: <class 'numpy.ndarray'>
        """

        # The root of the tree
        self.root = Node(attr_ix=None)

    def decision_tree_learning(self, data):
        """Helper function for ID3 decision tree learning.

        :param data: Contains the learning set and attribute sets.

        :return: None
        """
        self.root = self.__id3(data=data, node=self.root)

    def __id3(self, data, node):
        """Create the ID3DecisionTree.

        :param data:

        :return:
        """

        # Compute entropy of subset
        unique_labels, unique_labels_counts = np.unique(
            data[:, -1], return_counts=True)
        learning_set_entropy = self.__entropy(unique_labels_counts)

        # All labels are the same
        if learning_set_entropy == 0:
            node.attr_ix = -1
            node.value = unique_labels[0]
        else:
            pass

    def test_tree(self,):
        """Test the ID3DecisionTree.

        :param:

        :return:
        """
        pass

    # TODO: Remove
    def entropy(self, obj_counts):
        return self.__entropy(obj_counts)

    # TODO: Remove
    def expected_information(self, label_counts, attr_counts):
        return self.__expected_information(label_counts, attr_counts)

    # TODO: Remove
    def information_gain(self, entropy, expected_information):
        return self.__information_gain(entropy, expected_information)

    def matrix_of_attrs_belonging_to_labels(self, attr_label_arr):
        """Computes matrix matching p_i, n_i, ... for any number of values.

        The notation p_i and n_i is from
        https://hunch.net/~coms-4771/quinlan.pdf

        :param attr_label_arr: 2D <class 'numpy.ndarray'> where the first column
            is the attribute and the values that the attribute can take
            on while the second column is the labels (classification)
            for each value of the attribute.

        :return: <class 'list'> of <class 'list'>
        """

        # Compute the unique discrete values that the attributes can
        # take on
        unique_attr_values = np.unique(attr_label_arr[:, 0])

        # Compute the unique discrete values that the labels can
        # take on
        unique_label_values = np.unique(attr_label_arr[:, 1])

        # A matrix with 1 row for each discrete value the attribute
        # can take on and 1 column for each discrete value the
        # label can take on. This data structure holds the counts
        # of each attribute value that matches the discrete label value
        # Example,
        # ['overcast' 'rain' 'sunny'] for labels = {0, 1}
        # outlook [[0, 4], [2, 3], [3, 2]]
        # This means that the outlook attribute which can take on 3 discrete
        # values has 0 overcast days that are also labeled 0, while it has
        # 4 overcast days that are also labeled 1.
        attr_label_count_matrix = []

        # Iterate through each unique value of the attributes
        # and count the occurences for which the attribute value
        # is unique value AND the label is a unique value of that label
        for unique_attr_value in unique_attr_values:
            counts = []
            for unique_label_value in unique_label_values:

                # Boolean vector... elements are True where condition
                # holds, False otherwise
                match_vector = np.logical_and(
                    attr_label_arr[:, 0] == unique_attr_value,
                    attr_label_arr[:, 1] == unique_label_value)

                # Non-zero means the instances in which the condition is True
                match_count = np.count_nonzero(match_vector)

                # Append the value to the counts list
                counts.append(match_count)

            # Append the counts list to the parent matrix
            attr_label_count_matrix.append(counts)

        # Resulting matrix
        return attr_label_count_matrix

    def discretize_continuous_data(self, attr_label_arr, return_bins=False):
        """Converts continuous attributes to discrete valued array."""

        threshold_indices = self.__get_threshold_indices(attr_label_arr)
        discretized_arr, bins = self.__get_binned_arrs(
            attr_label_arr, threshold_indices, return_bins)

        if return_bins:
            return discretized_arr, bins
        else:
            return discretized_arr

    def __get_threshold_indices(self, attr_label_arr):
        """Returns list of indices where adjacent labels differ.

        Array should be sorted.
        """

        # Check sorted
        if attr_label_arr[0, 0] > attr_label_arr[1, 0]:
            raise ValueError(':param attr_label_arr: must be sorted')

        # For all rows except the last
        threshold_tup_indices = []
        for row in range(len(attr_label_arr)-1):

            # Get the current label and the next label
            cur_label = attr_label_arr[row, -1]
            next_label = attr_label_arr[row+1, -1]

            # If the labels don't equal each other, use this
            # as splitting information
            if cur_label != next_label:
                threshold_tup_indices.append((row, row+1))

        # Resulting indices
        return threshold_tup_indices

    def __get_binned_arrs(self, attr_label_arr, threshold_indices, return_bins):
        """Use thresholds to return list of data with different bins."""

        # Compute the bins
        bins = []
        for threshold_ix in threshold_indices:
            bin_ = (attr_label_arr[threshold_ix[0], 0]
                    + attr_label_arr[threshold_ix[1], 0]) / 2
            bins.append(bin_)

        # Discretize the data into binned arrs
        lst_of_discretized_arrs = []
        for b in bins:

            # Binary thresholds
            discretized_attr_label_arr = attr_label_arr.copy()
            for ix, row in enumerate(discretized_attr_label_arr):

                # attr > bin_ or can be framed as attr >= bin_
                if row[0] > b:
                    discretized_attr_label_arr[ix, 0] = 0
                else:
                    discretized_attr_label_arr[ix, 0] = 1

            lst_of_discretized_arrs.append(discretized_attr_label_arr)

        if return_bins:
            return lst_of_discretized_arrs, bins
        else:
            return lst_of_discretized_arrs

    def sort_arr(attr_label_arr):
        """Sorts the array and keeps labels lined up."""

        sorted_attr_indices = np.argsort(attr_label_arr[:, 0])
        attr_label_arr = attr_label_arr[sorted_attr_indices, :]
        return attr_label_arr

    def __information_gain(self, entropy, expected_information):
        """Calcluate information gain for split point.

        :param:
        :param:

        :return:
        """
        return entropy - expected_information

    def __entropy(self, obj_counts):
        """Computed entropy for message.

        This is the uncertainty in the message.

        For E(x) = x * log_2(x), the limit as x -> 0^+ [E(x)] = 0.

        :param obj_counts: <class 'list'> of <class 'int'> 
            List where each element is the
            number of objects belonging to a particular class. E.g.,
            if there are 3 label classes [0 1 2], the first
            element of obj_counts should be the number of objects belonging
            to class 0, the second element should be the number
            objects belonging to class 1, and the third element of
            should be the number of objects belonging to class 2.

        :return: E(a, b, c, ...)
        """

        return -sum([(
            obj_i / sum(obj_counts)) * math.log(obj_i / sum(obj_counts), 2)
            for obj_i in obj_counts if obj_i != 0])

    def __expected_information(self, label_counts, attr_counts):
        """Expected information required for tree with some attribute as node.

        :param label_counts: <class 'list'> of <class 'int'>
        :param attr_counts: <class 'list'> of <class 'list'> where the 
            number of rows is the number of discrete values
            that the attribute can take on while the number of columns
            is the number of values that the label can take on. For example,
            if an attribute 'outlook' = {overcast, rain, sunny} while
            the 'label' = {0, 1}, then the attr_counts matrix will be
            3 x 2, [0, 0] will be the number of overcast 'outlook's
            that also belong to the label 0.

        :return: E(A)
        """

        total_num_labels = sum(label_counts)
        return sum((sum(c_i) / total_num_labels) * self.__entropy(c_i)
                   for c_i in attr_counts)


if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='ID3 decision tree')
    parser.add_argument(
        'training_data',
        help='path to txt file with training data for ID3 decision tree.',
        type=str)
    parser.add_argument(
        'testing_data',
        help='path to txt file with testing data for ID3 decision tree.')
    args = parser.parse_args()

    # Tree instantiation

    # Train tree

    # Test tree

    # Output number of testing examples that are correct
