"""
Author: Jared Frazier
Project: OLA 3
File: id3.py
Class: CSCI 4350
Instructor: Dr. Joshua Phillips
Description: Module for ID3 Decision Tree.

Ch. 18 AIMA 3ed: Learning from Examples
"""

import logging
import datetime
import sys
import argparse
import numpy as np
import pandas as pd
import math


class Node:
    def __init__(
            self, value=None):
        """Define state for Node.

        :param value:
        """

        self.value = value

        self.attribute = None
        self.learning_set = None
        self.children = []
        self.decision = None

    def add_child(self, node):
        """Appends a child to the list of children of the node."""

        self.children.append(node)

    def get_children(self,):
        """Returns the list of children of the node."""

        return self.children

    def get_attribute(self,):
        """Returns the attribute of the node."""

        return self.attribute

    def get_value(self,):
        """Returns the value of the node."""

        return self.value

    def get_labels(self,) -> np.ndarray:
        """Returns the labels for all rows."""

        return self.learning_set[:, -1]

    def set_learning_set(self, learning_set):
        """Sets the learning set for node."""

        self.learning_set = learning_set

    def set_attribute(self, attribute):
        """Sets the attribute for node."""

        self.attribute = attribute

    def set_unanimous_decision(self):
        """Sets unambiguous decision for node."""

        self.decision = self.__get_unanimous_decision()

    def set_probabilistic_decision(self,):
        """Sets decision for node based on max probabilities of labels and ties."""

        self.decision = self.__get_probabilistic_decision()

    def is_root(self,) -> bool:
        """Returns bool for whether node is root node.

        A root node WILL have a learning_set, it COULD have a decision,
        it COULD have children, but it should NOT have any attribute
        or value associated with it.
        """

        return self.attribute is None and self.value is None

    def is_leaf(self,) -> bool:
        """Returns bool for whether node is leaf.

        A leaf node necessarily has no children associated with it.
        """

        return len(self.children) == 0

    def __get_unanimous_decision(self,):
        """Returns a single element of the label subset if all labels are equal."""

        if self.__all_labels_equal():
            return self.get_labels()[0]

        else:
            raise ValueError('All labels are NOT equal!')

    def __get_probabilistic_decision(self,):
        """Returns best possible decision for node."""
        return

    def __all_labels_equal(self,) -> bool:
        """Check to see if all labels in learning set are the same."""

        return np.all(self.get_labels() == self.get_labels()[0])

    def __repr__(self,):
        """Information about a node"""

        rep = f'{self.__class__} object at {hex(id(self))}:'
        rep += f' (attribute={self.attribute},'
        rep += f' decision={self.decision})'
        return rep


class ID3DecisionTree:
    """ID3 Decision Tree."""

    def __init__(self):
        """Define state for ID3DecisionTree."""

        # The root of the tree
        self.root = Node()

    def decision_tree_learning(self, learning_set: np.ndarray) -> None:
        """Helper function for ID3 decision tree learning.

        :param data: Contains the learning set and attribute sets.

        :return: None
        """

        self.__id3(learning_set=learning_set, node=self.root)

    def __id3(self, learning_set: np.ndarray, node: Node) -> None:
        """Create the ID3DecisionTree.

        :param learning_set: The set containing all variables 
            (features and labels). The learning_set must only have a 
            single label. The label MUST be the last element in a given
            row, i.e., the final (-1) column for all rows in the set.

        :return: A tree node.
        """

        # Sort the learning set if there is continuous data
        pass

        # Discretize the data -- >= instead of >
        pass

        # Add the learning set to the node
        node.set_learning_set(learning_set=learning_set)

        # Compute the counts of each category of the label
        label_counts = np.unique(node.get_labels(), return_counts=True)[-1]

        # Compute entropy of subset
        learning_set_entropy = self.__entropy(label_counts)

        # Entropy is 0 for data, therefore all records
        # have same value for categorical attribute
        # return a leaf node with decision attribute: attribute value
        # Meaning if the input to the decision tree is a variable
        # whose variable matches variable of a leaf node
        # then the leaf node yields its decision attribute
        if learning_set_entropy == 0:
            node.set_unanimous_decision()

            exit()

            # # Consider how this recurses... it can only return
            # # a node if the node is a leaf node..., this means
            # # if the root has children, then the root node
            # # will not be returned to the calling function,
            # # thus self.root will not be set...

            # return tree_node

        # If not 0, compute information gain
        # for each attribute and find attribute with max IG(S, A)
        # create child node of the root
        elif learning_set_entropy != 0:

            # List that holds the information gain of each feature
            info_gain_lst = []

            # For each feature in the learning set, find the number
            # of the feature that belongs to each of the label
            # categories, compute the information expected information
            # gain, compute the actual information gain, then
            # add that information gain to the list
            # TODO: Will have to use the discretized_learning_set
            # instead of the learning_set for the project
            for feature in range(learning_set.shape[1] - 1):

                # N x M matrix where N is the number of categories
                # of the feature and M is the number of categories
                # of the label.
                subset_counts = self.count_feature_categories_belonging_to_label_categories(
                    learning_set)

                expected_info_gain = self.__expected_information(
                    label_counts=label_counts, subset_counts=subset_counts)

                info_gain = self.__information_gain(
                    entropy=learning_set_entropy,
                    expected_information=expected_info_gain)

                info_gain_lst.append(info_gain)

            # Get the index of the highest information gain...
            # this corresponds to the feature with the highest
            # information gain
            # TODO: Tie breakers
            best_feature_ix = np.argmax(info_gain_lst)

            # Should set the attribute of the current node
            # to this best feature ix per the pseudocode
            # "Decision Tree Attribute for Root (Node) = A"
            # from https://en.wikipedia.org/wiki/ID3_algorithm#cite_note-1
            node.set_attribute(attribute=best_feature_ix)

            # LOGGING
            LOG.debug('\n')
            LOG.debug(best_feature_ix)

            # Get the categories corresponding to the attribute (feature)
            # that had the highest information gain
            best_feature_categories = np.unique(
                learning_set[:, best_feature_ix])

            # List of the best learning sets
            best_category_learning_sets = []

            # Iterate through each of the unique categories for the
            # best feature and get the learning subsets coresponding
            # to only that value of the feature...
            # e.g., For a feature such as
            # 'outlook = {overcast, sunny, rainy}', the learning subset
            # containing only one category ('overcast') of the feature
            # ('outlook') might look like the below array
            # array(
            #   [['overcast', 'hot', 'high', False, 'P'],
            #    ['overcast', 'cool', 'normal', True, 'P'],
            #    ['overcast', 'mild', 'high', True, 'P'],
            #    ['overcast', 'hot', 'normal', False, 'P']], dtype=object)
            for category in best_feature_categories:

                # Uses the indices of the rows for which the feature
                # category exists to extract the relevant subset (see eg.
                # if category = 'overcast')...
                # np.where() returns a tuple, the first element is the
                # row indices matching the condition. The second element
                # is not used since it will just be the column
                # corresponding to the feature.
                best_category_learning_set = learning_set[
                    np.where(learning_set == category)[0], :]

                # Append the subset to the list
                best_category_learning_sets.append(best_category_learning_set)

            # For each of the subsets, create a child node
            child_nodes = [Node()
                           for i in range(len(best_category_learning_sets))]

            # Add the child nodes to the current node and call id3
            for child_node, learning_subset in zip(child_nodes, best_category_learning_sets):

                LOG.debug(child_node)
                LOG.debug(learning_subset)
                LOG.debug('\n')

                # node.add_child(child_node)
                # self.__id3(learning_set=learning_subset, node=child_node)

            exit()

        # Returns nothing since the calling function passes the
        # root node by obj-ref
        return

    def test_tree(self,):
        """Test the ID3DecisionTree.

        :param:

        :return:
        """
        pass

    # TODO: Remove
    def entropy(self, obj_counts: list) -> float:
        return self.__entropy(obj_counts)

    # TODO: Remove
    def expected_information(self, label_counts: list, attr_counts: list[list]):
        return self.__expected_information(label_counts, attr_counts)

    # TODO: Remove
    def information_gain(self, entropy: float, expected_information: float) -> float:
        return self.__information_gain(entropy, expected_information)

    def count_feature_categories_belonging_to_label_categories(
            self,
            attr_label_arr: np.ndarray) -> list[list]:
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

            # Counts the subset matches of attr value and label
            counts = []

            # Iterate through each unique value of the labels
            for unique_label_value in unique_label_values:

                # Boolean vector... elements are True where condition
                # holds, False otherwise
                subset_vector = np.logical_and(
                    attr_label_arr[:, 0] == unique_attr_value,
                    attr_label_arr[:, 1] == unique_label_value)

                # Non-zero means the instances in which the condition is True
                subset_count = np.count_nonzero(subset_vector)

                # Append the value to the counts list
                counts.append(subset_count)

            # Append the counts list to the parent matrix
            attr_label_count_matrix.append(counts)

        # Resulting matrix
        return attr_label_count_matrix

    def discretize_continuous_data(
            self,
            attr_label_arr: np.ndarray,
            return_bins=False) -> list[np.ndarray] \
            or tuple[list[np.ndarray], list[tuple]]:
        """Converts continuous attributes to discrete valued array."""

        threshold_indices = self.__get_threshold_indices(attr_label_arr)
        discretized_arr, bins = self.__get_binned_arrs(
            attr_label_arr, threshold_indices, return_bins)

        if return_bins:
            return discretized_arr, bins
        else:
            return discretized_arr

    def __get_threshold_indices(self, attr_label_arr: np.ndarray) -> list[tuple]:
        """Returns list of indices where adjacent labels differ.

        Array should be sorted.
        """

        # Check sorted
        if attr_label_arr[0, 0] > attr_label_arr[1, 0]:
            raise ValueError(':param attr_label_arr: must be sorted')

        # For all rows except the last
        threshold_indices = []
        for row in range(len(attr_label_arr)-1):

            # Get the current label and the next label
            cur_label = attr_label_arr[row, -1]
            next_label = attr_label_arr[row+1, -1]

            # If the labels don't equal each other, use this
            # as splitting information
            if cur_label != next_label:
                threshold_indices.append((row, row+1))

        # Resulting indices
        return threshold_indices

    def __get_binned_arrs(
        self,
        attr_label_arr: np.ndarray,
        threshold_indices: list[list],
        return_bins: bool) -> list[np.ndarray] \
            or tuple[list[np.ndarray], list[tuple]]:
        """Use thresholds to return list of data with different bins."""

        # Compute the bins
        bins = []
        for threshold_ix in threshold_indices:

            # Extract the indices from the tuple
            ix, adj_ix = threshold_ix

            # The bin is the average of the two
            bin_ = (attr_label_arr[ix, 0] + attr_label_arr[adj_ix, 0]) / 2

            # Append to a list of potential bins
            bins.append(bin_)

        # Discretize the data into binned arrs
        lst_of_discretized_arrs = []
        for b in bins:

            # Binary thresholds
            discretized_attr_label_arr = attr_label_arr.copy()
            for ix, row in enumerate(discretized_attr_label_arr):

                # TODO: Change for specs
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

    def sort_arr(attr_label_arr: np.ndarray) -> np.ndarray:
        """Sorts the array and keeps labels lined up."""

        sorted_attr_indices = np.argsort(attr_label_arr[:, 0])
        attr_label_arr = attr_label_arr[sorted_attr_indices, :]
        return attr_label_arr

    def __information_gain(
            self,
            entropy: float,
            expected_information: float) -> float:
        """Calcluate information gain for split point.

        :param entropy:
        :param expected_information:

        :return:
        """
        return entropy - expected_information

    def __entropy(self, obj_counts: list[list]) -> float:
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

    def __expected_information(
            self,
            label_counts: list,
            subset_counts: list[list]) -> float:
        """Expected information required for tree with some attribute as node.

        :param label_counts: <class 'list'> of <class 'int'>
        :param subset_counts: <class 'list'> of <class 'list'> where the 
            number of rows is the number of discrete values
            that the attribute can take on while the number of columns
            is the number of values that the label can take on. For example,
            if an attribute 'outlook' = {overcast, rain, sunny} while
            the 'label' = {0, 1}, then the attr_counts matrix will be
            3 x 2, [0, 0] will be the number of overcast 'outlook's
            that also belong to the label 0.

        :return: E(A)
        """

        # c_i is a list where the elements are the counts of the
        # attribute value belonging to each class of the labels
        total_num_labels = sum(label_counts)
        return sum((sum(c_i) / total_num_labels) * self.__entropy(c_i)
                   for c_i in subset_counts)


if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='ID3 decision tree')
    parser.add_argument(
        'training_data',
        help='path to txt file with training data for ID3 decision tree.',
        type=str)
    parser.add_argument(
        '--testing_data',
        help='path to txt file with testing data for ID3 decision tree.')
    args = parser.parse_args()

    # TODO: Remove Logging
    global LOG
    LOG = logging.getLogger()
    LOG.setLevel(logging.DEBUG)
    dtime_lst = str(datetime.datetime.now()).split(' ')
    dtime = dtime_lst[0].replace('-', '') + '_' + \
        dtime_lst[1].replace(':', '-')[: dtime_lst[1].find('.')]
    file_out = logging.FileHandler('./logs/' + dtime + '.log')
    stdout = logging.StreamHandler(sys.stdout)
    LOG.addHandler(file_out)
    LOG.addHandler(stdout)

    # Tree instantiation
    tree = ID3DecisionTree()

    # Load learning data
    learning_set = pd.read_excel(args.training_data).to_numpy()

    LOG.debug('\nIn __main__')
    LOG.debug(learning_set)

    # Load testing data
    testing_set = None

    # Train tree
    tree.decision_tree_learning(learning_set=learning_set)

    # Test tree

    # Output number of testing examples that are correct
