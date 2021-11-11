"""
Author: Jared Frazier
Project: OLA 3
File: id3.py
Class: CSCI 4350
Instructor: Dr. Joshua Phillips
Description: Module for ID3 Decision Tree.

Ch. 18 AIMA 3ed: Learning from Examples
"""

from __future__ import annotations
import logging
import datetime
import sys
import argparse
import numpy as np
import pandas as pd
import math
from copy import deepcopy

# TODO: Remove
from anytree import Node, RenderTree


class TreeNode:
    def __init__(self, category=None):
        """Define state for TreeNode.

        :param category: <class 'str'> or <class 'int'>
        Attributes consist of unique (discrete)
        categories. E.g., if self.attribute = outlook,
        then category for node is in {sunny, overcast, rain}. For
        cancer or iris dataset, the category will be the numpy interval.
        """

        self.attribute = None
        self.category = category
        self.learning_set = None
        self.children = []
        self.decision = None

    def add_child(self, node):
        """Appends a child to the list of children of the node."""

        self.children.append(node)

    def get_children(self,) -> list[TreeNode]:
        """Returns the list of children of the node."""

        return self.children

    def get_attribute(self,):
        """Returns the attribute of the node."""

        return self.attribute

    def get_category(self,):
        """Returns the value of the node."""

        return self.category

    def get_decision(self,):
        """Returns the decision of the node."""

        return self.decision

    def get_labels(self,) -> np.ndarray:
        """Returns the labels for all rows."""
        if len(self.learning_set.shape) == 2:
            return self.learning_set[:, -1]
        elif len(self.learning_set.shape) == 1:
            return self.learning_set[-1]

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

        return self.attribute is None and self.category is None

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
        rep += f' category={self.category},'
        rep += f' decision={self.decision})'
        return rep

    def name(self,):
        return f'[{self.attribute}:{self.category}:{self.decision}]'


class ID3DecisionTree:
    """ID3 Decision Tree."""

    def __init__(self):
        """Define state for ID3DecisionTree."""

        # The root of the tree
        self.root = TreeNode()

    # TODO: Remove
    def display_tree(self):
        anytree = self.convert_tree_to_anytree(self.root)
        for pre, fill, node in RenderTree(anytree):
            # pre = pre.encode(encoding='UTF-8', errors='strict')
            LOG.debug("%s%s" % (pre, node.name))

    # TODO: Remove
    def convert_tree_to_anytree(self, tree: TreeNode):
        anytree = Node(tree.name())
        self.attach_children(tree, anytree)
        return anytree

    # TODO: Remove
    # Attach the children from the decision tree into the anytree
    # tree format.
    def attach_children(self, parent_node: TreeNode, parent_anytree_node: Node):
        for child_node in parent_node.get_children():
            child_anytree_node = Node(
                child_node.name(), parent=parent_anytree_node)
            self.attach_children(child_node, child_anytree_node)

    def decision_tree_learning(self, learning_set: np.ndarray) -> None:
        """Helper function for ID3 decision tree learning.

        :param data: Contains the learning set and attribute sets.

        :return: None
        """

        self.__id3(learning_set=learning_set, node=self.root)

    def __id3(self,
              learning_set: np.ndarray,
              node: TreeNode,
              given=None) -> None:
        """Create the ID3DecisionTree.

        :param learning_set: The set containing all variables
            (features and labels). The learning_set must only have a
            single label. The label MUST be the last element in a given
            row, i.e., the final (-1) column for all rows in the set.
        :param node: <class 'TreeNode'>
        :param given: The attribute value in the learning subset
            which is to be ignored during information gain calculations.

        TODO: continuous: bool

        :return: None
        """

        # Learning is just a row vector??? data.shape -> len((,)) < 2
        # then you have only a row vector. np.array[learning_set]
        # converts row vector into column vector...
        # if len(learning_set.shape) < 2:
        #     data = np.array([learning_set])

        # Sort the learning set if there is continuous data
        # sorted_indices = np.argsort(learning_set)

        # Discretize the data -- >= instead of >
        pass

        # Want to keep the original learning set
        # this is because if you don't, then you will have issues
        # with dividing subsets based on categories...

        # Add the learning set to the node
        node.set_learning_set(learning_set=learning_set)

        # LOG
        LOG.debug('\nLabels:')
        LOG.debug(node.get_labels())

        # Compute the counts of each category of the label
        label_counts = np.unique(node.get_labels(), return_counts=True)[-1]

        # Log
        LOG.debug('Label counts:')
        LOG.debug(label_counts)

        # Compute entropy of subset
        learning_set_entropy = self.__entropy(label_counts)

        # LOGGING
        LOG.debug('\nLearning Set Entropy:' + str(learning_set_entropy))

        # Entropy is 0 for data, therefore all records
        # have same value for categorical attribute
        # return a leaf node with decision attribute: attribute value
        # Meaning if the input to the decision tree is a variable
        # whose variable matches variable of a leaf node
        # then the leaf node yields its decision attribute
        if learning_set_entropy == 0:

            LOG.debug('\n---------------------------------')
            LOG.debug('In `if learning_set_entropy == 0`')
            LOG.debug('setting unaminous decision for node...')
            node.set_unanimous_decision()
            LOG.debug('node ->')
            LOG.debug(node)
            LOG.debug('---------------------------------')

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
            # feature is assumed to be numeric here (i.e., column is numeric
            # and not str)
            for feature in range(learning_set.shape[1] - 1):

                if given is None or (given is not None and given != feature):
                    # N x M matrix where N is the number of categories
                    # of the feature and M is the number of categories
                    # of the label.
                    subset_counts = self.count_feature_categories_belonging_to_label_categories(
                        learning_set[:, [feature, -1]])

                    LOG.debug('\nsubset_counts')
                    for row in subset_counts:
                        LOG.debug(str(row))

                    expected_info_gain = self.__expected_information(
                        label_counts=label_counts, subset_counts=subset_counts)

                    info_gain = self.__information_gain(
                        entropy=learning_set_entropy,
                        expected_information=expected_info_gain)

                    info_gain_lst.append(info_gain)

                else:

                    # A 'given' category is ignored for the purposes
                    # of information gain computations...
                    # but the ix is still relevant
                    info_gain_lst.append(-np.inf)

            # Get the index of the highest information gain...
            # this corresponds to the feature with the highest
            # information gain
            # TODO: Tie breakers
            LOG.debug('\nInformation gain list:')
            LOG.debug(str(info_gain_lst))
            best_feature_ix = np.argmax(info_gain_lst)

            # Should set the attribute of the current node
            # to this best feature ix per the pseudocode
            # "Decision Tree Attribute for Root (TreeNode) = A"
            # from https://en.wikipedia.org/wiki/ID3_algorithm#cite_note-1
            node.set_attribute(attribute=best_feature_ix)

            # LOGGING
            LOG.debug('\n')
            LOG.debug(
                f'Best feature index (i.e., attribute):{best_feature_ix}')

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
                # corresponding to the feature....
                # the learning subset should include all other features
                best_category_learning_set = learning_set[
                    np.where(learning_set == category)[0], :]

                # Append the subset to the list
                best_category_learning_sets.append(best_category_learning_set)

            # For each of the subsets, create a child node with the
            # associated category...
            # this way children will have attribute: category
            # and when testing, the traversal will search each
            # child until the correct category is found....
            # e.g.,
            #           node(attribute=outlook, children=[addresses of child_1-3], category=None)
            #           /              |                   \ # Visit children until the target category
            #         node(           node(             node(
            # attribute=humidity,     attribute=None       attribute=wind
            # category=sunny)         category=overcast)   category=rain)
            child_nodes = [TreeNode(category=best_feature_categories[i])
                           for i in range(len(best_category_learning_sets))]

            # Add the child nodes to the current node and call id3
            for child_node, learning_subset in zip(child_nodes, best_category_learning_sets):

                LOG.debug('\n')
                LOG.debug(child_node)
                LOG.debug(
                    f'Learning subset given Feature `{best_feature_ix}`: Category `{child_node.get_category()}`')
                LOG.debug(learning_subset)
                LOG.debug('\n')

                # TODO: Deal with recursion issue and building
                # the tree...

                node.add_child(child_node)
                self.__id3(learning_set=learning_subset,
                           node=child_node, given=best_feature_ix)

        # Returns nothing since the calling function passes the
        # root node by obj-ref
        return

    def test_tree(self,):
        """Test the ID3DecisionTree.

        :param:

        :return:
        """
        pass

    def traverse_tree(
            self,
            row_vector: np.ndarray or list or tuple,  # or pd.DataFrame
            node: TreeNode) -> str or int or bool:
        """Recursive traversal of tree until a decision returned.

        :param row_vector: Vector of features.
        :param node: <class 'TreeNode'>

        :return: The decision for the row vector.
        """

        # Set attribute identifiers based on row vector type...
        # attribute identifiers will be indices if not dataframe, if dataframe
        # the attribute identifiers will be the column names...
        # an attribute identifier might be 'outlook' or it might just
        # be the index of a given column
        if isinstance(row_vector, np.ndarray) \
                or isinstance(row_vector, list) \
                or isinstance(row_vector, tuple):
            attributes = range(len(row_vector))

        # # TODO: Might be interesting to include processing of dataframe
        # # later...
        # elif isinstance(row_vector, pd.DataFrame):
        #     attributes = row_vector.columns
        # else:
        #     raise TypeError(
        #         ':param row_vector: must be array like or \
        #             or <class `pandas.DataFrame`>')

        # Recursive traversal -- at least one of the attributes
        # should be in the current nodes attribute
        LOG.debug(node)
        # if node.get_attribute() in attributes:

        # Consider a row vector...
        #  A_i = A1         A2          A3
        #       [A1_cat     A2_cat      A3_cat]
        # Each attribute has categories corresponding to it, and
        # indexing the i^th attribute gives the category (or value)
        # associated with that particular attribute
        category = row_vector[node.get_attribute()]
        LOG.debug(category)

        # Special case
        if node.is_root() and node.is_leaf():
            return node.get_decision()

        # All other cases where children are generated
        for child in node.get_children():

            # Base case
            if category == child.get_category() and child.is_leaf():

                LOG.debug('At Leaf:')
                LOG.debug(child)
                LOG.debug('Returning!!')
                return child.get_decision()

            # Recursive case
            elif category == child.get_category() and not child.is_leaf():
                return self.traverse_tree(row_vector=row_vector, node=child)

        # else:
        #     raise ValueError(
        #         ':param row_vector: does not have an attribute that matches'
        #         + ' the decision tree`s node')

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
        unique_label_values = np.unique(attr_label_arr[:, -1])

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
            # TODO: Is the copy here necessary???
            discretized_attr_label_arr = attr_label_arr.copy().astype(object)
            for ix, row in enumerate(discretized_attr_label_arr):

                # TODO: Change for specs
                # attr > bin_ or can be framed as attr >= bin_
                # b <= attr < inf
                if row[0] >= b:
                    discretized_attr_label_arr[ix, 0] = pd.Interval(
                        left=b, right=np.inf, closed='left')

                # -inf < attr < b
                else:
                    discretized_attr_label_arr[ix, 0] = pd.Interval(
                        left=-np.inf, right=b, closed='neither')

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

    def get_root(self,):
        """Gets the root of the tree."""

        return self.root


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

    print('------------------------')
    print('MUST REMOVE ANYTREE')
    print('------------------------')

    # TODO: Remove Logging
    global LOG
    LOG = logging.getLogger()
    LOG.setLevel(logging.DEBUG)
    dtime_lst = str(datetime.datetime.now()).split(' ')
    dtime = dtime_lst[0].replace('-', '') + '_' + \
        dtime_lst[1].replace(':', '-')[: dtime_lst[1].find('.')]
    file_out = logging.FileHandler('./logs/' + dtime + '.log', 'w', 'utf-8')
    stdout = logging.StreamHandler(sys.stdout)
    LOG.addHandler(file_out)
    LOG.addHandler(stdout)

    # Tree instantiation
    tree = ID3DecisionTree()

    # Load learning data
    data = pd.read_excel(args.training_data).to_numpy()
    testing_set = data
    learning_set = data[0, :]

    LOG.debug('\nIn __main__')
    LOG.debug(learning_set)

    # Train tree
    LOG.debug('\nTraining decision tree!')
    tree.decision_tree_learning(learning_set=learning_set)

    # TODO: Remove -- display tree
    tree.display_tree()

    # Test tree traversal
    LOG.debug('-----------------------')
    LOG.debug('\nTesting!!!')
    LOG.debug('-----------------------')

    if len(testing_set.shape) == 2:

        preds = np.empty(shape=(testing_set.shape[0], ), dtype=object)
        ix = 0

        for row_vector in testing_set:
            LOG.debug('\n' + str(row_vector[: -1]))
            pred = tree.traverse_tree(row_vector[: -1], tree.get_root())
            LOG.debug(pred)
            preds[ix] = pred
            ix += 1

        LOG.debug('Do the predictions and targets match?')
        LOG.debug(
            f'{np.count_nonzero(preds == testing_set[:, -1])} / {learning_set.shape[0]}')

    elif len(testing_set.shape) == 1:

        LOG.debug(str(testing_set))

        pred = tree.traverse_tree(
            row_vector=testing_set[: -1], node=tree.get_root())

        LOG.debug(f'Prediction: {pred} -- Target: {testing_set[-1]}')

    # Output number of testing examples that are correct
