#!usr/bin/python

"""
Author: Jared Frazier
Project: OLA 3
File: id3.py
Class: CSCI 4350
Instructor: Dr. Joshua Phillips
Description: Module for ID3 Decision Tree.

for n in 5 10 25 50 75 100 125 140 145 149; do for ((x=0; x<100; x++)); do echo "cat iris-data.txt | ./split.bash $n python id3.py --percentage True --precision 3"; done >> iris_out_$n.txt; done | parallelize.bash

?? iris
for n in 1 5 10 25 50 75 100 125 140 145 149; do for ((x=0; x<100; x++)); do echo "cat iris-data.txt | ./split.bash $n python id3.py --percentage True --precision 3"; done | ./parallelize.bash; done >> iris_out.txt

?? cancer
for n in 1 5 10 25 50 75 90 100 104; do for ((x=0; x<100; x++)); do echo "cat cancer-data.txt | ./split.bash $n python id3.py --percentage True --precision 3"; done | ./parallelize.bash; done >> cancer_out.txt

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
from distutils.util import strtobool

class TreeNode:
    def __init__(self, category=None,):
        """Define state for TreeNode.

        :param category: <class 'list'>
        Attributes consist of unique (discrete)
        categories. E.g., if self.attribute = outlook,
        then category for node is in {sunny, overcast, rain}. For
        cancer or iris dataset, the category will be the numpy interval.
        """

        self.category = category
        self.attribute = None
        self.learning_set = None
        self.discrete_feature_label_set = None
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

    def get_category(self,) -> str or int or pd.Interval:
        """Returns the value of the node."""

        return self.category

    def get_decision(self,):
        """Returns the decision of the node."""

        return self.decision

    def get_labels(self,) -> np.ndarray:
        """Returns the labels for all rows."""

        # Handling matrix or row vector [n, m] where n=1 for row vector
        return self.learning_set[:, -1]

    def get_features(self,) -> np.ndarray:
        """Returns the features for all rows."""

        return self.learning_set[:, :-1]

    def get_learning_set(self,) -> np.ndarray:
        """Returns the learning set"""

        return self.learning_set

    def get_discrete_feature_label_set(self,):
        """return the discrete feature label set."""

        return self.discrete_feature_label_set

    def set_learning_set(self, learning_set: np.ndarray) -> None:
        """Sets the learning set for node."""

        self.learning_set = learning_set

    def set_discrete_feature_label_set(
            self,
            feature_ix: int, categories: tuple[pd.Interval]) -> None:
        """Sets the discrete feature label set.

        Converts a single feature vector into a boolean array based
        on interval.
        """

        # Get the interval threshold (-np.inf, val) -> val
        threshold = categories[0].right

        # Extract the appropriate feature vector
        feature_vector = self.get_features()[:, feature_ix]

        # Make categorical feature vector (boolean)
        boolean_feature_vector = np.apply_along_axis(
            lambda ele: ele < threshold, axis=0, arr=feature_vector)

        # Make an empty numpy array which will be (n, 2)..
        # 2 columns since there is a single feature and the corresponding
        # label
        discrete_feature_label_set = np.empty(
            shape=(feature_vector.shape[0], 2))

        # Populate the array
        discrete_feature_label_set[:, 0] = boolean_feature_vector
        discrete_feature_label_set[:, 1] = self.get_labels()

        # Set the data member
        self.discrete_feature_label_set = discrete_feature_label_set

    def set_attribute(self, attribute) -> None:
        """Sets the attribute for node."""

        self.attribute = attribute

    def set_unanimous_decision(self) -> None:
        """Sets unambiguous decision for node."""

        self.decision = self.__get_unanimous_decision()

    def set_majority_decision(self,) -> None:
        """Sets decision for node based on majority of labels in learning set."""

        self.decision = self.__get_majority_decision()

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

    def all_features_equal(self,) -> bool:
        """Returns a bool for whether all attributes are the same.

        Compares all values of features in the learning set at all rows and
        a single value of the first feature of the first row.
        """

        return np.all(self.learning_set[:, :-1] == self.learning_set[0, 0])

    def name(self,):
        return f'[{self.attribute}:{self.category}:{self.decision}]'

    def __get_unanimous_decision(self,):
        """Returns a single element of the label subset if all labels are equal."""

        if self.__all_labels_equal():
            return self.get_labels()[0]

        else:
            raise ValueError('All labels are NOT equal!')

    def __get_majority_decision(self,):
        """Returns decision for node based on majority label."""

        labels, label_counts = np.unique(self.get_labels(), return_counts=True)
        majority_label = labels[np.argmax(label_counts)]
        return majority_label

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


class ID3DecisionTree:
    """ID3 Decision Tree."""

    def __init__(self):
        """Define state for ID3DecisionTree."""

        # The root of the tree
        self.root = TreeNode()

    def get_root(self,):
        """Gets the root of the tree."""

        return self.root

    def train(self, learning_set: np.ndarray,) -> None:
        """Train the decision tree on continuous data only."""

        if len(learning_set.shape) == 1:
            learning_set = np.expand_dims(learning_set, axis=0)

        self.__train(learning_set=learning_set, node=self.root)

    def predict(self, testing_set: np.ndarray) -> int or np.ndarray:
        """Prediction on 1D or 2D data"""

        # Convert testing set to row vector if 1D
        if len(testing_set.shape) == 1:
            return self.__predict(testing_set)

        else:
            preds = []
            for row in testing_set:
                pred = self.__predict(row_vector=row)
                preds.append(pred)
            return np.array(preds)

    def __predict(self, row_vector: np.ndarray) -> int:
        """Make prediction on 1D array of attributes (features)."""

        return self.traverse_tree(row_vector=row_vector,
                                  node=self.root)

    def __train(self,
                learning_set: np.ndarray,
                node: TreeNode,
                given: list = None,) -> None:
        """Private helper method for training decision tree on continous data."""

        # Add learning set into
        node.set_learning_set(learning_set=learning_set)

        # Compute entropy of learning set
        learning_set_entropy = self.__entropy(
            np.unique(node.get_labels(), return_counts=True)[-1])

        # LOG

        # Determine splitting or terminal node conditions
        if learning_set_entropy == 0:
            node.set_unanimous_decision()

        elif node.all_features_equal():
            node.set_majority_decision()

        else:

            # Set given if not existing
            if given is None:
                given = []

            # Get potential splits -- will have to consider the
            # attributes already given here so that they can
            # be ignored...
            # or (((interval11, interval12), (interval13, interval14), ),
            #      (interval21, interval22,), (interval23, interval24),
            #       )
            split_categories = self.__compute_split_categories(
                node=node, given=given)

            # Determine the split (category)
            information_gain_arr = self.__compute_category_information_gain(
                split_categories=split_categories,
                node=node,
                given=given,
                learning_set_entropy=learning_set_entropy)

            # LOG THE INFORMATION GAIN ARRAY

            # if the data is continuous, you will have to flatten it
            if self.__same_information_gain(
                    information_gain_arr=information_gain_arr, given=given, ):

                node.set_majority_decision()

            else:

                # Compute the best information gain ix across all categories
                # for each feature
                best_information_gain_ixs = np.unravel_index(
                    np.nanargmax(information_gain_arr), information_gain_arr.shape)

                # Get the best feature ix
                best_feature_ix = best_information_gain_ixs[0]

                # Set the attribute of the node... e.g.,
                #  ----------------------
                # | node(atrribute=None) |
                # |----------------------|
                #       NOW BECOMES
                #  ---------------------------------
                # | node(atrribute=best_feature_ix) |
                # |---------------------------------|
                node.set_attribute(best_feature_ix)

                # Now that a feature has been explored, it is added
                # to the `given` set to prevent splitting on an
                # an attribute twice
                given.append(best_feature_ix)

                # Get the category corresponding ot the highest information gain...
                # note the category will be a tuple
                best_category_arr = split_categories[best_information_gain_ixs]

                # Get the subset associated with the best split point
                # if the data is continuous, then the subset
                # are those points where the feature's rows correspond
                # to the best split point category
                learning_subsets = self.__get_learning_subsets(
                    best_feature_ix=best_feature_ix,
                    best_category_arr=best_category_arr,
                    node=node,)

                # LOGGING

                # Create child nodes with the left node's category being the
                # the first interval in the best binary split point
                # list and the right node's category being the second
                # interval in the same list
                children = [TreeNode(category=category)
                            for category in best_category_arr]

                # Continue to build the tree
                for (learning_subset, child) in zip(learning_subsets, children):
                    node.add_child(child)
                    self.__train(
                        learning_set=learning_subset,
                        node=child,
                        given=given,)

    def traverse_tree(
            self,
            row_vector: np.ndarray,
            node: TreeNode,) -> int:
        """Recursive traversal of tree until a decision returned.

        :param row_vector: Vector of features.
        :param node: <class 'TreeNode'>

        :return: The decision for the row vector.
        """

        # Consider a row vector...
        #  A_i = A1         A2          A3
        #       [A1_val     A2_val      A3_val]
        # Each attribute has val corresponding to it, and
        # indexing the i^th attribute gives the category (or value)
        # associated with that particular attribute
        # axis 0 has size 1 so use of `:` in indexing does not
        value = row_vector[node.get_attribute()]

        # Special case
        if node.is_root() and node.is_leaf():
            return node.get_decision()

        # All other cases where children exist
        for child in node.get_children():

            # Base case
            if value in child.get_category() and child.is_leaf():
                return child.get_decision()

            # Recursive case
            elif value in child.get_category() and not child.is_leaf():
                return self.traverse_tree(row_vector=row_vector, node=child)

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

        # TODO: Remove this
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

    def __compute_split_categories(
            self,
            node: TreeNode,
            given: list,) -> list[list[tuple[pd.Interval] or None]]:
        """Return 3D list of potential splits for each valid feature."""

        features = node.get_features()
        all_feature_categories = []
        for feature in range(features.shape[1]):

            feature_categories = []
            if feature not in given:

                # Extract 1D array of values for a feature
                feature_vector = features[:, feature]

                # Sort the values in the feature vector
                sorted_feature_vector = feature_vector[np.argsort(
                    feature_vector)]

                for val in range(len(sorted_feature_vector) - 1):

                    # Compute split point
                    cur_val = sorted_feature_vector[val]
                    adj_val = sorted_feature_vector[val+1]
                    avg_val = (cur_val + adj_val) / 2

                    # Compute bounds
                    lower_bound = pd.Interval(
                        left=-np.inf, right=avg_val, closed='neither')
                    upper_bound = pd.Interval(
                        left=avg_val, right=np.inf, closed='left')

                    # Add potential bound tuple to feature categories
                    feature_categories.append([lower_bound, upper_bound])

            else:
                for val in range(features.shape[0] - 1):
                    feature_categories.append([None, None])

            # Append feature categories to parent list
            all_feature_categories.append(feature_categories)

        return np.array(all_feature_categories, dtype=object)

    def __compute_category_information_gain(
            self,
            learning_set_entropy: float,
            split_categories: list[list[tuple[pd.Interval] or None]],
            node: TreeNode,
            given: list) -> np.ndarray:
        """Computes information gain for each feature and each feature category."""

        # Information gain array -- (m features, n - 1 categories)
        information_gain_arr = np.empty(
            shape=(node.get_features().shape[1],
                   node.get_features().shape[0] - 1))

        # Compute label counts as they will be needed later
        label_counts = np.unique(node.get_labels(), return_counts=True)[-1]

        # For a given feature, there will be a number of categories
        # equal to n - 1. For each category in a feature, compute
        # the information gain associated with that category
        # and append it to a list
        # if a None is encounter for a feature, do np.nan for that
        # spot for argmax computation reasons
        for feature_ix, feature in enumerate(split_categories):
            for category_ix, categories in enumerate(feature):

                if None in categories:
                    information_gain_arr[feature_ix, :] = np.nan

                else:
                    # Create a distinct feature label set using
                    # the categories above
                    node.set_discrete_feature_label_set(
                        feature_ix=feature_ix,
                        categories=categories)

                    # Compute the number of the categories that
                    # are in the same set as each of the labels
                    count_matrix = self.count_feature_categories_belonging_to_label_categories(
                        attr_label_arr=node.get_discrete_feature_label_set())

                    # Compute the expected information gain for such a matrix
                    expected_information_gain = self.__expected_information(
                        label_counts=label_counts, subset_counts=count_matrix)

                    # Compute information gain
                    information_gain = self.__information_gain(
                        learning_set_entropy,
                        expected_information=expected_information_gain)

                    # Update the information gain array
                    information_gain_arr[feature_ix,
                                         category_ix] = information_gain

        # Return the array
        return information_gain_arr

    def __get_learning_subsets(
            self,
            best_feature_ix: int,
            best_category_arr: np.ndarray[pd.Interval],
            node: TreeNode,) -> tuple[np.ndarray]:
        """Extracts subsets from current learning set based on the categories.

        This only works when the splitting is binary since a boolean
        vector is used.
        """

        # Indices where the attribute: value < threshold
        feature_vector = node.get_features()[:, best_feature_ix]
        threshold = best_category_arr[0].right  # (-inf, threshold)
        category_1_ixs = np.where(feature_vector < threshold)[0]

        # Learnings subsets
        cur_learning_set = node.get_learning_set()
        num_rows = cur_learning_set.shape[0]
        learning_subset_less = cur_learning_set[category_1_ixs, :]
        learning_subset_greater_than_or_eq = cur_learning_set[[
            ix for ix in range(num_rows) if ix not in category_1_ixs], :]

        # Return a tuple where the first element is the subset
        # corresponding to attribute:values < threshold and the second
        # attribute:values >= threshold
        return learning_subset_less, learning_subset_greater_than_or_eq

    def __same_information_gain(self, information_gain_arr: np.ndarray, given: list) -> bool:
        """Return bool for whether all values of information gain are the same.

        :param information_gain_arr:
        :param given: None or <class 'list'>
        """

        if given is not None:
            sliced_information_gain_arr = information_gain_arr[[
                ix for ix in range(information_gain_arr.shape[0]) if ix not in given], :]
            return not np.any(sliced_information_gain_arr)
        return not np.any(information_gain_arr)

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
        'testing_data',
        help='path to txt file with testing data for ID3 decision tree.',
        type=str,)
    parser.add_argument(
        '--percentage',
        help='bool to output proportion or just the number of accurate predictions. (default: False)',
        choices=['True', 'False'],
        default='False')
    parser.add_argument(
        '--precision',
        help='number of decimal points to keep. Not used by default (default: None)',
        type=int,
        default=None)
    args = parser.parse_args()

    # Tree instantiation
    tree = ID3DecisionTree()

    # Load data
    learning_set = np.loadtxt(args.training_data)
    testing_set = np.loadtxt(args.testing_data)
    
    # Reshape data
    if len(learning_set.shape) == 1:
        learning_set = np.expand_dims(learning_set, axis=0)
    if len(testing_set.shape) == 1:
        testing_set = np.expand_dims(testing_set, axis=0)

    # Instantiate tree obj
    tree = ID3DecisionTree()
    
    # Train tree
    tree.train(learning_set=learning_set)

    # TODO: Remove tree
    # tree.display_tree()

    # Simple case where the test set has more than 1 data item
    preds = tree.predict(testing_set=testing_set)

    # Compute accuracy
    accuracy = np.count_nonzero(preds == testing_set[:, -1])

    # Output accuracy
    if bool(strtobool(args.percentage)):
        percentage = accuracy/testing_set.shape[0]
        if args.precision is not None:
            print(round(percentage, args.precision))
        else:
            print(percentage)
    else:
        print(accuracy)
