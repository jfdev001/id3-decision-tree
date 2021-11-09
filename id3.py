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
    """Node for decision tree."""

    def __init__(self):
        """Define state for Node.

        TODO: Is this state formulation valid??
        """

        self.attr = None
        self.next = None
        self.children = None


class ID3DecisionTree:
    """ID3 Decision Tree."""

    def __init__(self, X, y):
        """Define state for ID3DecisionTree.

        :param x:
        :param y:
        """

        # Save data
        self.X = X
        self.y = y

        # # Number of classes
        # self.labels, self.labels_counts = np.unique(
        #     self.y, return_counts=True)

        # # Compute entropy of set
        # self.entropy_of_dataset = self.__entropy(self.labels_counts)

    def id3(self,):
        """Create the ID3DecisionTree.

        :param:

        :return:
        """

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

    def __sort(self,):
        """Sort training data along each attribute.

        :param:

        :return:
        """

        pass

    def __binary_split(self,):
        """Determine potential binary split point based on attr value changes.

        :param:

        :return:
        """
        pass

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


        :param label_counts:
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
