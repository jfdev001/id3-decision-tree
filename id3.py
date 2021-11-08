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


class ID3DecisionTree:
    """ID3 Decision Tree."""

    def __init__(self,):
        """Define state for ID3DecisionTree.

        :param:
        """

        pass

    def train(self,):
        """Train ID3DecisionTree.

        :param:

        :return:
        """

        pass

    def test(self,):
        """Test ID3DecisionTree.

        :param:

        :return:
        """
        pass

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

    def __compute_information_gain(self,):
        """Calcluate information gain for split point.

        :param:

        :return:
        """
        pass


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
