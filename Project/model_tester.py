# Python Standard Libraries
from typing import List, Tuple

# Third Party Libraries
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# Project Libraries
import utils


class ModelTester:
    def _get_prediction_label(self, prediction: ndarray) -> int:
        """Receives a (1, 2) numpy array with the probabilities of sarcasm for
        the first and second answer and returns the label of the answer with the
        highest probability (i.e., the prediction label of the model).

        Parameters
        ----------
        prediction : ndarray
            A (1, 2)-shaped numpy array with the probabilities of sarcasm for
            the first and second answer.

        Returns
        -------
        int : label
            The label of the answer with the highest probability.
        """
        return np.argmax(prediction)

    def _get_row_prediction_and_label(
        self,
        row: Series,
        answer_number: int,
        model: CrossEncoder,
    ) -> Tuple[float, int]:
        """Returns the probability of sarcasm for an answer and the true
        sarcasm label of that answer.

        Parameters
        ----------
        row : Series
            A row Series of the test DataFrame.
        answer_number : int
            The number of the answer. It can be 1 or 2.
        model : CrossEncoder
            The model used to predict the sarcasm probability.

        Returns
        -------
        prediction : float
            The probability of sarcasm for the answer.
        label : int
            The true sarcasm label of the answer.
        """
        comment, answer, label = utils.get_example_input(row, answer_number)

        input = [[comment, answer]]
        prediction = model.predict(input, apply_softmax=True)

        return prediction, label

    def get_probabilities_and_labels(
        self,
        dataset: DataFrame,
        model: CrossEncoder,
        verbose: bool = False,
    ) -> Tuple[List[float], List[int]]:
        """Returns the probabilities of sarcasm and the true sarcasm labels of
        the test DataFrame.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to test the model on.
        model : CrossEncoder
            The model used to predict the sarcasm probability.
        verbose : bool, default=False
            If True, prints the progress of the test.
        preloaded_datasets : bool, default=False
            If True, uses the preloaded datasets.

        Returns
        -------
        predictions, labels : Tuple[List[float], List[int]]
            A tuple with the probabilities of sarcasm and the true sarcasm
            labels of the test DataFrame.
        """
        predictions = list()
        labels = list()
        iterator = list(dataset.iterrows())
        for _, row in tqdm(iterator, desc="Predicting", disable=not verbose):
            for answer_number in (1, 2):
                prediction, label = self._get_row_prediction_and_label(
                    row, answer_number, model
                )
                predictions.append(prediction)
                labels.append(label)

        return predictions, labels

    def get_labels_from_probs(self, predictions_probs) -> List[int]:
        """Receives a list of probabilities and returns a list with the same
        length with the corresponding labels.

        Parameters
        ----------
        labels : List[float]
            A list of labels.
        """
        return list(map(self._get_prediction_label, predictions_probs))
