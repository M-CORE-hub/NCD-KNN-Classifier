import os
import pickle
import logging
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Union, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .utils import compressed_size, ncd


class CompNCDClassifier:
    """Classifier that uses the normalized compression distance (NCD) to predict labels.

    Attributes:
        k (int): Number of nearest neighbors to consider
        compressor (str): Compression algorithm to use
        pickle_path (str): Path to save/load precomputed data
        training_data_with_sizes (list): Precomputed training data with sizes
        text_key (str): Key for text data in dataset
        label_key (str): Key for label data in dataset
        train_dataset (Union[List[Dict], 'Dataset']): Training dataset
        test_dataset (Union[List[Dict], 'Dataset']): Test dataset
    """

    def __init__(
        self,
        train_dataset: Union[List[Dict], 'Dataset'],
        test_dataset: Optional[Union[List[Dict], 'Dataset']] = None,
        k: int = 3,
        compressor: str = "gzip",
        pickle_path: str = "train_data_with_sizes.pkl",
        verbose: bool = True,
        text_key: str = 'text',
        label_key: str = 'label'
    ) -> None:
        """Initialize the NCD classifier.

        Args:
            train_dataset: Training dataset with text and labels
            test_dataset: Optional test dataset
            k: Number of nearest neighbors (default: 3)
            compressor: Compression algorithm ('gzip', etc.) (default: "gzip")
            pickle_path: Path for saving/loading precomputed data (default: "train_data_with_sizes.pkl")
            verbose: Enable detailed logging (default: True)
            text_key: Dataset key for text data (default: 'text')
            label_key: Dataset key for label data (default: 'label')
        """
        self.k = k
        self.compressor = compressor.lower()
        self.pickle_path = pickle_path
        self.training_data_with_sizes = None
        self.text_key = text_key
        self.label_key = label_key
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self._setup_logging(verbose)

    def _setup_logging(self, verbose: bool) -> None:
        """Configure logging based on verbosity.

        Args:
            verbose: If True, set logging level to INFO, else WARNING
        """
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def save_to_pickle(self, pickle_path: Optional[str] = None) -> None:
        """Precompute and save compressed sizes of training data.

        Args:
            pickle_path: Optional custom path for pickle file
        """
        self.logger.info("Precomputing compressed sizes...")
        self.training_data_with_sizes = [
            (
                example[self.label_key],
                example[self.text_key],
                compressed_size(example[self.text_key], self.compressor)
            )
            for example in tqdm(
                self.train_dataset,
                desc="Precomputing sizes",
                disable=not self.logger.isEnabledFor(logging.INFO)
            )
        ]
        target_path = pickle_path if pickle_path is not None else self.pickle_path
        with open(target_path, 'wb') as f:
            pickle.dump(self.training_data_with_sizes, f)
        self.logger.info(f"Data saved to {target_path}")

    def load_from_pickle(self, pickle_path: Optional[str] = None) -> None:
        """Load precomputed training data from pickle file.

        Args:
            pickle_path: Optional custom path to pickle file

        Raises:
            FileNotFoundError: If pickle file doesn't exist
        """
        target_path = pickle_path if pickle_path is not None else self.pickle_path
        if os.path.exists(target_path):
            with open(target_path, 'rb') as f:
                self.training_data_with_sizes = pickle.load(f)
            self.logger.info(f"Data loaded from {target_path}")
        else:
            raise FileNotFoundError(f"File not found: {target_path}. Run save_to_pickle first.")

    def _predict_single(self, args: tuple[str, List[tuple]]) -> int:
        """Predict label for a single text using NCD.

        Args:
            args: Tuple of (text, training_data) where text is the input
                 and training_data is list of (label, text, compressed_size)

        Returns:
            int: Predicted label
        """
        text, training_data = args
        c_test = compressed_size(text, self.compressor)
        distances = [
            (label, ncd(text, train_text, c_test, c_train, self.compressor))
            for label, train_text, c_train in training_data
        ]
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:self.k]
        label_counts = Counter([neighbor[0] for neighbor in k_nearest])
        return label_counts.most_common(1)[0][0]

    def predict(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Predict labels for one or multiple texts.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Union[int, List[int]]: Predicted label(s)
        """
        if self.training_data_with_sizes is None:
            self.load_from_pickle()

        if isinstance(texts, str):
            return self._predict_single((texts, self.training_data_with_sizes))
        
        self.logger.info("Predicting in parallel...")
        with Pool(cpu_count()) as pool:
            predictions = pool.map(
                self._predict_single,
                [(text, self.training_data_with_sizes) for text in texts]
            )
        return predictions

    def evaluate(
        self,
        dataset: Optional[Union[List[Dict], 'Dataset']] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """Evaluate classifier performance.

        Args:
            dataset: Optional dataset to evaluate on (uses test_dataset if None)
            average: Averaging method for metrics ('weighted', 'micro', etc.)

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision, recall, and F1 score

        Raises:
            ValueError: If no test dataset is available
        """
        if dataset is None:
            if self.test_dataset is None:
                raise ValueError("No test dataset provided in __init__ and no dataset parameter given.")
            eval_dataset = self.test_dataset
        else:
            eval_dataset = dataset
        
        if self.training_data_with_sizes is None:
            self.load_from_pickle()
        
        self.logger.info("Evaluating on the dataset...")
        texts = [example[self.text_key] for example in eval_dataset]
        true_labels = [example[self.label_key] for example in eval_dataset]
        total = len(texts)
        predictions = []
        
        with Pool(cpu_count()) as pool:
            tasks = [(text, self.training_data_with_sizes) for text in texts]
            with tqdm(
                total=total,
                desc="Evaluating 0/{}".format(total),
                disable=not self.logger.isEnabledFor(logging.INFO)
            ) as pbar:
                for i, pred in enumerate(pool.imap(self._predict_single, tasks)):
                    predictions.append(pred)
                    pbar.set_description(f"Evaluating {i + 1}/{total}")
                    pbar.update(1)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=average
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }