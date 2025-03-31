import bz2
import gzip
import zlib
import logging
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict, Union, Optional
from multiprocessing import Pool, cpu_count
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compressed_size(text: str, compressor: str = "gzip") -> int:
    """Calculate the compressed size of text using specified compressor.

    Args:
        text: Input text to compress
        compressor: Compression algorithm to use ('gzip', 'zlib', 'bz2') (default: "gzip")

    Returns:
        int: Size of compressed text in bytes

    Raises:
        ValueError: If an unsupported compressor is specified
    """
    encoded = text.encode('utf-8')
    compressors = {
        "gzip": lambda: len(gzip.compress(encoded)),
        "zlib": lambda: len(zlib.compress(encoded)),
        "bz2": lambda: len(bz2.compress(encoded))
    }
    return compressors.get(
        compressor,
        lambda: exec(f"raise ValueError('Unsupported compressor: {compressor}')")
    )()


def ncd(x: str, y: str, c_x: int, c_y: int, compressor: str = "gzip") -> float:
    """Calculate Normalized Compression Distance (NCD) between two texts.

    Args:
        x: First text string
        y: Second text string
        c_x: Precomputed compressed size of x
        c_y: Precomputed compressed size of y
        compressor: Compression algorithm to use (default: "gzip")

    Returns:
        float: NCD value between 0 and 1+
    """
    sep = '\0'
    c_xy = compressed_size(x + sep + y, compressor)
    return (c_xy - min(c_x, c_y)) / max(c_x, c_y)


def classify(
    test_text: str,
    training_data_with_sizes: List[Tuple],
    k: int = 3,
    compressor: str = "gzip",
    explain: bool = False
) -> Tuple[int, Optional[Dict[str, List[Tuple[int, float, str]]]]]:
    """Classify a text using precomputed training data and k-nearest neighbors with optional explainability.

    Args:
        test_text: Text to classify
        training_data_with_sizes: List of tuples (label, train_text, compressed_size)
        k: Number of nearest neighbors to consider (default: 3)
        compressor: Compression algorithm to use (default: "gzip")
        explain: If True, return explanation with k-nearest neighbors (default: False)

    Returns:
        Tuple[int, Optional[Dict[str, List[Tuple[int, float, str]]]]]:
            - int: Predicted label
            - Optional[Dict]: If explain=True, dictionary with 'k_nearest' key mapping to a list of
                (label, distance, train_text) for the k nearest neighbors; otherwise None
    """
    c_test = compressed_size(test_text, compressor)
    distances = [
        (label, ncd(test_text, train_text, c_test, c_train, compressor), train_text)
        for label, train_text, c_train in training_data_with_sizes
    ]
    distances.sort(key=lambda x: x[1])
    k_nearest = distances[:k]
    label_counts = Counter([neighbor[0] for neighbor in k_nearest])
    predicted_label = label_counts.most_common(1)[0][0]
    if explain:
        explanation = {
            "k_nearest": [(label, distance, text) for label, distance, text in k_nearest]
        }
        return predicted_label, explanation
    return predicted_label, None


def _classify_with_params(args: Tuple[str, List[Tuple], int, str]) -> int:
    """Helper function for parallel classification with explicit parameters.

    Args:
        args: Tuple of (text, training_data_with_sizes, k, compressor)
            - text: Text to classify
            - training_data_with_sizes: List of tuples (label, train_text, compressed_size)
            - k: Number of nearest neighbors
            - compressor: Compression algorithm to use

    Returns:
        int: Predicted label
    """
    text, training_data, k, compressor = args
    prediction, _ = classify(text, training_data, k, compressor, explain=False)
    return prediction


def optimizer(
    training_data_with_sizes: List[Dict],
    eval_dataset: List[Dict],
    k_range: tuple[int, int] = (1, 10),
    compressors: List[str] = ["gzip", "zlib", "bz2"],
    reference_metric: str = 'f1_score',
    average: str = 'weighted',
    text_key: str = 'text',
    label_key: str = 'label',
    verbose: bool = True
) -> Dict[str, Union[int, str, Dict[tuple[int, str], Dict[str, float]]]]:
    """Optimize k and compressor hyperparameters for NCD classification.

    Args:
        training_data_with_sizes: List of dicts with text and label keys (compressed sizes computed internally)
        eval_dataset: Dataset to evaluate performance on (list of dicts with text and label)
        k_range: Tuple of (min_k, max_k) defining the range of k values to test (default: (1, 10))
        compressors: List of compression algorithms to test (default: ["gzip", "zlib", "bz2"])
        reference_metric: Metric to optimize ('accuracy', 'precision', 'recall', 'f1_score') (default: 'f1_score')
        average: Averaging method for metrics ('weighted', 'micro', etc.) (default: 'weighted')
        text_key: Key for text data in datasets (default: 'text')
        label_key: Key for label data in datasets (default: 'label')
        verbose: Enable detailed logging (default: True)

    Returns:
        Dict[str, Union[int, str, Dict[tuple[int, str], Dict[str, float]]]]: Dictionary containing
            - best_k: The optimal k value
            - best_compressor: The optimal compressor
            - results: Dictionary mapping (k, compressor) tuples to performance metrics
                Each metric dict contains 'accuracy', 'precision', 'recall', and 'f1_score'

    Raises:
        ValueError: If reference_metric is invalid or datasets are empty
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    if reference_metric not in ['accuracy', 'precision', 'recall', 'f1_score']:
        raise ValueError(f"Invalid reference_metric. Must be one of: 'accuracy', 'precision', 'recall', 'f1_score'")
    if not training_data_with_sizes or not eval_dataset:
        raise ValueError("Training data and evaluation dataset cannot be empty")

    texts = [example[text_key] for example in eval_dataset]
    true_labels = [example[label_key] for example in eval_dataset]
    min_k, max_k = k_range
    results = {}
    best_k = min_k
    best_compressor = compressors[0]
    best_score = -float('inf')

    for compressor in compressors:
        logger.info(f"Precomputing compressed sizes for training data with {compressor}...")
        training_data = [
            (item[label_key], item[text_key], compressed_size(item[text_key], compressor))
            for item in tqdm(
                training_data_with_sizes,
                desc=f"Computing sizes ({compressor})",
                disable=not logger.isEnabledFor(logging.INFO)
            )
        ]
        logger.info(f"Optimizing k over range {k_range} for {reference_metric} with {compressor}...")
        for k in tqdm(
            range(min_k, max_k + 1),
            desc=f"Testing k values ({compressor})",
            disable=not logger.isEnabledFor(logging.INFO)
        ):
            with Pool(cpu_count()) as pool:
                predictions = pool.map(
                    _classify_with_params,
                    [(text, training_data, k, compressor) for text in texts]
                )
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average=average
            )
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            results[(k, compressor)] = metrics
            if metrics[reference_metric] > best_score:
                best_score = metrics[reference_metric]
                best_k = k
                best_compressor = compressor

    logger.info(f"Best combination found: k={best_k}, compressor={best_compressor} with {reference_metric} = {best_score}")
    return {
        "best_k": best_k,
        "best_compressor": best_compressor,
        "results": results
    }