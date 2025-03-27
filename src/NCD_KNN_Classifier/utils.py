import bz2
import gzip
import zlib
from typing import List, Tuple
from collections import Counter


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
    compressor: str = "gzip"
) -> int:
    """Classify a text using precomputed training data and k-nearest neighbors.

    Args:
        test_text: Text to classify
        training_data_with_sizes: List of tuples (label, train_text, compressed_size)
        k: Number of nearest neighbors to consider (default: 3)
        compressor: Compression algorithm to use (default: "gzip")

    Returns:
        int: Predicted label
    """
    c_test = compressed_size(test_text, compressor)
    distances = [
        (label, ncd(test_text, train_text, c_test, c_train, compressor))
        for label, train_text, c_train in training_data_with_sizes
    ]
    distances.sort(key=lambda x: x[1])
    k_nearest = distances[:k]
    label_counts = Counter([neighbor[0] for neighbor in k_nearest])
    return label_counts.most_common(1)[0][0]