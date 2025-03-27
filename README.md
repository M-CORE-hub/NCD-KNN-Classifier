# NCD-KNN-Classifier

A kNN classifier based on the Normalized Compression Distance (NCD) for text classification.

## Installation

Install the package using pip:

```bash
pip install NCD-KNN-Classifier
```

## Usage

Here's a basic example of how to use the classifier:

```python
from datasets import load_dataset
from NCD_KNN_Classifier import CompNCDClassifier

# Example of imdb dataset
dataset = load_dataset("imdb")
test_samples = dataset["test"].shuffle(seed=42).select(range(200))

# Compressing and Saving the Training Dataset Footprint
classifier = CompNCDClassifier(
    train_dataset=dataset['train'],
    test_dataset=test_samples,
    k=3,
    compressor="gzip",
    verbose=True
)
classifier.save_to_pickle("train_footprints.pkl")

# Prediction on the test set
metrics = classifier.evaluate()
print("Evaluation metrics:", metrics)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
