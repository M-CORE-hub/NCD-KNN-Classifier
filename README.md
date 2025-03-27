# NCDClassifier

A kNN classifier based on the Normalized Compression Distance (NCD) for text classification.

## Installation

Install the package using pip:

```bash
pip install NCD_KNN_Classifier
```

## Usage

Here's a basic example of how to use the classifier:

```python
from NCD_KNN_Classifier import CompNCDClassifier

# Assuming you have train and test datasets prepared
classifier = CompNCDClassifier(
    train_dataset=dataset['train'],
    test_dataset=dataset['test'],
    k=3,
    compressor="gzip",
    verbose=True
)
classifier.save_to_pickle("train_footprints.pkl")
metrics = classifier.evaluate()
print("Evaluation metrics:", metrics)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.