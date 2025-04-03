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

# Example of imdb dataset for classification of two labels
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
classifier.save_to_pickle("train_footprint.pkl")

# Inference on the test set
metrics = classifier.evaluate()
print("Evaluation metrics:", metrics)
```
### Output Example

After running the example, you should see output similar to this:

```bash
INFO:NCD_KNN_Classifier.classifier:Evaluating on the dataset...
Evaluating 200/200: 100%|██████████| 200/200 [00:55<00:00,  3.63it/s]
Evaluation metrics: {'accuracy': 0.69, 'precision': 0.6969369369369369, 'recall': 0.69, 'f1_score': 0.68476726342711}
```

## New Features in v0.2.1

- Added explainability in the **classify** function, providing information about the nearest neighbors and their influence on classification decisions.

- Introduced the **optimizer** function to assist in finding the optimal k and compressor type.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
