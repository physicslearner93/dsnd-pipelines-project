# Fashion Forward Forecasting: Predicting Product Recommendations from Reviews

This is my project for the **Data Science Pipelines** course of the Udacity Data Scientist Nanodegree.

StyleSense, an online women's clothing retailer, has a backlog of product reviews in which customers wrote a review but did not say whether they recommend the product. This project builds a machine learning pipeline that reads a review, together with the customer's age and the product's category, and predicts whether the customer would recommend the product.

## Summary of the project

* **One pipeline from raw data to prediction.** A scikit-learn `Pipeline` holds every step: filling missing values, scaling numbers, encoding categories, processing text, creating new features and the prediction model. Nothing is prepared outside the pipeline, so the same object is used for training and for predicting on new reviews.
* **Text processing with spaCy.** Reviews and titles are tokenized and lemmatized, stop words and punctuation are removed, and negations such as "not" are kept. The lemmas are turned into TF-IDF features on single words and word pairs.
* **New features.** The pipeline creates review length and punctuation counts, the share of adjectives, adverbs, nouns and verbs (part-of-speech tags), and a count of negations.
* **Fine-tuning.** A grid search with five-fold cross-validation tries 12 combinations of settings and optimizes macro F1, because the two classes are unbalanced (82% of reviews recommend the product).
* **Evaluation.** The tuned pipeline is trained on the training data only and then scored once on a held-out test set.

### Result on the test data

| Model | Accuracy | Macro F1 | ROC AUC | Precision (not recommended) | Recall (not recommended) |
|---|---|---|---|---|---|
| Baseline (always "recommended") | 0.823 | 0.451 | 0.500 | 0.000 | 0.000 |
| Pipeline, default settings | 0.908 | 0.830 | 0.945 | 0.790 | 0.654 |
| Pipeline, tuned | 0.905 | 0.848 | 0.947 | 0.691 | 0.835 |

The tuned pipeline finds 83.5% of the reviews that do not recommend the product, at the cost of lower precision for that class. Tuning was aimed at macro F1 on the assumption that finding those reviews matters to the business; the cost of each kind of error was not measured.

## Files in the repository

| File | Description |
|---|---|
| `starter/starter.ipynb` | The project notebook: data exploration, the pipeline, training, fine-tuning, evaluation on the test data and conclusions. It is saved with all of its outputs. |
| `starter/data/reviews.csv` | The dataset: 18,442 anonymized reviews with 8 features and the target `Recommended IND`. |
| `starter/README.md` | The note that came with the starter folder. |
| `requirements.txt` | The Python packages the notebook needs. |
| `README.md` | This file. |
| `LICENSE.txt`, `CODEOWNERS`, `.gitignore` | Files supplied with the starter repository. |

## Getting started

### Dependencies

* Python 3.10 or higher (the notebook was run with Python 3.11)
* scikit-learn
* pandas
* spaCy, with the English model `en_core_web_sm`
* matplotlib
* notebook (Jupyter)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/physicslearner93/dsnd-pipelines-project.git
cd dsnd-pipelines-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install the packages
python -m pip install -r requirements.txt

# 4. Download the spaCy English model used for lemmatization
python -m spacy download en_core_web_sm
```

### Running the notebook

```bash
cd starter
jupyter notebook starter.ipynb
```

Run all cells from top to bottom. The whole notebook takes about three minutes on a laptop. Most of that time is spaCy processing the review text and the grid search.

## How the pipeline is built

| Branch | Columns | Steps |
|---|---|---|
| Numeric | Age, Positive Feedback Count | fill gaps with the median, then scale |
| Categorical | Clothing ID, Division Name, Department Name, Class Name | fill gaps with the most frequent value, then one-hot encode, grouping rare values |
| Review words | Review Text | spaCy lemmas, then TF-IDF |
| Title words | Title | spaCy lemmas, then TF-IDF |
| Review statistics | Review Text | character count, word count, exclamation marks, question marks, longest word |
| Review grammar | Review Text | share of adjectives, adverbs, nouns and verbs, and negation count |

The branches are joined by a `ColumnTransformer` and feed a logistic regression classifier.

## Built with

* [scikit-learn](https://scikit-learn.org/) for the pipeline, the model, the grid search and the metrics
* [spaCy](https://spacy.io/) for tokenizing, lemmatizing and part-of-speech tagging
* [pandas](https://pandas.pydata.org/) for loading and exploring the data
* [Matplotlib](https://matplotlib.org/) for the charts

## Acknowledgements

The starter notebook and the dataset come from the [Udacity starter repository](https://github.com/udacity/dsnd-pipelines-project).

## License

[License](LICENSE.txt)
