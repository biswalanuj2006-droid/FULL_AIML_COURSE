# DATASETS — Registry & Download Guide

This folder documents the public datasets referenced throughout the course.
Download files **into this folder** (one subfolder per dataset) so every
project has one predictable location.

General rules:

1. **Check the license before you use a dataset commercially** — licenses
   differ (CC-BY, CC0, research-only, Kaggle terms, UCI terms). Never assume.
2. Prefer downloads via script / official API over manual browser downloads.
3. Record what you used: dataset version + date + source URL in your
   project README (reproducibility).
4. Do not commit large data files to git — download on setup or use DVC.

---

## Quick pick table

| Module | Dataset | Task | Size | Source |
|---|---|---|---|---|
| 06-10 ML / classification | Iris | 3-class flower classification | 150 rows | sklearn built-in |
| 10 Regression | California housing | median house value | 20.6k rows | sklearn built-in |
| 07-09 Preprocessing/EDA | Titanic | survival classification | 891 rows | Kaggle: `titanic` |
| 10 Regression | House prices | sale price regression | 1.4k rows | Kaggle: `house-prices-advanced-regression-techniques` |
| 09 Classification | Spam SMS | spam/ham | 5.5k msgs | UCI `sms_spam` / Kaggle: `sms-spam-collection-dataset` |
| 09/13 Classification | Adult income | income >50k | 48k rows | UCI `adult` |
| 11 Clustering | Mall customers | customer segmentation | 200 rows | Kaggle: `mall-customer-segmentation-data` |
| 16 Time series | Air passengers | monthly totals | 144 pts | built into seaborn/statmodels datasets |
| 16 Time series | Store sales | daily demand | ~900k rows | Kaggle: `store-sales-time-series-forecasting` |
| 18/20 Deep learning | MNIST | digit recognition | 70k images | torchvision / keras / `openml/mnist_784` |
| 20 CNN | CIFAR-10 | 10-class images | 60k images | torchvision / keras |
| 20 CNN | Fashion-MNIST | clothing 10-class | 70k images | torchvision / keras |
| 23 NLP | IMDB reviews | sentiment | 50k | `datasets` lib / Stanford AI `aclimdb` |
| 23 NLP | 20 Newsgroups | news topic | 18k | sklearn built-in |
| 26/28 LLM | WikiText-103 | language modeling | ~100M tokens | `datasets` lib / Hugging Face |
| 29 RAG | PDFs of your choice | document Q&A | — | project-owned corpus |
| 33 MLOps | Red wine quality | regression + tracking demo | 1.6k | UCI `wine-quality` |

## Where to find each

### sklearn built-ins (zero download)
```python
from sklearn.datasets import (load_iris, fetch_california_housing,
                              load_diabetes, fetch_20newsgroups)
```

### Seaborn / statsmodels built-ins
```python
import seaborn as sns
df = sns.load_dataset("flights")            # air passengers
```

### Deep learning datasets (auto-download on first use)
```python
from torchvision import datasets            # MNIST, CIFAR-10, Fashion-MNIST
from tensorflow import keras                # keras.datasets.mnist / cifar10 / fashion_mnist
```

### Hugging Face datasets
```python
pip install datasets
from datasets import load_dataset
ds = load_dataset("imdb")
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
```

### UCI (stable archive)
Browse: https://archive.ics.uci.edu/datasets
Classic files keep stable paths, e.g. the adult dataset CSV at
`https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
(UCI has been migrating to its new site; if a direct link 404s, search the
dataset name on the UCI site and download from the current page.)

### Kaggle
Requires an account + `pip install kaggle` + `~/.kaggle/kaggle.json` API key.
```bash
kaggle datasets download -d uciml/iris -p datasets/iris --unzip
kaggle competitions download -c titanic -p datasets/titanic
```
Dataset slugs: `uciml/iris`, `titanic` (competition),
`house-prices-advanced-regression-techniques` (competition),
`sms-spam-collection-dataset`, `mall-customer-segmentation-data`,
`store-sales-time-series-forecasting` (competition).

### OpenML
```python
pip install openml
import openml
task = openml.tasks.get_task(59)            # iris classification
```

## License / usage notes (conservative summary — verify at the source)
- sklearn/seaborn/keras/torchvision built-ins: fine for teaching and
  research; see each library's data page for redistribution terms.
- Kaggle datasets: governed by the Kaggle Terms and any dataset-specific
  license shown on the dataset page; free to use for learning, check for
  commercial projects.
- UCI: per-dataset licenses vary; most are research-friendly.
- IMDB: intended for non-commercial research/education.
- Always record license + version in project docs if you publish work.

## Data dictionary convention
Alongside each dataset put a `data_dictionary.md` (columns, dtypes, missing
values, target definition, units, leakage warnings). This is a professional
habit the course enforces in every project (see module 59 rule: document
the data before modeling it).

## Repository hygiene
Keep only small files (< ~1 MB) committed. Use `.gitignore` with:
```
datasets/**/*.csv
datasets/**/*.zip
datasets/**/raw/
```
and provide a `download.sh` / notebook that recreates the data.
