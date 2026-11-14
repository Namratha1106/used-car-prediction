# Used Car Price Prediction System

A Flask application that trains and compares Linear Regression, Decision Tree, and
Random Forest models using the actual used-car CSV files in `dataset/`. It provides
dynamic price prediction, model evaluation and dataset analysis dashboards, and
SQLite-backed prediction history.

## Dataset

Place the extracted Kaggle CSV file(s) in the existing `dataset/` folder. The
training script inspects each CSV, selects the most usable file containing a
positive numeric selling-price column, and discovers its input features. A
combined/all-records file is preferred when the usable record counts tie. City
exports should not be concatenated with a combined file because they may repeat
the same listings.

For the dataset used while building this project, the selected file is
`UsedCars_Combined.csv`, with `price` as the target and the available car
attributes discovered directly from its columns. Features such as transmission
or previous-owner count are shown only if they exist in the selected dataset.

## Setup

From the project directory (the directory containing `dataset/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open <http://127.0.0.1:5000>. Retrain after replacing or updating the CSV files.
The three fitted preprocessing/model pipelines and evaluation metadata are
written to `models/`; the prediction history database is created in `database/`.

## Dataset and model handling

- The target is identified from the actual CSV headers; malformed, non-positive,
  and missing target values are excluded.
- Empty/index-like columns and constant or empty features are excluded; remaining
  missing values are imputed within the model pipeline.
- Numeric and categorical features are inferred from the selected data.
- The same seeded 80/20 split is used for all three models. Imputation, scaling,
  and one-hot encoding are fit only on the training partition.
- Evaluation metrics and charts are generated from held-out predictions and the
  selected dataset's actual records.
- `full_model_name`, model, fuel, and other columns are not invented if absent.

## Project layout

```text
dataset/                 Place source CSV file(s) here
models/                  Trained model pipelines and metadata
templates/               Flask/Jinja pages
static/css/style.css     Responsive interface styles
static/js/script.js      Charts and prediction/history interactions
database/                SQLite prediction history
train_model.py           Dataset inspection and training
app.py                   Flask routes and prediction API
```
