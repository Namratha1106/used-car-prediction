from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
TARGET_NAMES = {
    "price",
    "sellingprice",
    "resaleprice",
    "saleprice",
    "sellingpriceinrupees",
}
ID_COLUMN = re.compile(r"^(id|index|unnamed)(?:\b|[_\s:]|$)", re.IGNORECASE)


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_target(columns: list[str]) -> str | None:
    for column in columns:
        normalized = normalized_name(column)
        if normalized in TARGET_NAMES:
            return column
    for column in columns:
        normalized = normalized_name(column)
        if "price" in normalized and any(
            word in normalized for word in ("sell", "sale", "resale", "price")
        ):
            return column
    return None


def inspect_csv(path: Path) -> tuple[pd.DataFrame, str] | None:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        print(f"Skipping {path.name}: could not read CSV ({exc}).")
        return None
    target = find_target([str(column) for column in frame.columns])
    if target is None:
        print(f"Skipping {path.name}: no recognizable selling-price column.")
        return None
    return frame, target


def load_dataset() -> tuple[pd.DataFrame, str, Path]:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}. Create it and place your car CSV file(s) inside."
        )
    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATASET_DIR}. Place the extracted used-car dataset CSV in the dataset folder."
        )

    candidates: list[tuple[int, int, str, pd.DataFrame, str, Path]] = []
    for path in csv_files:
        inspected = inspect_csv(path)
        if inspected is None:
            continue
        frame, target = inspected
        numeric_target = pd.to_numeric(frame[target], errors="coerce")
        valid_rows = int((numeric_target.notna() & (numeric_target > 0)).sum())
        if valid_rows == 0:
            print(f"Skipping {path.name}: target column has no positive numeric prices.")
            continue
        combined_preference = int(
            any(term in path.stem.lower() for term in ("combined", "complete", "all"))
        )
        candidates.append(
            (valid_rows, combined_preference, path.name.lower(), frame, target, path)
        )

    if not candidates:
        raise ValueError(
            "The CSV files in the dataset folder do not contain a usable selling-price column "
            "with positive numeric values."
        )
    _, _, _, frame, target, path = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    print(f"Selected {path.name}: {len(frame):,} source records; target column is {target!r}.")
    return frame, target, path


def display_label(column: str) -> str:
    label = re.sub(r"[_()]+", " ", column).strip()
    label = re.sub(r"\s+", " ", label)
    return label[:1].upper() + label[1:]


def prepare_dataset(
    source: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    frame = source.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    target = next((column for column in frame.columns if normalized_name(column) == normalized_name(target)), target)
    if target not in frame.columns:
        raise ValueError("The selected target column disappeared while reading the dataset.")

    prices = pd.to_numeric(frame[target], errors="coerce")
    valid_target = prices.notna() & np.isfinite(prices) & (prices > 0)
    frame = frame.loc[valid_target].copy()
    frame[target] = prices.loc[valid_target].astype(float)
    frame = frame.drop_duplicates()
    prices = frame[target].astype(float)
    if len(frame) < 10:
        raise ValueError(
            f"Only {len(frame)} usable records have a positive numeric selling price; at least 10 are required."
        )

    omitted = [
        column
        for column in frame.columns
        if column == target or ID_COLUMN.match(column.strip()) or not column.strip()
    ]
    features = frame.drop(columns=omitted).copy()

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            features[column] = pd.to_numeric(series, errors="coerce")
            continue
        converted = pd.to_numeric(series, errors="coerce")
        observed = int(series.notna().sum())
        numeric_ratio = float(converted.notna().sum() / observed) if observed else 0.0
        if observed and numeric_ratio >= 0.9:
            numeric_columns.append(column)
            features[column] = converted
        else:
            categorical_columns.append(column)
            features[column] = series.map(
                lambda value: str(value).strip() if pd.notna(value) else np.nan
            )

    usable_features = [
        column
        for column in features.columns
        if features[column].notna().any() and features[column].nunique(dropna=True) > 1
    ]
    features = features[usable_features]
    numeric_columns = [column for column in numeric_columns if column in usable_features]
    categorical_columns = [column for column in categorical_columns if column in usable_features]
    if not usable_features:
        raise ValueError("No usable car feature columns remain after cleaning the dataset.")
    if len(features) < 10:
        raise ValueError(
            f"Only {len(features)} usable records remain after cleaning; at least 10 are required."
        )
    return features, prices.loc[features.index], numeric_columns, categorical_columns, omitted


def category_counts(series: pd.Series, limit: int = 18) -> dict[str, Any]:
    counts = series.dropna().astype(str).value_counts()
    return {"labels": counts.index[:limit].tolist(), "values": counts.iloc[:limit].astype(int).tolist()}


def histogram(series: pd.Series, bins: int = 18) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"labels": [], "values": []}
    counts, edges = np.histogram(values, bins=min(bins, max(1, len(np.unique(values)))))
    labels = [
        f"{edges[index]:,.0f}–{edges[index + 1]:,.0f}"
        for index in range(len(counts))
    ]
    return {"labels": labels, "values": counts.astype(int).tolist()}


def build_analysis(
    features: pd.DataFrame, prices: pd.Series, target_name: str
) -> dict[str, Any]:
    raw = features.copy()
    raw[target_name] = prices
    normalized_columns = {normalized_name(column): column for column in raw.columns}

    def find_column(*terms: str) -> str | None:
        return next(
            (
                column
                for normalized, column in normalized_columns.items()
                if any(term in normalized for term in terms)
            ),
            None,
        )

    fuel_column = find_column("fuel", "fueltype")
    brand_column = find_column("brand", "make", "manufacturer")
    year_column = find_column("year", "manufactureyear", "modelyear")
    distance_column = find_column("distance", "kilomet", "mileage", "odometer")
    price_values = prices.to_numpy(dtype=float)
    charts: dict[str, Any] = {
        "price": histogram(prices),
        "fuel": category_counts(raw[fuel_column]) if fuel_column else None,
        "brand": category_counts(raw[brand_column], 15) if brand_column else None,
        "year": histogram(raw[year_column]) if year_column else None,
        "distance": histogram(raw[distance_column]) if distance_column else None,
    }
    return {
        "records": int(len(features)),
        "feature_count": int(features.shape[1]),
        "target_label": display_label(target_name),
        "average_price": float(np.mean(price_values)),
        "minimum_price": float(np.min(price_values)),
        "maximum_price": float(np.max(price_values)),
        "charts": charts,
    }


def make_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def feature_specs(
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in features.columns:
        if column in numeric_columns:
            values = pd.to_numeric(features[column], errors="coerce").dropna()
            result[column] = {
                "label": display_label(column),
                "kind": "number",
                "minimum": float(values.min()) if not values.empty else 0,
                "maximum": float(values.max()) if not values.empty else 0,
                "step": "1" if pd.api.types.is_integer_dtype(features[column].dtype) else "any",
                "non_negative": bool(not values.empty and values.min() >= 0),
            }
        elif column in categorical_columns:
            options = sorted(features[column].dropna().astype(str).unique().tolist())
            result[column] = {
                "label": display_label(column),
                "kind": "category",
                "options": options,
                "free_text": len(options) > 100,
            }
    return result


def train() -> dict[str, Any]:
    source, target, source_path = load_dataset()
    features, prices, numeric_columns, categorical_columns, omitted = prepare_dataset(source, target)
    x_train, x_test, y_train, y_test = train_test_split(
        features, prices, test_size=0.2, random_state=42
    )
    estimators = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42, min_samples_leaf=2),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float | str]] = {}
    actual = y_test.to_numpy(dtype=float)
    predictions: dict[str, np.ndarray] = {}

    for key, estimator in estimators.items():
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
                ("regressor", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predicted = np.asarray(pipeline.predict(x_test), dtype=float)
        predictions[key] = predicted
        metrics[key] = {
            "name": {
                "linear_regression": "Linear Regression",
                "decision_tree": "Decision Tree Regressor",
                "random_forest": "Random Forest Regressor",
            }[key],
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r2": float(r2_score(y_test, predicted)),
        }
        joblib.dump(pipeline, MODELS_DIR / f"{key}.pkl")
        print(
            f"{metrics[key]['name']}: RMSE {metrics[key]['rmse']:,.2f}, "
            f"R2 {metrics[key]['r2']:.4f}"
        )

    best_key = max(
        metrics,
        key=lambda key: (float(metrics[key]["r2"]), -float(metrics[key]["rmse"])),
    )
    random_forest = joblib.load(MODELS_DIR / "random_forest.pkl")
    regressor = random_forest.named_steps["regressor"]
    preprocessor = random_forest.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()
    importance_items = sorted(
        zip(transformed_names, regressor.feature_importances_),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:15]
    importance = {
        "labels": [str(name).replace("numeric__", "").replace("categorical__", "") for name, _ in importance_items],
        "values": [float(value) for _, value in importance_items],
    }
    specs = feature_specs(features, numeric_columns, categorical_columns)
    analysis = build_analysis(features, prices, target)
    metadata: dict[str, Any] = {
        "source_file": source_path.name,
        "target_column": target,
        "target_label": display_label(target),
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "excluded_columns": omitted,
        "feature_specs": specs,
        "metrics": metrics,
        "best_model": best_key,
        "actual_vs_predicted": {
            "actual": actual.tolist(),
            "predicted": predictions[best_key].tolist(),
        },
        "feature_importance": importance,
        "analysis": analysis,
    }
    joblib.dump(metadata, MODELS_DIR / "model_metadata.pkl")
    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"\nBest model: {metrics[best_key]['name']}")
    print(f"Saved three pipelines and metadata in {MODELS_DIR}")
    return metadata


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
