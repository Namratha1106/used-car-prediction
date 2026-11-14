from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import joblib
from flask import Flask, jsonify, redirect, render_template, request, url_for


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATABASE_PATH = ROOT / "database" / "predictions.db"
MODEL_KEYS = ("linear_regression", "decision_tree", "random_forest")


def connect_database() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            car_details TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    connection = connect_database()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_artifacts() -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    missing = [
        str(MODELS_DIR / f"{key}.pkl")
        for key in MODEL_KEYS
        if not (MODELS_DIR / f"{key}.pkl").is_file()
    ]
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    if not metadata_path.is_file():
        missing.append(str(metadata_path))
    if missing:
        return None, {}, (
            "Trained model files are missing. Add your dataset CSV file(s) to the dataset folder "
            "and run `python train_model.py`."
        )
    try:
        models = {key: joblib.load(MODELS_DIR / f"{key}.pkl") for key in MODEL_KEYS}
        metadata = joblib.load(metadata_path)
    except (OSError, ValueError, EOFError, KeyError) as exc:
        return None, {}, f"Could not load the trained model artifacts: {exc}"
    return models, metadata, None


def model_label(key: str, metadata: dict[str, Any]) -> str:
    if key == "best":
        key = metadata["best_model"]
    return str(metadata["metrics"][key]["name"])


def create_app() -> Flask:
    app = Flask(__name__)
    connect_database().close()
    models, metadata, artifact_error = load_artifacts()
    app.config["MODELS"] = models
    app.config["METADATA"] = metadata
    app.config["ARTIFACT_ERROR"] = artifact_error

    @app.context_processor
    def inject_navigation_state() -> dict[str, Any]:
        return {
            "dataset_ready": app.config["MODELS"] is not None,
            "artifact_error": app.config["ARTIFACT_ERROR"],
        }

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            analysis=metadata.get("analysis", {}),
            metrics=metadata.get("metrics", {}),
            source_file=metadata.get("source_file"),
            best_model=(
                model_label("best", metadata)
                if metadata
                else None
            ),
        )

    @app.get("/predict")
    def predict_page():
        if app.config["MODELS"] is None:
            return redirect(url_for("home"))
        return render_template(
            "predict.html",
            feature_specs=metadata["feature_specs"],
            model_metrics=metadata["metrics"],
            best_model=metadata["best_model"],
        )

    @app.post("/api/predict")
    def predict_api():
        if app.config["MODELS"] is None:
            return jsonify({"error": app.config["ARTIFACT_ERROR"]}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send the prediction details as a JSON object."}), 400
        requested_model = payload.get("model")
        if requested_model == "best":
            selected_model = metadata["best_model"]
        elif requested_model in MODEL_KEYS:
            selected_model = requested_model
        else:
            return jsonify({"error": "Select one of the three trained models or the best model."}), 400

        submitted = payload.get("features")
        if not isinstance(submitted, dict):
            return jsonify({"error": "Car features are missing."}), 400
        specs = metadata["feature_specs"]
        missing_fields = [key for key in specs if key not in submitted]
        if missing_fields:
            return jsonify(
                {"error": f"Complete all car details before predicting: {', '.join(missing_fields)}."}
            ), 400

        values: dict[str, Any] = {}
        for key, spec in specs.items():
            value = submitted[key]
            if spec["kind"] == "number":
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return jsonify({"error": f"{spec['label']} must be a valid number."}), 400
                if not math.isfinite(number):
                    return jsonify({"error": f"{spec['label']} must be a finite number."}), 400
                if spec.get("non_negative") and number < 0:
                    return jsonify({"error": f"{spec['label']} cannot be negative."}), 400
                values[key] = number
            else:
                text = str(value).strip() if value is not None else ""
                if not text:
                    return jsonify({"error": f"{spec['label']} is required."}), 400
                values[key] = text

        try:
            import pandas as pd

            row = pd.DataFrame([values], columns=metadata["feature_columns"])
            amount = float(app.config["MODELS"][selected_model].predict(row)[0])
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": f"The prediction could not be calculated: {exc}"}), 422
        if not math.isfinite(amount):
            return jsonify({"error": "The selected model returned an invalid price."}), 422

        model_name = model_label(selected_model, metadata)
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with database_connection() as connection:
            connection.execute(
                """
                INSERT INTO predictions (created_at, model_key, model_name, predicted_price, car_details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    selected_model,
                    model_name,
                    amount,
                    json.dumps(values, ensure_ascii=False),
                ),
            )
        return jsonify(
            {
                "predicted_price": amount,
                "formatted_price": f"₹{amount:,.0f}",
                "model_key": selected_model,
                "model_name": model_name,
                "created_at": created_at,
            }
        )

    @app.get("/performance")
    def performance_page():
        if app.config["MODELS"] is None:
            return redirect(url_for("home"))
        return render_template(
            "performance.html",
            metrics=metadata["metrics"],
            best_model=metadata["best_model"],
            best_model_name=model_label("best", metadata),
            actual_vs_predicted=metadata["actual_vs_predicted"],
            feature_importance=metadata["feature_importance"],
        )

    @app.get("/analysis")
    def analysis_page():
        if app.config["MODELS"] is None:
            return redirect(url_for("home"))
        return render_template(
            "analysis.html",
            analysis=metadata["analysis"],
            source_file=metadata["source_file"],
            feature_specs=metadata["feature_specs"],
        )

    @app.get("/history")
    def history_page():
        with database_connection() as connection:
            rows = connection.execute(
                "SELECT id, created_at, model_name, predicted_price, car_details "
                "FROM predictions ORDER BY id DESC"
            ).fetchall()
        history = []
        for row in rows:
            item = dict(row)
            try:
                item["car_details"] = json.loads(item["car_details"])
            except json.JSONDecodeError:
                item["car_details"] = {}
            history.append(item)
        return render_template("history.html", predictions=history)

    @app.delete("/api/history")
    def clear_history_api():
        with database_connection() as connection:
            connection.execute("DELETE FROM predictions")
        return jsonify({"message": "Prediction history cleared."})

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error("Unhandled application error: %s", error)
        return render_template(
            "error.html",
            message="Something went wrong while loading this page. Check the application logs for details.",
        ), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
