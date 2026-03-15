from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "students_performance.csv"
MODEL_DIR = PROJECT_DIR / "model"
DASHBOARD_DIR = PROJECT_DIR / "dashboard_data"

DATA_URLS = [
    "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv",
    "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv",
]


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, sep=";")

    for url in DATA_URLS:
        try:
            df = pd.read_csv(url, sep=";")
            df.to_csv(DATA_PATH, sep=";", index=False)
            return df
        except (OSError, ValueError, pd.errors.ParserError):
            continue

    raise FileNotFoundError(
        "Dataset tidak ditemukan secara lokal dan gagal diunduh dari sumber resmi."
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_pipeline(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def prepare_dashboard_data(df: pd.DataFrame, model: Pipeline, train_columns: list[str]) -> None:
    DASHBOARD_DIR.mkdir(exist_ok=True)

    dashboard_df = df.copy()
    dashboard_df["dropout_target"] = (
        dashboard_df["Status"] == "Dropout").astype(int)
    dashboard_df["dropout_risk_score"] = model.predict_proba(
        dashboard_df[train_columns])[:, 1]

    dashboard_df["age_group"] = pd.cut(
        dashboard_df["Age_at_enrollment"],
        bins=[0, 20, 25, 30, 100],
        labels=["<=20", "21-25", "26-30", ">30"],
        include_lowest=True,
    )

    dashboard_df.to_csv(
        DASHBOARD_DIR / "student_labeled_data.csv", index=False)

    pd.DataFrame(
        {
            "metric": [
                "total_student",
                "dropout_count",
                "non_dropout_count",
                "dropout_rate",
            ],
            "value": [
                len(dashboard_df),
                int(dashboard_df["dropout_target"].sum()),
                int((dashboard_df["dropout_target"] == 0).sum()),
                float(dashboard_df["dropout_target"].mean()),
            ],
        }
    ).to_csv(DASHBOARD_DIR / "kpi_summary.csv", index=False)

    (
        dashboard_df.groupby("Course", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
        .to_csv(DASHBOARD_DIR / "dropout_by_course.csv", index=False)
    )

    (
        dashboard_df.groupby("Gender", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
        .to_csv(DASHBOARD_DIR / "dropout_by_gender.csv", index=False)
    )

    (
        dashboard_df.groupby("Scholarship_holder", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
        .to_csv(DASHBOARD_DIR / "dropout_by_scholarship.csv", index=False)
    )

    (
        dashboard_df.groupby("Tuition_fees_up_to_date", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
        .to_csv(DASHBOARD_DIR / "dropout_by_tuition_status.csv", index=False)
    )

    (
        dashboard_df.groupby("age_group", as_index=False, observed=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
        .to_csv(DASHBOARD_DIR / "dropout_by_age_group.csv", index=False)
    )


def main() -> None:
    df = load_data()

    labeled_df = df[df["Status"].isin(
        ["Dropout", "Graduate", "Enrolled"])].copy()
    labeled_df["dropout_target"] = (
        labeled_df["Status"] == "Dropout").astype(int)

    x = labeled_df.drop(columns=["Status", "dropout_target"])
    y = labeled_df["dropout_target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(x)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        ),
    }

    best_name = None
    best_metrics = None
    best_pipeline = None
    best_score = (-1.0, -1.0)

    for name, model in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate_pipeline(pipeline, x_test, y_test)

        current_score = (metrics["roc_auc"], metrics["f1_score"])
        if current_score > best_score:
            best_name = name
            best_metrics = metrics
            best_pipeline = pipeline
            best_score = current_score

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "student_dropout_pipeline.joblib")
    joblib.dump(
        {
            "selected_model": best_name,
            "metrics": best_metrics,
            "train_columns": x_train.columns.tolist(),
            "target_distribution": y_train.value_counts().to_dict(),
            "target_definition": "1=Dropout, 0=Non-Dropout (Graduate/Enrolled)",
            "data_source": DATA_URLS[0],
        },
        MODEL_DIR / "model_metadata.joblib",
    )

    prepare_dashboard_data(labeled_df.drop(
        columns=["dropout_target"]), best_pipeline, x_train.columns.tolist())

    print("Model terbaik:", best_name)
    print("Metrik evaluasi:", best_metrics)
    print("Model tersimpan di:", MODEL_DIR)
    print("Data dashboard tersimpan di:", DASHBOARD_DIR)


if __name__ == "__main__":
    main()
