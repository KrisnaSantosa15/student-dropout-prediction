import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "student_dropout_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "model" / "model_metadata.joblib"


def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}. Jalankan train_model.py terlebih dahulu."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata model tidak ditemukan di {METADATA_PATH}. Jalankan train_model.py terlebih dahulu."
        )

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def predict(input_csv: Path, output_csv: Path, threshold: float) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_csv}")

    model, metadata = load_assets()
    expected_columns = metadata["train_columns"]

    data = pd.read_csv(input_csv)

    missing_cols = [col for col in expected_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(
            "Kolom input belum lengkap. Kolom yang kurang: " + ", ".join(missing_cols))

    scoring_data = data[expected_columns].copy()
    proba = model.predict_proba(scoring_data)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = data.copy()
    result["dropout_prediction"] = pred
    result["dropout_risk_score"] = proba
    result["dropout_risk_label"] = result["dropout_prediction"].map(
        {1: "High Risk", 0: "Low Risk"})

    result.to_csv(output_csv, index=False)
    print(f"Prediksi selesai. Output disimpan di: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prediksi risiko dropout siswa.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Path CSV input untuk prediksi")
    parser.add_argument(
        "--output",
        default=PROJECT_DIR / "prediction_output.csv",
        type=Path,
        help="Path CSV output hasil prediksi",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold klasifikasi dropout (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predict(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
