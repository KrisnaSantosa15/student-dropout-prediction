from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "student_dropout_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "model" / "model_metadata.joblib"
DASHBOARD_DIR = PROJECT_DIR / "dashboard_data"

COURSE_LABELS = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (Evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening attendance)",
}

GENDER_LABELS = {
    0: "Female",
    1: "Male",
}

BINARY_STATUS_LABELS = {
    0: "No",
    1: "Yes",
}

TUITION_STATUS_LABELS = {
    0: "Not up to date",
    1: "Up to date",
}

RISK_LEVELS = ["Low", "Medium", "High", "Very High"]


@st.cache_data
def load_labeled_data() -> pd.DataFrame:
    file_path = DASHBOARD_DIR / "student_labeled_data.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            "Data dashboard belum tersedia. Jalankan train_model.py terlebih dahulu."
        )
    return pd.read_csv(file_path)


@st.cache_resource
def load_model_assets():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan train_model.py terlebih dahulu."
        )
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def to_binary_dropout(status_series: pd.Series) -> pd.Series:
    return (status_series == "Dropout").astype(int)


def score_to_risk_level(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def _code_to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def add_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    course_code = _code_to_int(result["Course"])
    gender_code = _code_to_int(result["Gender"])
    scholarship_code = _code_to_int(result["Scholarship_holder"])
    tuition_code = _code_to_int(result["Tuition_fees_up_to_date"])

    result["Course_label"] = course_code.map(COURSE_LABELS).fillna(
        course_code.map(lambda x: f"Course {x}" if pd.notna(x) else "Unknown")
    )
    result["Gender_label"] = gender_code.map(GENDER_LABELS).fillna(
        gender_code.map(lambda x: f"Gender {x}" if pd.notna(x) else "Unknown")
    )
    result["Scholarship_holder_label"] = scholarship_code.map(BINARY_STATUS_LABELS).fillna(
        scholarship_code.map(lambda x: str(x) if pd.notna(x) else "Unknown")
    )
    result["Tuition_fees_up_to_date_label"] = tuition_code.map(TUITION_STATUS_LABELS).fillna(
        tuition_code.map(lambda x: str(x) if pd.notna(x) else "Unknown")
    )

    return result


def render_kpi(df: pd.DataFrame) -> None:
    total = len(df)
    dropout_count = int(df["dropout_target"].sum())
    dropout_rate = (dropout_count / total * 100) if total > 0 else 0.0
    non_dropout = total - dropout_count

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Siswa", f"{total:,}")
    c2.metric("Jumlah Dropout", f"{dropout_count:,}")
    c3.metric("Jumlah Non-Dropout", f"{non_dropout:,}")
    c4.metric("Dropout Rate", f"{dropout_rate:.2f}%")


def render_charts(df: pd.DataFrame) -> None:
    st.subheader("Analisis Segmentasi Dropout")

    left, right = st.columns(2)

    by_course = (
        df.groupby("Course_label", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
        .head(10)
    )
    by_course["dropout_rate"] = by_course["dropout_rate"] * 100

    by_gender = (
        df.groupby("Gender_label", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
    )
    by_gender["dropout_rate"] = by_gender["dropout_rate"] * 100

    with left:
        st.markdown("**Top 10 Course Dengan Dropout Rate Tertinggi**")
        st.bar_chart(by_course.set_index("Course_label")[["dropout_rate"]])

    with right:
        st.markdown("**Dropout Rate Berdasarkan Gender**")
        st.bar_chart(by_gender.set_index("Gender_label")[["dropout_rate"]])

    age_group = (
        df.groupby("age_group", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
    )
    age_group["dropout_rate"] = age_group["dropout_rate"] * 100

    tuition = (
        df.groupby("Tuition_fees_up_to_date_label", as_index=False)
        .agg(total_student=("dropout_target", "size"), dropout_rate=("dropout_target", "mean"))
        .sort_values("dropout_rate", ascending=False)
    )
    tuition["dropout_rate"] = tuition["dropout_rate"] * 100

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("**Dropout Rate Berdasarkan Kelompok Usia**")
        st.bar_chart(age_group.set_index("age_group")[["dropout_rate"]])
    with right2:
        st.markdown("**Dropout Rate Berdasarkan Status Pembayaran**")
        st.bar_chart(tuition.set_index(
            "Tuition_fees_up_to_date_label")[["dropout_rate"]])


def render_risk_table(df: pd.DataFrame) -> None:
    st.subheader("Siswa Dengan Risiko Dropout Tertinggi")
    top_risk_cols = [
        "Course_label",
        "Age_at_enrollment",
        "Gender_label",
        "Tuition_fees_up_to_date_label",
        "dropout_risk_score",
        "Status",
    ]
    show_df = (
        df[top_risk_cols]
        .sort_values("dropout_risk_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    show_df = show_df.rename(
        columns={
            "Course_label": "Course",
            "Gender_label": "Gender",
            "Tuition_fees_up_to_date_label": "Tuition Fees Up To Date",
            "dropout_risk_score": "Risk Score",
        }
    )
    st.dataframe(show_df)


def build_default_row(df: pd.DataFrame, expected_columns: list[str]) -> dict:
    base = {}
    for col in expected_columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            base[col] = float(series.median())
        else:
            mode = series.mode(dropna=True)
            base[col] = mode.iloc[0] if not mode.empty else ""
    return base


def render_single_prediction(
    model,
    expected_columns: list[str],
    labeled_data: pd.DataFrame,
    threshold: float,
) -> None:
    st.markdown("### Prediksi Individual")
    defaults = build_default_row(labeled_data, expected_columns)

    with st.form("single_prediction_form"):
        input_df = pd.DataFrame([defaults])[expected_columns]
        edited_df = st.data_editor(
            input_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
        )
        submitted = st.form_submit_button("Prediksi Risiko")

    if not submitted:
        return

    scoring_df = edited_df[expected_columns].copy()
    for col in expected_columns:
        if pd.api.types.is_numeric_dtype(labeled_data[col]):
            scoring_df[col] = pd.to_numeric(scoring_df[col], errors="coerce")

    invalid_cols = scoring_df.columns[scoring_df.isna().any()].tolist()
    if invalid_cols:
        st.error(
            "Terdapat nilai kosong/tidak valid pada kolom: "
            + ", ".join(invalid_cols)
            + "."
        )
        return

    row = scoring_df.iloc[0].to_dict()
    score = float(model.predict_proba(scoring_df)[:, 1][0])
    pred = int(score >= threshold)
    risk_level = score_to_risk_level(score)

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Score", f"{score:.2%}")
    m2.metric("Risk Level", risk_level)
    m3.metric("Prediction", "Dropout Risk" if pred ==
              1 else "Non-Dropout Risk")

    if pred == 1:
        st.error(
            "Model menandai siswa ini sebagai berisiko dropout. Prioritaskan intervensi akademik.")
    else:
        st.success(
            "Model menandai siswa ini sebagai non-dropout risk pada threshold saat ini.")

    recs = []
    if row["Tuition_fees_up_to_date"] == 0:
        recs.append(
            "Follow-up status pembayaran dan berikan rencana dukungan finansial.")
    if row["Curricular_units_1st_sem_grade"] < 11 or row["Curricular_units_2nd_sem_grade"] < 11:
        recs.append(
            "Aktifkan pendampingan akademik pada mata kuliah inti semester berjalan.")
    if row["Debtor"] == 1:
        recs.append(
            "Koordinasikan bantuan administrasi untuk isu tunggakan/non-akademik.")

    if recs:
        st.markdown("**Rekomendasi otomatis:**")
        for item in recs:
            st.write(f"- {item}")


def render_batch_prediction(
    model,
    expected_columns: list[str],
    threshold: float,
) -> None:
    st.markdown("### Prediksi Batch CSV")
    uploaded_file = st.file_uploader(
        "Upload file CSV untuk prediksi", type=["csv"])
    if uploaded_file is None:
        return

    incoming = pd.read_csv(uploaded_file)
    missing_cols = [
        col for col in expected_columns if col not in incoming.columns]
    if missing_cols:
        st.error("Kolom input belum lengkap: " + ", ".join(missing_cols))
        return

    scored = incoming.copy()
    scored_input = scored[expected_columns]
    scored["dropout_risk_score"] = model.predict_proba(scored_input)[:, 1]
    scored["dropout_prediction"] = (
        scored["dropout_risk_score"] >= threshold).astype(int)
    scored["dropout_risk_label"] = scored["dropout_risk_score"].apply(
        score_to_risk_level)

    total = len(scored)
    high_risk = int((scored["dropout_prediction"] == 1).sum())
    high_risk_rate = (high_risk / total * 100) if total else 0.0

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Data Scored", f"{total:,}")
    s2.metric("Predicted Dropout Risk", f"{high_risk:,}")
    s3.metric("Risk Rate", f"{high_risk_rate:.2f}%")

    level_dist = (
        scored["dropout_risk_label"]
        .value_counts()
        .reindex(RISK_LEVELS, fill_value=0)
        .rename_axis("Risk Level")
        .to_frame("Count")
    )
    st.markdown("**Distribusi Risk Level**")
    st.bar_chart(level_dist)

    st.markdown("**Preview Hasil Prediksi**")
    st.dataframe(scored.head(20))
    st.download_button(
        label="Download Hasil Prediksi",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="prediction_output.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Dashboard Monitoring Risiko Dropout Siswa")
    st.caption("Jaya Jaya Institut - Early Warning Analytics")

    try:
        labeled_data = load_labeled_data()
        model, metadata = load_model_assets()
    except (FileNotFoundError, OSError, ValueError) as err:
        st.error(str(err))
        st.stop()

    labeled_data["dropout_target"] = to_binary_dropout(labeled_data["Status"])
    labeled_data = add_display_labels(labeled_data)

    st.sidebar.header("Filter Data")
    course_options = sorted(
        labeled_data["Course_label"].dropna().unique().tolist())
    gender_options = sorted(
        labeled_data["Gender_label"].dropna().unique().tolist())
    scholarship_options = sorted(
        labeled_data["Scholarship_holder_label"].dropna().unique().tolist())
    tuition_options = sorted(
        labeled_data["Tuition_fees_up_to_date_label"].dropna().unique().tolist())

    selected_course = st.sidebar.multiselect(
        "Course", course_options, default=course_options)
    selected_gender = st.sidebar.multiselect(
        "Gender", gender_options, default=gender_options)
    selected_scholarship = st.sidebar.multiselect(
        "Scholarship Holder", scholarship_options, default=scholarship_options
    )
    selected_tuition = st.sidebar.multiselect(
        "Tuition Fees Up To Date", tuition_options, default=tuition_options
    )

    filtered = labeled_data[
        labeled_data["Course_label"].isin(selected_course)
        & labeled_data["Gender_label"].isin(selected_gender)
        & labeled_data["Scholarship_holder_label"].isin(selected_scholarship)
        & labeled_data["Tuition_fees_up_to_date_label"].isin(selected_tuition)
    ].copy()

    if filtered.empty:
        st.warning("Tidak ada data yang sesuai dengan kombinasi filter saat ini.")
        st.stop()

    expected_columns = metadata["train_columns"]
    filtered["dropout_risk_score"] = model.predict_proba(
        filtered[expected_columns])[:, 1]

    tab_dashboard, tab_prediction = st.tabs(
        ["Monitoring Dashboard", "Prediction Center"]
    )

    with tab_dashboard:
        render_kpi(filtered)
        render_charts(filtered)
        render_risk_table(filtered)

    with tab_prediction:
        st.markdown("### Konfigurasi Prediksi")
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Semakin rendah threshold, model semakin sensitif menandai siswa berisiko.",
        )
        p1, p2 = st.columns([1, 1])
        with p1:
            render_single_prediction(
                model, expected_columns, labeled_data, threshold)
        with p2:
            render_batch_prediction(model, expected_columns, threshold)


if __name__ == "__main__":
    main()
