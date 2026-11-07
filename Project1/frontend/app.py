
import os
import requests
import streamlit as st
import pandas as pd


# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="GitHub Repo Classifier",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================
# 2. API CONFIGURATION
# =====================================

# Check Streamlit secrets first (for Streamlit Cloud deployment),
# then environment variable, then fall back to localhost
try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# =====================================
# 3. API HELPERS (cached)
# =====================================

@st.cache_data(ttl=60)
def fetch_metrics():
    r = requests.get(f"{API_URL}/metrics", timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def fetch_features():
    r = requests.get(f"{API_URL}/features", timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def fetch_sample():
    r = requests.get(f"{API_URL}/sample", timeout=10)
    r.raise_for_status()
    return r.json()

def api_post(endpoint, payload):
    r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# =====================================
# 4. SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("## :material/hub: GitHub Classifier")
    st.caption("Random Forest · ML System")

    st.markdown("---")

    # Backend health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        health_data = health.json()
        if health_data.get("status") == "healthy":
            st.badge("Backend online", icon=":material/check_circle:", color="green")
        else:
            st.badge("Backend unhealthy", icon=":material/error:", color="red")
    except requests.RequestException:
        st.badge("Backend offline", icon=":material/wifi_off:", color="red")
        st.error(
            "FastAPI backend is not reachable. Start it with:",
            icon=":material/error:"
        )
        st.code(
            "cd Project1/backend\n"
            "python -m uvicorn main:app --reload"
        )
        st.stop()

    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=["Dashboard", "Prediction", "Model Info"],
        format_func=lambda x: {
            "Dashboard":   ":material/dashboard: Dashboard",
            "Prediction":  ":material/model_training: Prediction",
            "Model Info":  ":material/analytics: Model Info",
        }[x],
    )

    st.markdown("---")
    st.caption("Built with FastAPI · Streamlit · scikit-learn")


# =====================================
# 5. LOAD MODEL DATA
# =====================================

try:
    metrics     = fetch_metrics()
    feature_data = fetch_features()
    sample_data  = fetch_sample()
    feature_names = feature_data["features"]
    sample        = sample_data["sample"]
except requests.RequestException as e:
    st.error(f"Failed to load model data: {e}", icon=":material/error:")
    st.stop()


# =====================================
# 6. DASHBOARD PAGE
# =====================================

if page == "Dashboard":

    st.title("GitHub Repo popularity classifier")
    st.caption(
        "Predict whether a GitHub repository falls into the "
        "**Established**, **High**, or **Mega** popularity tier — "
        "powered by a Random Forest model trained on real repository signals."
    )

    st.space("medium")

    # --- KPI row ---
    with st.container(horizontal=True):
        st.metric(
            "Model accuracy",
            f"{metrics['accuracy_percent']}%",
            border=True,
        )
        st.metric(
            "Input features",
            metrics["feature_count"],
            border=True,
        )
        st.metric(
            "Algorithm",
            "Random Forest",
            border=True,
        )
        st.metric(
            "Output classes",
            "3",
            border=True,
        )

    st.space("medium")

    # --- Two column explainer ---
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        with st.container(border=True, height="stretch"):
            st.markdown("#### :material/hub: What this app does")
            st.write(
                "Given key GitHub repository signals — such as stars, forks, "
                "open issues, repo age, and community health — this model "
                "classifies the repository into one of three popularity tiers."
            )
            st.space("small")
            st.markdown(
                ":green-badge[Established (100–1 k)] &nbsp; "
                ":orange-badge[High (1 k–10 k)] &nbsp; "
                ":red-badge[Mega (10 k+)]"
            )

    with col_b:
        with st.container(border=True, height="stretch"):
            st.markdown("#### :material/input: Input signals")
            feature_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Description": [
                        "GitHub star count",
                        "Number of forks",
                        "Open issues count",
                        "Repository size (KB)",
                        "Repository age (days)",
                        "Days since last push",
                        "Number of releases",
                        "GitHub community health %",
                    ],
                }
            )
            st.dataframe(feature_df, hide_index=True)

    st.space("medium")

    # --- How it works ---
    with st.container(border=True):
        st.markdown("#### :material/route: How a prediction is made")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.info("**1. Enter repo stats**\nFill in the 8 feature fields on the Prediction page.", icon=":material/edit:")
        with c2:
            st.info("**2. Send to API**\nThe UI posts the values to the FastAPI `/predict` endpoint.", icon=":material/send:")
        with c3:
            st.info("**3. Model scores**\nThe Random Forest evaluates 100 decision trees.", icon=":material/forest:")
        with c4:
            st.success("**4. Result returned**\nThe predicted tier + class probabilities are displayed.", icon=":material/check_circle:")


# =====================================
# 7. PREDICTION PAGE
# =====================================

elif page == "Prediction":

    st.title("Make a prediction")
    st.caption(
        "Enter the repository metrics below. "
        "Values are pre-filled from a real sample in the dataset."
    )

    st.space("small")

    with st.form("prediction_form", border=True):

        st.markdown("#### :material/input: Repository features")

        input_features = {}
        col_left, col_right = st.columns(2, gap="large")

        for idx, feature in enumerate(feature_names):
            value = sample.get(feature)
            target_col = col_left if idx % 2 == 0 else col_right

            with target_col:
                if isinstance(value, bool):
                    input_features[feature] = st.checkbox(
                        feature.replace("_", " ").capitalize(),
                        value=value,
                        key=f"feat_{idx}",
                    )
                elif isinstance(value, (int, float)):
                    input_features[feature] = st.number_input(
                        feature.replace("_", " ").capitalize(),
                        value=float(value),
                        key=f"feat_{idx}",
                    )
                else:
                    input_features[feature] = st.text_input(
                        feature.replace("_", " ").capitalize(),
                        value=str(value),
                        key=f"feat_{idx}",
                    )

        st.space("small")

        submitted = st.form_submit_button(
            ":material/model_training: Run prediction",
            type="primary",
        )

    # --- Result ---
    if submitted:
        with st.spinner("Running model inference…"):
            try:
                result = api_post("/predict", {"features": input_features})
            except requests.HTTPError as exc:
                try:
                    detail = exc.response.json()
                except Exception:
                    detail = str(exc)
                st.error(f"Prediction failed: {detail}", icon=":material/error:")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"API connection error: {exc}", icon=":material/wifi_off:")
                st.stop()

        label = result["predicted_label"]
        probabilities = result.get("probabilities", {})

        # Colour-coded badge per tier
        badge_color = {
            "Established (100-1k)": "green",
            "High (1k-10k)":        "orange",
            "Mega (10k+)":          "red",
        }.get(label, "blue")

        st.success("Prediction complete!", icon=":material/check_circle:")
        st.space("small")

        res_col, prob_col = st.columns([1, 2], gap="large")

        with res_col:
            with st.container(border=True):
                st.markdown("#### :material/emoji_events: Predicted tier")
                st.space("small")
                st.badge(label, color=badge_color, icon=":material/hub:")
                st.space("small")
                st.markdown(f"Raw class: `{result['prediction']}`")
                st.caption("Based on Random Forest majority vote across 100 trees.")

        with prob_col:
            if probabilities:
                with st.container(border=True):
                    st.markdown("#### :material/bar_chart: Class probabilities")
                    prob_df = (
                        pd.DataFrame(
                            list(probabilities.items()),
                            columns=["Tier", "Probability (%)"],
                        )
                        .sort_values("Probability (%)", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.bar_chart(
                        prob_df.set_index("Tier"),
                        color="#60A5FA",
                    )
                    st.dataframe(prob_df, hide_index=True)

        with st.expander("View raw API response", icon=":material/code:"):
            st.json(result)


# =====================================
# 8. MODEL INFO PAGE
# =====================================

elif page == "Model Info":

    st.title("Model information")
    st.caption("Technical details about the trained Random Forest classifier.")

    st.space("small")

    top_col, _ = st.columns([2, 1])
    with top_col:
        with st.container(border=True):
            st.markdown("#### :material/analytics: Performance snapshot")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", f"{metrics['accuracy_percent']}%")
            with m2:
                st.metric("Algorithm", metrics["algorithm"])
            with m3:
                st.metric("Features", metrics["feature_count"])

    st.space("medium")

    info_col, feat_col = st.columns(2, gap="medium")

    with info_col:
        with st.container(border=True, height="stretch"):
            st.markdown("#### :material/forest: About Random Forest")
            st.write(
                "A Random Forest Classifier builds an ensemble of decision "
                "trees during training and outputs the class chosen by the "
                "majority of trees. It is robust to overfitting, handles "
                "non-linear relationships, and natively provides class "
                "probability estimates."
            )
            st.space("small")
            st.markdown("**Target labels**")
            for lbl in ["Established (100-1k)", "High (1k-10k)", "Mega (10k+)"]:
                color = {"Established (100-1k)": "green", "High (1k-10k)": "orange", "Mega (10k+)": "red"}[lbl]
                st.badge(lbl, color=color)

    with feat_col:
        with st.container(border=True, height="stretch"):
            st.markdown("#### :material/list: Feature reference")
            feat_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Type": ["Numeric"] * len(feature_names),
                }
            )
            st.dataframe(feat_df, hide_index=True)

    st.space("medium")

    with st.container(border=True):
        st.markdown("#### :material/code: Full metrics payload")
        st.json(metrics)


# =====================================
# 9. FOOTER
# =====================================

st.markdown("---")
st.caption(
    ":material/code: Built with **Python** · **FastAPI** · **Streamlit** · **scikit-learn**"
)
