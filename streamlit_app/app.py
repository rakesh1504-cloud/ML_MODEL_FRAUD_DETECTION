import requests
import streamlit as st

API_URL = "https://fraud-detection-api.onrender.com"  # update after Render deploy

st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🔍",
    layout="centered",
)

st.title("Real-Time Fraud Detection")
st.markdown(
    "Enter transaction details below. The model scores **fraud probability** using "
    "**LightGBM** trained on 590k IEEE-CIS transactions, with **SHAP** explanations "
    "for every decision."
)
st.divider()

# --- Sidebar: model info ---
with st.sidebar:
    st.header("Model Info")
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        if r.status_code == 200:
            info = r.json()
            st.metric("AUC-PR", f"{info.get('auc_pr', 'N/A')}")
            st.metric("Recall", f"{info.get('recall', 'N/A'):.0%}" if info.get("recall") else "N/A")
            st.metric("Threshold", f"{info.get('threshold', 0.5):.3f}")
            st.metric("Features", info.get("n_features", "N/A"))
            st.caption(f"Model: {info.get('model_type', 'LightGBM')}")
        else:
            st.warning("API unreachable — enter your Render URL above.")
    except Exception:
        st.warning("API not reachable. Start the API locally or deploy to Render.")

# --- Input form ---
col1, col2 = st.columns(2)
with col1:
    amount    = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50_000.0, value=250.0, step=10.0)
    card_type = st.selectbox("Card Network", ["visa", "mastercard", "discover", "american express"])
with col2:
    card_cat  = st.selectbox("Card Category", ["debit", "credit", "charge card"])
    email_dom = st.selectbox("Purchaser Email Domain", ["gmail.com", "yahoo.com", "outlook.com", "anonymous.com", "other"])

device   = st.selectbox("Device Type", ["desktop", "mobile", "unknown (missing)"])
device_v = None if device == "unknown (missing)" else device

st.divider()

if st.button("Analyse Transaction", type="primary", use_container_width=True):
    payload = {
        "TransactionAmt": amount,
        "card4":          card_type,
        "card6":          card_cat,
        "P_emaildomain":  email_dom,
        "DeviceType":     device_v,
    }

    with st.spinner("Scoring transaction..."):
        try:
            r   = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            res = r.json()

            prob     = res["fraud_probability"]
            is_fraud = res["is_fraud"]
            risk     = res["risk_level"]

            # --- Verdict ---
            if risk == "CRITICAL":
                st.error(f"🚨 FRAUD DETECTED — Risk: {risk}")
            elif is_fraud:
                st.warning(f"⚠️ LIKELY FRAUD — Risk: {risk}")
            else:
                st.success(f"✅ Transaction Cleared — Risk: {risk}")

            # --- Metrics ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Probability", f"{prob:.1%}")
            m2.metric("Decision", "🔴 BLOCK" if is_fraud else "🟢 ALLOW")
            m3.metric("Risk Level", risk)

            # --- SHAP explanation ---
            st.subheader("Why this decision? (SHAP)")
            st.caption("Positive values push toward fraud; negative values reduce fraud score.")
            top = res.get("top_shap_features", {})
            if top:
                for feat, val in top.items():
                    direction = "↑ pushed toward fraud" if val > 0 else "↓ reduced fraud score"
                    colour    = "🔴" if val > 0 else "🟢"
                    st.write(f"{colour} **{feat}** &nbsp; `{val:+.4f}` — {direction}")
            else:
                st.info("SHAP explanations not available (model may not support TreeExplainer).")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure it is running or update the API_URL.")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption(
    "LightGBM · IEEE-CIS 590k transactions · AUC-PR primary metric · "
    "Built by Rakesh Kumar Dubey"
)
