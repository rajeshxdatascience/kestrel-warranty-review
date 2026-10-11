
import requests
import streamlit as st

st.set_page_config(page_title="Kestrel Claim Review", page_icon="🔎")

st.title("Kestrel Home — Warranty Claim Review")
st.caption("Decision support prototype — not an automatic fraud verdict.")

with st.form("claim_form"):
    claim_id = st.text_input("Claim ID", "DEMO-001")
    submitted_at = st.text_input(
        "Submission date and time", "2026-07-15 10:30:00"
    )
    partner_id = st.text_input("Partner ID", "SP3150")
    sku = st.text_input("Product SKU", "KH-RV-03")
    serial = st.text_input("Product serial", "KH222151083")

    days = st.number_input("Days since purchase", 0, 5000, 60)
    amount = st.number_input("Claim amount (INR)", 1.0, 100000.0, 2500.0)
    prior = st.number_input("Customer prior claims", 0, 100, 0)

    photo = st.selectbox("Photo attached?", ["Y", "N"])
    inspected = st.selectbox("Partner inspected?", ["Y", "N"])
    source = st.selectbox("Source", ["crm", "legacy_zoho"])
    description = st.text_area("Claim description", "Product stopped working.")

    submitted = st.form_submit_button("Assess claim")

if submitted:
    payload = {
        "claim_id": claim_id,
        "submitted_at": submitted_at,
        "partner_id": partner_id,
        "sku": sku,
        "product_serial": serial,
        "days_since_purchase": days,
        "claim_amount_inr": amount,
        "photo_attached": photo,
        "partner_inspected": inspected,
        "claim_description": description,
        "customer_prior_claims": prior,
        "source": source,
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        st.metric("Model risk score", f"{result['risk_score']:.1%}")
        st.info(result["interpretation"])
        st.subheader("Review notes")

        for reason in result["reasons_to_review"]:
            st.write(f"- {reason}")

    except requests.RequestException:
        st.error(
            "API is unavailable. Start the FastAPI server and try again."
        )
