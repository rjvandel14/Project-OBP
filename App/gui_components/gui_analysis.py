import streamlit as st
from routing import all_crvp

def render_analysis(vehicle_capacity, cost_per_km, fixed_cost_per_truck, ranking_data):
    """Handles company selection and performs collaboration analysis."""
    # Dropdowns for company selection
    unique_companies = ranking_data["Company A"].unique()
    placeholder_companies = ["Select a company", *unique_companies]

    st.subheader("Select Companies for Detailed Analysis")
    company_a = st.selectbox("Select Company A", placeholder_companies, index=0)
    company_b = st.selectbox("Select Company B", placeholder_companies, index=0)

    # Analyze collaboration
    if st.button("Analyze Collaboration"):
        if company_a == "Select a company" or company_b == "Select a company":
            st.error("Please select valid companies for both dropdowns.")
        elif company_a == company_b:
            st.error("Please select two different companies.")
        else:
            # Mock collaboration analysis
            results = all_crvp(vehicle_capacity, cost_per_km, fixed_cost_per_truck, company_a, company_b)
            cost_a = results["Cost (€)"][0]
            cost_b = results["Cost (€)"][1]
            cost_collab = results["Cost (€)"][2]

            # Display the results
            st.subheader("Analysis Results")
            st.write(f"Cost for {company_a}: {cost_a}")
            st.write(f"Cost for {company_b}: {cost_b}")
            st.write(f"Cost for collaboration: {cost_collab}")
            st.write(f"Total savings: {cost_a + cost_b - cost_collab}")
