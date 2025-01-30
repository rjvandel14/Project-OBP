import pandas as pd
import streamlit as st
from routing import all_cvrp
from ranking_functions.ranking_clustering import get_cluster_kmeans


def render_ranking(dmatrix, data, vehicle_capacity, cost_per_km, fixed_cost_per_truck):
    ranking_data = get_cluster_kmeans(data, dmatrix)

    # Create two columns: one for the title and the other for the filter options
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.subheader("Ranked Collaborations")

    # Display the filter options in the second column
    with col2:
        companies = sorted(sorted(data["name"].unique()))
        selected_company = st.selectbox("Select a company to filter", ["All"] + companies)
        if selected_company != "All":
            ranking_data = ranking_data[
                (ranking_data['Company A'] == selected_company) | 
                (ranking_data['Company B'] == selected_company)
            ]

    # Initialize session state variables
    if "rows_to_display" not in st.session_state:
        st.session_state.rows_to_display = 5  # Start with the top 5 rows
    if "click_count" not in st.session_state:
        st.session_state.click_count = 0  # Initialize click_count
    if "toggle_states" not in st.session_state:
        st.session_state.toggle_states = {}
    if "results" not in st.session_state:
        st.session_state.results = {}

    # Generate a hash for the current dataset
    current_data_hash = hash(pd.util.hash_pandas_object(ranking_data).sum())

    # Reset states if the dataset changes
    if (
        "current_data_hash" not in st.session_state
        or st.session_state.current_data_hash != current_data_hash
    ):
        st.session_state.current_data_hash = current_data_hash
        st.session_state.rows_to_display = 5
        st.session_state.click_count = 0
        st.session_state.toggle_states = {index: False for index in ranking_data.index}
        st.session_state.results = {}

    # Decide how many rows to display
    rows_to_display = ranking_data.head(st.session_state.rows_to_display)

    # Show headers
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5]) 

    with col1:
        st.markdown("**Rank**")

    with col2:
        st.markdown("**First company**")

    with col3:
        st.markdown("**Second company**")

    with col4:
        st.markdown(
            """
            <style>
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                font-weight: 600; /* Match Streamlit header font weight */
                text-align: center;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 250px; /* Define the box width */
                background-color: rgba(50, 50, 50, 0.9); /* Dark background for contrast */
                color: #fff; /* White text for readability */
                text-align: left; /* Align text to the left inside the box */
                border-radius: 5px; /* Rounded corners for a modern look */
                padding: 10px; /* Add padding for spacing */
                position: absolute;
                z-index: 1;
                top: 130%; /* Position below the header */
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s ease;
                font-size: 12px; /* Smaller font size for clarity */
                line-height: 1.6; /* Add spacing between lines */
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                white-space: normal; /* Allow text wrapping for multi-line */
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            </style>
            <div class="tooltip">
                Short Analysis
                <span class="tooltiptext">
                    This option allows you to set a time limit for calculations. 
                    By enabling it, the system will use faster methods to provide approximate results, 
                    which can save time during analysis but may reduce precision.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border: 1px solid #ccc; margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    # Loop through rows to show data
    for index, row in rows_to_display.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5])  # Adjust column widths

        with col1:
            st.write(f"{row['Rank']}")  # Display the rank

        with col2:
            st.write(f"{row['Company A']}")  # First company 

        with col3:
            st.write(f"{row['Company B']}")  # Second company

        with col4:
            # Analyze button
            if st.button(f"Analyze {index + 1}", key=f"analyze_{index}"):
                if index not in st.session_state.results:
                    # Run analysis inside the expander
                    st.session_state.toggle_states[index] = True

        # Show analysis only if the user clicks "Analyze"
        if st.session_state.toggle_states.get(index, False):
            with st.expander(f"Analysis for {row['Company A']} ↔ {row['Company B']}", expanded=True):
                
                # If no results exist yet, show the spinner
                if index not in st.session_state.results:
                    with st.spinner(f"Analyzing collaboration between {row['Company A']} and {row['Company B']}..."):
                        timelimit = 10 + 0.5 * len(data[data['name'].isin([row['Company A'], row['Company B']])].copy())
                        results = all_cvrp(
                            vehicle_capacity,
                            cost_per_km,
                            fixed_cost_per_truck,
                            row["Company A"],
                            row["Company B"],
                            data,
                            dmatrix,
                            timelimit,
                        )
                        st.session_state.results[index] = results  # Save results persistently
                
                # Retrieve and display results
                results = st.session_state.results.get(index)
                if results:
                    st.subheader(f"Analysis Results for {row['Company A']} ↔ {row['Company B']}")

                    # Prepare data for the table
                    analysis_data = {
                        "Category": [row["Company A"], row["Company B"], "Collaboration"],
                        "Total Costs (€)": [results["Total Cost"][0], results["Total Cost"][1], results["Total Cost"][2]],
                        "Fixed Truck Costs (€)": [results["Truck Cost"][0], results["Truck Cost"][1], results["Truck Cost"][2]],
                        "Kilometer Costs (€)": [results["Driving Cost"][0], results["Driving Cost"][1], results["Driving Cost"][2]]
                    }

                    # Convert to DataFrame and format
                    df = pd.DataFrame(analysis_data)
                    df.set_index("Category", inplace=True)  # Remove default numeric index
                    df = df.style.format({
                        "Total Costs (€)": "{:.2f}", 
                        "Fixed Truck Costs (€)": "{:.2f}", 
                        "Kilometer Costs (€)": "{:.2f}"
                    })

                    # Display the table
                    st.table(df)

                    # Calculate and display total savings separately
                    total_savings = results["Total Cost"][0] + results["Total Cost"][1] - results["Total Cost"][2]
                    st.markdown(f"**:violet[Total savings: €{total_savings:.2f}]**")

        st.markdown("<hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

    # Callback to handle "Show More" button
    def show_more_callback():
        st.session_state.click_count += 1  # Add to click count

        if st.session_state.click_count == 1:
            st.session_state.rows_to_display += 5  # Add 5 rows after first click
        elif st.session_state.click_count == 2:
            st.session_state.rows_to_display += 40  # Add 40 rows adter second click
        else:
            st.session_state.rows_to_display += 50  # Add 50 rows after

    # Create a two-column layout for buttons
    col1, col2 = st.columns([4, 2]) 

    # Show the "Show More" button only if there are more rows to display
    with col1:
        if len(ranking_data) > st.session_state.rows_to_display:
            st.button(":violet[Show More]", key="show_more_button", on_click=show_more_callback)

    # Always show the "Download Complete Ranking" button
    with col2:
        csv_data = ranking_data.drop(columns=["Score"]).to_csv(index=False)
        st.download_button(
            label=":violet[Download Complete Ranking]",
            data=csv_data,
            file_name='ranking_data.csv',
            mime='text/csv',
        )

    return ranking_data


