import pandas as pd

def get_min_max_ranking(dmatrix, df):
    """
    Compute the min-max ranking for potential partnerships between companies 
    based on the maximum distances between customer pairs and the depot.

    Parameters:
    - dmatrix (pd.DataFrame): Distance matrix where rows and columns correspond to customers and the depot.
    - df (pd.DataFrame): DataFrame containing customer data, including the 'name' column for company names.

    Returns:
    - pd.DataFrame: A DataFrame with the ranking of company pairs based on their min-max scores.
    """
    partnership_scores = []
    company_names = df['name'].unique()  # Get the list of unique companies

    # Iterate through all unique pairs of companies (no repetitions)
    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Ensure each pair is processed once (avoid (A, B) and (B, A))

                # Get the indices of customers belonging to each company
                customers1 = df[df['name'] == company1].index.tolist()
                customers2 = df[df['name'] == company2].index.tolist()

                # Calculate the maximum distance between any customer from company1 and any customer from company2
                max_inter_customer = dmatrix.iloc[customers1, customers2].max().max()

                # Calculate the maximum distance from any customer to the depot
                max_depot_distance = max(
                    dmatrix.iloc[customers1, 0].max(),  # Max distance from customers1 to the depot
                    dmatrix.iloc[customers2, 0].max()   # Max distance from customers2 to the depot
                )

                # Min-max score: The greater of the inter-customer or depot distances
                min_max_score = max(max_inter_customer, max_depot_distance)

                # Store the results for this partnership
                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Score': min_max_score
                })

    # Create a DataFrame to store partnership scores
    partnership_df = pd.DataFrame(partnership_scores)

    # Sort the DataFrame by score in descending order and reset the index
    partnership_df = partnership_df.sort_values('Score', ascending=False).reset_index(drop=True)

    # Assign a rank to each partnership
    partnership_df['Rank'] = partnership_df.index + 1  # Ranks start from 1

    # Return the ranked partnerships with relevant details
    return partnership_df[['Rank', 'Company A', 'Company B', 'Score']]
