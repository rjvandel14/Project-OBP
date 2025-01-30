import pandas as pd

def get_min_max_ranking(dmatrix, df):
    partnership_scores = []
    company_names = df['name'].unique()  # Get unique company names

    # Iterate through all unique pairs of companies
    for i, company1 in enumerate(company_names):
        for j, company2 in enumerate(company_names):
            if i < j:  # Ensure each pair is only processed once

                customers1 = df[df['name'] == company1].index.tolist()
                customers2 = df[df['name'] == company2].index.tolist()

                # Max inter-customer distance
                max_inter_customer = dmatrix.iloc[customers1, customers2].max().max()

                # Max depot distance
                max_depot_distance = max(
                    dmatrix.iloc[customers1, 0].max(),  
                    dmatrix.iloc[customers2, 0].max()
                )

                # Calculate min-max score
                min_max_score = max(max_inter_customer, max_depot_distance)

                partnership_scores.append({
                    'Company A': company1,
                    'Company B': company2,
                    'Score': min_max_score
                })

    partnership_df = pd.DataFrame(partnership_scores)

    # Sort by score
    partnership_df = partnership_df.sort_values('Score', ascending=False).reset_index(drop=True)

    # Ensure all rows are included and ranked, even with duplicate scores
    partnership_df['Rank'] = partnership_df.index + 1

    return partnership_df[['Rank', 'Company A', 'Company B', 'Score']]