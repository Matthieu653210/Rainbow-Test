import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.utils.config_mngr import global_config


def get_data() -> pd.DataFrame:
    """Load and cache CO2 emissions data from configured dataset.

    Returns:
        DataFrame containing emissions data by country and sector.
    """
    data_file = global_config().get_path("datasets_root") / "carbon-monitor-data 1.xlsx"
    assert data_file.exists()
    if data_file.name.endswith(".csv"):
        df = pd.read_csv(data_file, decimal=",")
    else:
        df = pd.read_excel(data_file)
    return df


df = get_data()
# Inspect the DataFrame structure
print("DataFrame columns:", df.columns)

# Check the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Convert the 'date' column to datetime, handling mixed formats
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Extract the year from the date column
df["Year"] = df["date"].dt.year

# Group the data by Year and country, summing only the numeric column ('MtCO2 per day')
grouped_df = df.groupby(["Year", "country"])["MtCO2 per day"].sum().reset_index()

# Create a multiselect widget for selecting countries
selected_countries = st.sidebar.multiselect("Select countries to compare", grouped_df["country"].unique())

if selected_countries:
    # Filter the data based on selected countries
    filtered_df = grouped_df[grouped_df["country"].isin(selected_countries)]

    # Plot the CO2 emissions over time
    plt.figure(figsize=(10, 6))
    for country in selected_countries:
        country_data = filtered_df[filtered_df["country"] == country]
        plt.plot(country_data["Year"], country_data["MtCO2 per day"], label=country)

    plt.title("Evolution of CO2 Emissions")
    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions (MtCO2 per day)")
    plt.legend()

    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plot_path = tmpfile.name

    # Display the plot in the UI
    st.image(plot_path, use_column_width=True)
    st.markdown(f"![Evolution of CO2 Emissions]({plot_path})")

    # Print the title of the plot to stdio
    print("Evolution of CO2 Emissions")
else:
    st.markdown("Please select at least one country to compare.")
