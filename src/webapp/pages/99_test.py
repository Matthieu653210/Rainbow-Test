import streamlit as st


def main():
    """Display emissions chart image in Streamlit app."""
    st.title("Emissions by Sector - France 2022")
    image_path = "/tmp/emissions_by_sector_france_2022.png"

    try:
        # the image is not display. Try to fix AI!
        st.markdown(f"![CO2 Emissions by Sector in France (2022)]({image_path})")
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")


main()
