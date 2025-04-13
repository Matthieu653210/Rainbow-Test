import streamlit as st


def main():
    """Display emissions chart image in Streamlit app."""
    st.title("Emissions by Sector - France 2022")
    image_path = "/tmp/emissions_by_sector_france_2022.png"

    try:
        with open(image_path, "rb") as f:
            from base64 import b64encode
            image_base64 = b64encode(f.read()).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{image_base64}" alt="CO2 Emissions by Sector in France (2022)">',
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")


main()
