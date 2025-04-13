import streamlit as st

def main():
    """Display emissions chart image in Streamlit app."""
    st.title("Emissions by Sector - France 2022")
    image_path = "/tmp/emissions_by_sector_france_2022.png"
    
    try:
        st.image(image_path, caption="CO2 Emissions by Sector in France (2022)")
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")

if __name__ == "__main__":
    main()
