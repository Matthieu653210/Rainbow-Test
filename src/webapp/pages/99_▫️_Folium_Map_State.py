import tempfile
from pathlib import Path

import folium
import streamlit as st


def create_toulouse_map() -> folium.Map:
    """Create a Folium map centered on Toulouse with some markers."""
    # Toulouse coordinates
    toulouse_coords = (43.6045, 1.4442)

    # Create map centered on Toulouse
    m = folium.Map(location=toulouse_coords, zoom_start=13)

    # Add some markers
    folium.Marker(location=(43.6045, 1.4442), popup="Capitole de Toulouse", icon=folium.Icon(color="red")).add_to(m)

    folium.Marker(location=(43.6083, 1.4437), popup="Place du Capitole", icon=folium.Icon(color="blue")).add_to(m)

    return m


def main():
    st.title("Toulouse Map")

    # Create the map
    m = create_toulouse_map()

    # Save to temporary HTML file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        m.save(f.name)
        html_path = Path(f.name)

    # Display map using iframe
    st.html(html_path.read_text())


main()
