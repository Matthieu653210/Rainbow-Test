"""Demonstrate saving and restoring Folium map state using streamlit_folium."""

import json
import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import folium_static

# Page config
st.title("🌍 Folium Map State Demo")

# Initialize session state
if "saved_state" not in st.session_state:
    st.session_state.saved_state = None
if "component_value" not in st.session_state:
    st.session_state.component_value = json.dumps({
        'center': [45.5236, -122.6750],
        'zoom': 13
    })

# Create initial map
def create_map(center=None, zoom=None):
    if center and zoom:
        return folium.Map(location=center, zoom_start=zoom)
    return folium.Map(location=[45.5236, -122.6750], zoom_start=13)

# Main map container
map_container = st.container()

# Buttons for saving/restoring state
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💾 Save Map State"):
        # Create map with save capability
        m = create_map()
        Draw(export=True).add_to(m)
        
        # Create a component to capture the map state
        def save_map_state():
            return st.components.v1.html(
                """
                <script>
                function saveMapState() {
                    const center = map.getCenter();
                    const zoom = map.getZoom();
                    const state = { 
                        center: [center.lat, center.lng], 
                        zoom: zoom 
                    };
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: JSON.stringify(state)
                    }, '*');
                    return false;
                }
                // Save state when the function is called
                saveMapState();
                </script>
                """,
                height=0
            )
        
        # Render the map
        folium_static(m, height=500)
        
        # Save the state and handle the response
        save_map_state()
        st.session_state.saved_state = json.loads(st.session_state.component_value)
        st.success("Map state saved!")

with col2:
    if st.button("🔄 Restore Map State") and st.session_state.saved_state:
        # Restore map from saved state
        state = st.session_state.saved_state
        m = create_map(center=state['center'], zoom=state['zoom'])
        Draw(export=True).add_to(m)
        folium_static(m, height=500)
        st.success("Map state restored!")

with col3:
    if st.button("🔄 Reset Map"):
        # Reset to initial state
        m = create_map()
        Draw(export=True).add_to(m)
        folium_static(m, height=500)
        st.session_state.map_state = None
        st.session_state.saved_state = None
        st.success("Map reset to initial state!")

# Display the map
if st.session_state.saved_state:
    state = st.session_state.saved_state
    m = create_map(center=state['center'], zoom=state['zoom'])
else:
    m = create_map()
Draw(export=True).add_to(m)
folium_static(m, height=500)

# Handle the saved state from JavaScript
if st.session_state.component_value:
    try:
        st.session_state.saved_state = json.loads(st.session_state.component_value)
    except:
        st.session_state.saved_state = None
