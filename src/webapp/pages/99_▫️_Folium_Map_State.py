"""Demonstrate saving and restoring Folium map state using streamlit_folium."""

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import folium_static

# Page config
st.title("🌍 Folium Map State Demo")

# Initialize session state
if "map_state" not in st.session_state:
    st.session_state.map_state = None
if "saved_state" not in st.session_state:
    st.session_state.saved_state = None

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
        # JavaScript to get current map state
        save_js = """
        const center = map.getCenter();
        const zoom = map.getZoom();
        const layers = [];
        map.eachLayer(layer => {
            if (layer instanceof L.TileLayer || layer instanceof L.Marker) {
                layers.push(layer.toGeoJSON());
            }
        });
        const state = { center: [center.lat, center.lng], zoom, layers };
        parent.window.postMessage({type: 'mapState', data: state}, '*');
        """
        # Create map with save capability
        m = create_map()
        Draw(export=True).add_to(m)
        folium_static(m, height=500, key="map")
        st.session_state.map_state = save_js
        st.success("Map state saved!")

with col2:
    if st.button("🔄 Restore Map State") and st.session_state.saved_state:
        # Restore map from saved state
        state = st.session_state.saved_state
        m = create_map(center=state['center'], zoom=state['zoom'])
        Draw(export=True).add_to(m)
        folium_static(m, height=500, key="map")
        st.success("Map state restored!")

with col3:
    if st.button("🔄 Reset Map"):
        # Reset to initial state
        m = create_map()
        Draw(export=True).add_to(m)
        folium_static(m, height=500, key="map")
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
folium_static(m, height=500, key="map")

# Add JavaScript for state handling
st.markdown(
    """
<script>
// Listen for map state messages
window.addEventListener('message', function(event) {
    if (event.data.type === 'mapState') {
        // Store state in Streamlit session
        parent.window.streamlitAPI.setComponentValue(JSON.stringify(event.data.data));
    }
});

// Handle the map state from Streamlit
function handleMapState(state) {
    if (state) {
        const { center, zoom } = state;
        map.setView(center, zoom);
    }
}

// Get the initial state from Streamlit
const initialState = parent.window.streamlitAPI.getComponentValue();
if (initialState) {
    handleMapState(JSON.parse(initialState));
}
</script>
""",
    unsafe_allow_html=True,
)

# Handle the saved state from JavaScript
if st.session_state.map_state:
    try:
        state = json.loads(st.session_state.map_state)
        st.session_state.saved_state = {
            'center': state['center'],
            'zoom': state['zoom']
        }
    except:
        pass
