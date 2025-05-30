from streamlit_extras.sandbox import sandbox


def follium_ex():
    def embedded_app():
        import json

        import folium
        import pandas as pd
        import streamlit as st
        from pyodide.http import open_url
        from streamlit_folium import st_folium

        st.write("Hello")

        url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
        state_geo = f"{url}/us-states.json"
        state_unemployment = f"{url}/US_Unemployment_Oct2012.csv"
        state_data = pd.read_csv(open_url(state_unemployment))
        geo_json = json.loads(open_url(state_geo).read())

        m = folium.Map(location=[48, -102], zoom_start=3)

        folium.Choropleth(
            geo_data=geo_json,
            name="choropleth",
            data=state_data,
            columns=["State", "Unemployment"],
            key_on="feature.id",
            fill_color="YlGn",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Unemployment Rate (%)",
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # Get the HTML representation of the map
        map_html = m._repr_html_()
        # Display the map HTML directly in the sandbox

        st_folium(m)

    sandbox(embedded_app, requirements=["folium", "ipython", "streamlit_folium"])


def example1():
    def embedded_app():
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import streamlit as st

        @st.cache_data
        def get_data():
            dates = pd.date_range(start="01-01-2020", end="01-01-2023")
            data = np.random.randn(len(dates), 1).cumsum(axis=0)
            return pd.DataFrame(data, index=dates, columns=["Value"])

        data = get_data()

        value = st.slider(
            "Select a range of values",
            int(data.min()),
            int(data.max()),
            (int(data.min()), int(data.max())),
        )
        filtered_data = data[(data["Value"] >= value[0]) & (data["Value"] <= value[1])]
        st.plotly_chart(px.line(filtered_data, y="Value"))

    sandbox(embedded_app)


follium_ex()
