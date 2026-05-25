import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.stats import linregress

# --- PAGE CONFIG ---
st.set_page_config(page_title="Haryana AI Water Sandbox", layout="wide")

st.title("🌊 Haryana Groundwater Risk AI Sandbox")
st.markdown("### Prototype: NASA GRACE-FO Satellite Data Integration")
st.info("Innovator: Srinivasta | Tech: Python, xarray, Folium | Data: NASA JPL Mascon RL06.3")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Loading the pilot file you generated
    ds = xr.open_dataset('haryana_groundwater_pilot.nc')
    
    # Using the specific variable name found in your dataset
    target_var = '__xarray_dataarray_variable__'
    
    # Fallback logic in case of name changes
    if target_var not in ds.data_vars:
        target_var = list(ds.data_vars)[0]
        
    return ds, target_var

try:
    ds, target_var = load_data()
    hry_data = ds[target_var]

    # --- SIDEBAR ANALYSIS ---
    st.sidebar.header("📈 State-Wide Metrics")
    
    # Calculate state-wide average time series
    state_avg = hry_data.mean(dim=['lat', 'lon'])
    
    # Perform Linear Regression for Burn Rate
    x = np.arange(len(state_avg))
    y = state_avg.values
    # Handle potential NaNs for regression
    mask = ~np.isnan(y)
    slope, _, _, _, _ = linregress(x[mask], y[mask])
    
    st.sidebar.metric("Monthly Burn Rate", f"{slope:.4f} cm/mo", delta=f"{slope:.4f}", delta_color="inverse")
    st.sidebar.metric("Annual Depletion", f"{slope*12:.2f} cm/yr")
    
    if slope < -0.4:
        st.sidebar.error("STATUS: CRITICAL")
    else:
        st.sidebar.warning("STATUS: WATCH")

    # --- MAIN DASHBOARD LAYOUT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Risk Map")
        # Initialize Map centered on Haryana
        m = folium.Map(location=[29.05, 76.08], zoom_start=8, tiles='CartoDB Positron')
        
        # Calculate mean anomaly over the entire time period for the map
        data_avg = hry_data.mean(dim='time')
        
        for lat in data_avg.lat.values:
            for lon in data_avg.lon.values:
                val = float(data_avg.sel(lat=lat, lon=lon).values)
                if not np.isnan(val):
                    # Color Logic: Red for High Loss, Orange for Warning
                    if val < -35:
                        color = '#d73027' 
                        status = "CRITICAL"
                    elif val < -15:
                        color = '#fc8d59'
                        status = "WARNING"
                    else:
                        color = '#4575b4'
                        status = "STABLE"
                    
                    folium.Rectangle(
                        bounds=[[lat-0.25, lon-0.25], [lat+0.25, lon+0.25]],
                        color=color, fill=True, fill_opacity=0.5,
                        popup=f"Zone: {status}<br>Anomaly: {val:.2f}cm",
                        tooltip=f"{status}: {val:.2f}cm"
                    ).add_to(m)
        
        st_folium(m, width=800, height=500)

    with col2:
        st.subheader("Regional Risk Gap")
        # Divide state at 76.2 Longitude (Approx East/West divide)
        west_zone = hry_data.sel(lon=slice(74.4, 76.2)).mean(dim=['lat', 'lon'])
        east_zone = hry_data.sel(lon=slice(76.2, 77.8)).mean(dim=['lat', 'lon'])
        
        # Calculate Zone Specific Burn Rates
        x_w = np.arange(len(west_zone))
        x_e = np.arange(len(east_zone))
        s_w, _, _, _, _ = linregress(x_w, west_zone.fillna(0).values)
        s_e, _, _, _, _ = linregress(x_e, east_zone.fillna(0).values)
        
        st.write(f"**West Zone Burn:** {s_w:.4f} cm/mo")
        st.write(f"**East Zone Burn:** {s_e:.4f} cm/mo")
        
        # Risk Gap Calculation
        gap = abs((s_e - s_w) / s_w) * 100
        st.metric("Risk Gap (East vs West)", f"{gap:.1f}%")
        
        st.markdown("---")
        st.write("**Technical Roadmap:**")
        st.caption("1. Machine Learning Downscaling to 5km resolution")
        st.caption("2. Integration with Haryana Water Authority Well Data")
        st.caption("3. Predictive 'Day Zero' forecasting for 22 districts")

except Exception as e:
    st.error("Data Load Error")
    st.write(f"Ensure 'haryana_groundwater_pilot.nc' is in your GitHub repo. Technical details: {e}")

