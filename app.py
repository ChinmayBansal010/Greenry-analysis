import streamlit as st
import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from google.oauth2 import service_account
import io
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

# Import Prophet instead of SVR
from prophet import Prophet

@st.cache_resource
def init_ee():
    try:
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(key_dict)
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            ee.Initialize(credentials=scoped_credentials, project='satellite-454512')
            return True
        
        ee.Initialize(project='satellite-454512')
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project='satellite-454512')
            return True
        except Exception as e:
            st.error(f"Earth Engine Initialization failed: {e}")
            return False

@st.cache_data(show_spinner=False)
def fetch_ee_data_cached(coords, start_date, end_date):
    area_of_interest = ee.Geometry.Rectangle([coords[1], coords[0], coords[3], coords[2]])
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(area_of_interest)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

    def calculate_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        mean_dict = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_of_interest,
            scale=30,
            maxPixels=1e13
        )
        return ee.Feature(None, {
            'NDVI': mean_dict.get('NDVI'),
            'system:time_start': image.get('system:time_start')
        })

    ndvi_collection = collection.map(calculate_ndvi)
    
    try:
        info = ndvi_collection.getInfo()
    except Exception:
        return pd.DataFrame()

    features = info.get('features', [])
    dates = []
    ndvi_values = []
    for f in features:
        props = f.get('properties', {})
        val = props.get('NDVI')
        t = props.get('system:time_start')
        if val is not None and t is not None:
            dates.append(datetime.fromtimestamp(t / 1000.0))
            ndvi_values.append(val)

    df = pd.DataFrame({'Date': dates, 'NDVI': ndvi_values})
    df = df.sort_values('Date').reset_index(drop=True)
    return df

class GreenAreaAnalyzer:
    def __init__(self, locations, start_date, train_years=5, predict_years=2, flexibility=0.05, smooth_window=3):
        self.locations = locations
        self.start_date = start_date
        self.train_years = train_years
        self.predict_years = predict_years
        self.flexibility = flexibility
        self.smooth_window = smooth_window

    def analyze_sequential(self, start_date, end_date):
        results = {}
        for loc_name, coords in self.locations.items():
            df = fetch_ee_data_cached(tuple(coords), start_date, end_date)
            if df.empty:
                st.warning(f"No images found for {loc_name} in the given date range.")
            results[loc_name] = df
        return results

    def plot_and_predict(self):
        train_start_date = self.start_date.strftime('%Y-%m-%d')
        train_end_dt = self.start_date + pd.DateOffset(years=self.train_years)
        train_end_date = train_end_dt.strftime('%Y-%m-%d')

        with st.status("Querying Earth Engine...", expanded=True) as status:
            st.write(f"Fetching Sentinel-2 harmonized data from {train_start_date} to {train_end_date}...")
            results = self.analyze_sequential(train_start_date, train_end_date)
            status.update(label="Data processing complete!", state="complete", expanded=False)

        for name, df in results.items():
            if df.empty:
                st.error(f"No data available for {name}.")
                continue

            # Smooth data to remove extreme cloud anomalies
            df['NDVI_Smooth'] = df['NDVI'].rolling(window=self.smooth_window, min_periods=1).mean()
            
            st.markdown("### 📊 Region Summary & Analytics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average NDVI", f"{df['NDVI'].mean():.4f}")
            col2.metric("Max NDVI", f"{df['NDVI'].max():.4f}")
            col3.metric("Min NDVI", f"{df['NDVI'].min():.4f}")
            col4.metric("Data Points", f"{len(df)}")

            # ---------------------------------------------------------
            # PROPHET FORECASTING LOGIC
            # ---------------------------------------------------------
            # Prophet requires columns specifically named 'ds' (datestamp) and 'y' (value)
            df_prophet = df[['Date', 'NDVI_Smooth']].rename(columns={'Date': 'ds', 'NDVI_Smooth': 'y'})
            
            # Initialize Prophet Model
            # changepoint_prior_scale controls how flexible the trend line is.
            m = Prophet(
                changepoint_prior_scale=self.flexibility, 
                yearly_seasonality=True, 
                weekly_seasonality=False, 
                daily_seasonality=False
            )
            m.fit(df_prophet)

            # Create dataframe for future predictions (Monthly frequency)
            future = m.make_future_dataframe(periods=self.predict_years * 12, freq='ME')
            forecast = m.predict(future)

            # Calculate overall future trend mathematically
            future_only = forecast[forecast['ds'] > train_end_date]
            if len(future_only) > 1:
                trend_diff = future_only['yhat'].iloc[-1] - future_only['yhat'].iloc[0]
                future_trend = 'Increasing 📈' if trend_diff > 0 else 'Decreasing 📉'
            else:
                future_trend = 'Stable'

            st.info(f"🌿 Prophet Forecasted Trend for **{name}**: **{future_trend}**")

            tab1, tab2 = st.tabs(["📈 Prophet Forecast & Seasonality", "🗄️ Raw Data Extract"])

            with tab1:
                # Plot 1: Main Forecast Plot
                fig = m.plot(forecast, figsize=(14, 6))
                ax = fig.gca()
                ax.set_title(f'NDVI Forecast - {name} (Prophet Model)', fontsize=16, pad=15)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('NDVI Value', fontsize=12)
                
                # Add a vertical line to separate historical data from future prediction
                ax.axvline(pd.to_datetime(train_end_date), color='red', linestyle='--', alpha=0.6, label='Prediction Start')
                ax.legend()
                
                # Stretch Y-axis dynamically
                y_min = forecast['yhat_lower'].min()
                y_max = forecast['yhat_upper'].max()
                ax.set_ylim(y_min - 0.02, y_max + 0.02)
                
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="🖼️ Download Forecast Plot",
                    data=buf.getvalue(),
                    file_name=f"{name}_prophet_forecast.png",
                    mime="image/png"
                )

                st.markdown("---")
                st.markdown("#### 🔄 Seasonal Breakdown")
                st.markdown("This chart isolates the overall trend from the yearly repeating wet/dry season cycle.")
                # Plot 2: Prophet Components (Trend + Yearly Seasonality)
                fig_comp = m.plot_components(forecast, figsize=(14, 6))
                st.pyplot(fig_comp)

            with tab2:
                # Merge original data with forecast data for a complete CSV download
                export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Date', 'yhat': 'Predicted_NDVI', 'yhat_lower': 'Lower_Confidence', 'yhat_upper': 'Upper_Confidence'
                })
                # Join with actual historical data where available
                export_df = pd.merge(export_df, df[['Date', 'NDVI']], on='Date', how='left')
                
                st.dataframe(export_df, use_container_width=True)
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast Dataset (CSV)",
                    data=csv_data,
                    file_name=f"{name}_prophet_data.csv",
                    mime="text/csv",
                )

def main():
    st.set_page_config(page_title="Satellite Green Area Analyzer", page_icon="🌍", layout="wide")
    
    col_title, col_icon = st.columns([8, 1])
    with col_title:
        st.title("Satellite Green Area Analyzer")
    with col_icon:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Copernicus_logo.svg", width=60)
    
    st.markdown("Analyze historical Sentinel-2 satellite imagery and forecast future vegetation trends using Facebook Prophet.")
    st.markdown("---")
    
    if not init_ee():
        st.stop()

    # SIDEBAR CONTROLS
    st.sidebar.header("📅 Timeframe")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2019-01-01'))
    train_years = st.sidebar.slider("Training Horizon (Years)", 1, 10, 5)
    predict_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 2)

    with st.sidebar.expander("⚙️ Prophet Model Settings"):
        st.markdown("*Adjust how strictly the model follows trend changes.*")
        flexibility = st.slider("Trend Flexibility", 0.001, 0.500, 0.050, 0.010)
        smooth_window = st.slider("Pre-Smoothing (Records)", 1, 10, 3)

    # MAIN UI: INTERACTIVE MAP
    st.subheader("1. Select Region")
    st.markdown("Use the **square icon** on the left side of the map to draw a rectangle over the area you want to analyze.")
    
    m = folium.Map(location=[-3.25, -59.75], zoom_start=6)
    
    Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': True
        }
    ).add_to(m)

    st_data = st_folium(
        m, 
        width=1200, 
        height=400, 
        returned_objects=["all_drawings"] 
    )
    
    min_lat, max_lat, min_lon, max_lon = -3.50, -3.00, -60.00, -59.50
    loc_name = "Custom Drawn Region"
    
    if st_data.get("all_drawings"):
        coords = st_data["all_drawings"][0]["geometry"]["coordinates"][0]
        lons = [point[0] for point in coords]
        lats = [point[1] for point in coords]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        st.success(f"Region Captured! Bounds: Lat ({min_lat:.3f} to {max_lat:.3f}), Lon ({min_lon:.3f} to {max_lon:.3f})")
    else:
        st.info("No box drawn yet. The default Amazon Rainforest coordinates will be used.")

    st.markdown("---")
    st.subheader("2. Run Prophet Forecast Analysis")

    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False

    if st.button("🚀 Analyze Selected Region", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

    if st.session_state.run_analysis:
        locations = {
            loc_name: [min_lat, min_lon, max_lat, max_lon]
        }
        analyzer = GreenAreaAnalyzer(
            locations=locations, 
            start_date=start_date, 
            train_years=train_years, 
            predict_years=predict_years,
            flexibility=flexibility,
            smooth_window=smooth_window
        )
        analyzer.plot_and_predict()

if __name__ == "__main__":
    main()