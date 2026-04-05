import streamlit as st
import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from google.oauth2 import service_account
import io
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

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

class GreenAreaAnalyzer:
    def __init__(self, locations, start_date, train_years=5, predict_years=2, svr_c=1.0, svr_eps=0.1, smooth_window=3):
        self.locations = locations
        self.start_date = start_date
        self.train_years = train_years
        self.predict_years = predict_years
        self.svr_c = svr_c
        self.svr_eps = svr_eps
        self.smooth_window = smooth_window

    def fetch_and_calculate_ndvi(self, name, coords, start_date, end_date):
        area_of_interest = ee.Geometry.Rectangle([coords[1], coords[0], coords[3], coords[2]])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(area_of_interest)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                      .select(['B8', 'B4']))
        images = collection.toList(collection.size()).getInfo()

        if len(images) == 0:
            st.warning(f"No images found for {name} in the given date range.")
            return name, pd.DataFrame()

        ndvi_values, dates = [], []
        for image_info in images:
            image = ee.Image(image_info['id'])
            timestamp = image_info['properties']['system:time_start'] / 1000
            date = datetime.fromtimestamp(timestamp)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            result = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area_of_interest,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            ndvi_value = result.get('NDVI')
            if ndvi_value is not None:
                ndvi_values.append(float(ndvi_value))
                dates.append(date)

        df = pd.DataFrame({'Date': dates, 'NDVI': ndvi_values})
        df = df.sort_values('Date').reset_index(drop=True)
        return name, df

    def analyze_sequential(self, start_date, end_date):
        results = {}
        for loc in self.locations.items():
            name, data = self.fetch_and_calculate_ndvi(loc[0], loc[1], start_date, end_date)
            results[name] = data
        return results

    def plot_and_predict(self):
        train_start_date = self.start_date.strftime('%Y-%m-%d')
        train_end_dt = self.start_date + pd.DateOffset(years=self.train_years)
        train_end_date = train_end_dt.strftime('%Y-%m-%d')
        predict_end_dt = train_end_dt + pd.DateOffset(years=self.predict_years)
        predict_end_date = predict_end_dt.strftime('%Y-%m-%d')

        with st.status("Querying Earth Engine...", expanded=True) as status:
            st.write(f"Fetching Sentinel-2 harmonized data from {train_start_date} to {train_end_date}...")
            results = self.analyze_sequential(train_start_date, train_end_date)
            status.update(label="Data processing complete!", state="complete", expanded=False)
        
        regressor = SVR(C=self.svr_c, epsilon=self.svr_eps)

        for name, df in results.items():
            if df.empty:
                st.error(f"No data available for {name}.")
                continue

            df['Timestamp'] = pd.to_datetime(df['Date']).astype('int64') / 10**9
            df['NDVI_Smooth'] = df['NDVI'].rolling(window=self.smooth_window, min_periods=1).mean()
            
            X = df['Timestamp'].values.reshape(-1, 1)
            y = df['NDVI_Smooth'].values.reshape(-1, 1)

            st.markdown("### 📊 Region Summary & Analytics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average NDVI", f"{df['NDVI'].mean():.4f}")
            col2.metric("Max NDVI", f"{df['NDVI'].max():.4f}")
            col3.metric("Min NDVI", f"{df['NDVI'].min():.4f}")
            col4.metric("Data Points", f"{len(df)}")

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            future_dates = pd.date_range(start=train_end_date, end=predict_end_date, freq='ME')
            future_timestamps = (future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            future_timestamps = np.array(future_timestamps).reshape(-1, 1)
            future_timestamps_scaled = scaler_X.transform(future_timestamps)

            z = np.polyfit(X.flatten(), y.flatten(), 1)
            train_trend = 'Increasing 📈' if z[0] > 0 else 'Decreasing 📉'
            
            st.info(f"🌿 Historical Greenery Trend for **{name}**: **{train_trend}**")

            regressor.fit(X_scaled, y_scaled.ravel())
            
            fitted = scaler_y.inverse_transform(regressor.predict(X_scaled).reshape(-1, 1)).flatten()
            predictions_scaled = regressor.predict(future_timestamps_scaled)
            predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            
            pred_z = np.polyfit(future_timestamps.flatten(), predictions.flatten(), 1)
            future_trend = 'Increasing' if pred_z[0] > 0 else 'Decreasing'

            tab1, tab2 = st.tabs(["📈 Visualization & Forecast", "🗄️ Raw Data Extract"])

            with tab1:
                fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1.5, 1]})

                ax = axs[0]
                ax.scatter(df['Date'], df['NDVI'], color='lightgray', label='Raw NDVI', s=20, alpha=0.6)
                ax.scatter(df['Date'], df['NDVI_Smooth'], color='royalblue', label=f'Smoothed ({self.smooth_window}-pt)', s=30)
                ax.plot(df['Date'], fitted, color='darkorange', label='SVR Model Fit', linewidth=2)
                ax.set_title(f'Historical Fit - {name}', fontsize=14, pad=15)
                ax.set_ylabel('NDVI Value')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # STRETCH HISTORICAL GRAPH TO ABSOLUTE MIN/MAX
                y_min, y_max = df['NDVI'].min(), df['NDVI'].max()
                if y_min == y_max:
                    ax.set_ylim(y_min - 0.01, y_max + 0.01)
                else:
                    ax.set_ylim(y_min, y_max)

                ax2 = axs[1]
                line_color = 'forestgreen' if future_trend == 'Increasing' else 'crimson'
                ax2.plot(future_dates, predictions, color=line_color, marker='o', markersize=4, linewidth=2, label='Forecast')
                ax2.set_title(f'SVR Forecast ({future_trend})', fontsize=14, pad=15)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                
                # STRETCH FORECAST GRAPH TO ABSOLUTE MIN/MAX
                p_min, p_max = min(predictions), max(predictions)
                if p_min == p_max:
                    ax2.set_ylim(p_min - 0.01, p_max + 0.01)
                else:
                    ax2.set_ylim(p_min, p_max)

                plt.tight_layout()
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="🖼️ Download High-Res Plot",
                    data=buf.getvalue(),
                    file_name=f"{name}_forecast.png",
                    mime="image/png"
                )

            with tab2:
                st.dataframe(df[['Date', 'NDVI', 'NDVI_Smooth']], use_container_width=True)
                csv_data = df[['Date', 'NDVI', 'NDVI_Smooth']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset (CSV)",
                    data=csv_data,
                    file_name=f"{name}_ndvi_data.csv",
                    mime="text/csv",
                )

def main():
    st.set_page_config(page_title="Satellite Green Area Analyzer", page_icon="🌍", layout="wide")
    
    col_title, col_icon = st.columns([8, 1])
    with col_title:
        st.title("Satellite Green Area Analyzer")
    with col_icon:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Copernicus_logo.svg", width=60)
    
    st.markdown("Analyze historical Sentinel-2 satellite imagery and forecast future vegetation trends.")
    st.markdown("---")
    
    if not init_ee():
        st.stop()

    # SIDEBAR CONTROLS
    st.sidebar.header("📅 Timeframe")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2019-01-01'))
    train_years = st.sidebar.slider("Training Horizon (Years)", 1, 10, 5)
    predict_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 2)

    with st.sidebar.expander("⚙️ Advanced SVR & Data Settings"):
        svr_c = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
        svr_eps = st.slider("Epsilon (Tolerance)", 0.01, 1.0, 0.1, 0.01)
        smooth_window = st.slider("Data Smoothing (Records)", 1, 10, 3)

    # MAIN UI: INTERACTIVE MAP
    st.subheader("1. Select Region")
    st.markdown("Use the **square icon** on the left side of the map to draw a rectangle over the area you want to analyze.")
    
    # Initialize a clean Folium map centered roughly on the Amazon
    m = folium.Map(location=[-3.25, -59.75], zoom_start=6)
    
    # Add drawing tools (restrict to rectangles only for bounding box logic)
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

    # Render the map in Streamlit and capture user interactions
    st_data = st_folium(m, width=1200, height=400)
    
    # Extract coordinates if the user drew a rectangle
    min_lat, max_lat, min_lon, max_lon = -3.50, -3.00, -60.00, -59.50 # Default Amazon Coords
    loc_name = "Custom Drawn Region"
    
    if st_data.get("all_drawings"):
        # The coordinates come back as a GeoJSON polygon ring
        coords = st_data["all_drawings"][0]["geometry"]["coordinates"][0]
        lons = [point[0] for point in coords]
        lats = [point[1] for point in coords]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        st.success(f"Region Captured! Bounds: Lat ({min_lat:.3f} to {max_lat:.3f}), Lon ({min_lon:.3f} to {max_lon:.3f})")
    else:
        st.info("No box drawn yet. The default Amazon Rainforest coordinates will be used.")

    st.markdown("---")
    st.subheader("2. Run SVR Machine Learning Analysis")

    # Run Analysis Button
    if st.button("🚀 Analyze Selected Region", type="primary", use_container_width=True):
        locations = {
            loc_name: [min_lat, min_lon, max_lat, max_lon]
        }
        analyzer = GreenAreaAnalyzer(
            locations=locations, 
            start_date=start_date, 
            train_years=train_years, 
            predict_years=predict_years,
            svr_c=svr_c,
            svr_eps=svr_eps,
            smooth_window=smooth_window
        )
        analyzer.plot_and_predict()

if __name__ == "__main__":
    main()