import streamlit as st
import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from google.oauth2 import service_account

@st.cache_resource
def init_ee():
    try:
        if "gcp_service_account" in st.secrets:
            print("Connecting")
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
    def __init__(self, locations, train_years=5, predict_years=2):
        self.locations = locations
        self.train_years = train_years
        self.predict_years = predict_years

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

        return name, pd.DataFrame({'Date': dates, 'NDVI': ndvi_values})

    def analyze_sequential(self, start_date, end_date):
        results = {}
        for loc in self.locations.items():
            name, data = self.fetch_and_calculate_ndvi(loc[0], loc[1], start_date, end_date)
            results[name] = data
        return results

    def plot_and_predict(self):
        base_year = 2019
        train_start_date = datetime(base_year, 1, 1).strftime('%Y-%m-%d')
        train_end_date = datetime(base_year + self.train_years, 1, 1).strftime('%Y-%m-%d')
        predict_end_date = datetime(base_year + self.train_years + self.predict_years, 1, 1).strftime('%Y-%m-%d')

        with st.spinner("Fetching Sentinel-2 data from Earth Engine (this may take a minute)..."):
            results = self.analyze_sequential(train_start_date, train_end_date)
        
        regressors = {
            'SVR': SVR()
        }

        for name, df in results.items():
            if df.empty:
                st.error(f"No data available for {name}.")
                continue

            df['Timestamp'] = pd.to_datetime(df['Date']).astype('int64') / 10**9
            X = df['Timestamp'].values.reshape(-1, 1)
            y = df['NDVI'].values.reshape(-1, 1)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            future_dates = pd.date_range(start=train_end_date, end=predict_end_date, freq='ME')
            future_timestamps = (future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            future_timestamps = np.array(future_timestamps).reshape(-1, 1)
            future_timestamps_scaled = scaler_X.transform(future_timestamps)

            n_rows = len(regressors)
            fig, axs = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows), squeeze=False)

            train_trend = 'increasing' if y[-1][0] > y[0][0] else 'decreasing'
            st.subheader(f"🌿 Greenery Trend for {name} based on Training Data: {train_trend}")

            for i, (reg_name, regressor) in enumerate(regressors.items()):
                regressor.fit(X_scaled, y_scaled.ravel())

                ax = axs[i, 0]
                ax.scatter(df['Date'], y.flatten(), color='blue', label='Actual NDVI', s=30)

                fitted = scaler_y.inverse_transform(
                    regressor.predict(X_scaled).reshape(-1, 1)
                ).flatten()
                ax.plot(df['Date'], fitted, color='orange', label='Fitted NDVI', linewidth=1.5)

                predictions_scaled = regressor.predict(future_timestamps_scaled)
                predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

                ax.plot(future_dates, predictions, color='green', label='Predicted NDVI', linewidth=1.5)
                ax.set_title(f'{reg_name} - {name}')
                ax.legend()
                ax.grid(True)

                trend = 'increasing' if predictions[-1] > predictions[0] else 'decreasing'

                ax2 = axs[i, 1]
                ax2.plot(future_dates, predictions,
                         color='green' if trend == 'increasing' else 'red', marker='o')
                ax2.set_title(f'Greenery Trend - {reg_name} ({trend})')
                ax2.grid(True)
                ax2.set_yticks(np.arange(min(predictions) - 0.005, max(predictions) + 0.005, 0.005))

            plt.tight_layout()
            st.pyplot(fig)

def main():
    st.set_page_config(page_title="NDVI Greenery Analyzer", layout="wide")
    st.title("Satellite Green Area Analyzer")
    
    if not init_ee():
        st.stop()

    st.sidebar.header("Input Location Details")
    loc_name = st.sidebar.text_input("Location Name", "Amazon Rainforest")
    min_lat = st.sidebar.number_input("Min Latitude", value=-3.5000, format="%.4f")
    min_lon = st.sidebar.number_input("Min Longitude", value=-60.0000, format="%.4f")
    max_lat = st.sidebar.number_input("Max Latitude", value=-3.0000, format="%.4f")
    max_lon = st.sidebar.number_input("Max Longitude", value=-59.5000, format="%.4f")

    st.sidebar.markdown("---")
    train_years = st.sidebar.slider("Training Years (from 2019)", 1, 10, 5)
    predict_years = st.sidebar.slider("Prediction Years", 1, 5, 2)

    if st.sidebar.button("Analyze Region", type="primary"):
        locations = {
            loc_name: [min_lat, min_lon, max_lat, max_lon]
        }
        analyzer = GreenAreaAnalyzer(locations, train_years, predict_years)
        analyzer.plot_and_predict()

if __name__ == "__main__":
    main()