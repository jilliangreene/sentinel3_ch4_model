import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from datetime import datetime, timedelta
from shapely.geometry import mapping

# --- Paths ---
raster_folder = '/Volumes/Backup Plus/ML_inputs'
daily_climate_folder = '/Volumes/Backup Plus/gridMET_rasters'
output_folder = 'per_lake_rasters/'
shapefile_path = 'ALL_DRMs/attachments/Jillians_DRMs_polygons.shp'
temp_csv_path = 'MODIS_All_DRMS_wFID.csv'

# --- Variables ---
variables = ['CI', 'chla', 'PAR', 'Kd490', 'adg443', 'TSM']
derived_variables = ['NDCI', 'MCI'] 

# --- Dates ---
start_date = datetime.strptime("20240610", "%Y%m%d")
end_date = datetime.strptime("20240619", "%Y%m%d")
date_list = [(start_date + timedelta(days=x)) for x in range((end_date - start_date).days + 1)]

# --- Load data ---
gdf = gpd.read_file(shapefile_path)
temp_csv = pd.read_csv(temp_csv_path)
temp_csv['date'] = pd.to_datetime(temp_csv[['Year', 'Month', 'Day']])

# --- Ensure output folder exists ---
os.makedirs(output_folder, exist_ok=True)

# --- Utility Functions ---

def clip_and_resample(src_path, geometry, ref_shape=None, ref_transform=None, ref_crs=None, band=1):
    with rasterio.open(src_path) as src:
        geometry = gpd.GeoSeries([geometry], crs=gdf.crs).to_crs(src.crs)
        out_image, out_transform = mask(src, [mapping(geometry[0])], crop=True, indexes=band, filled=False, all_touched=True)

        if out_image.ndim == 3:
            masked_band = out_image[0]
        else:
            masked_band = out_image

        if np.ma.is_masked(masked_band):
            masked_band = masked_band.filled(0.0)

        if ref_shape is None:
            return masked_band, out_transform, src.crs

        resampled = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=masked_band,
            destination=resampled,
            src_transform=out_transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )
        return resampled, ref_transform, ref_crs

# Check: shp names
print("Shapefile columns:", gdf.columns)


# --- Main Loop ---
for date in date_list:
    date_str = date.strftime("%Y%m%d")
    print(f"\nProcessing date: {date_str}")

    for idx, row in gdf.iterrows():
        fid = row['fid_1']
        geom = row.geometry
        print(f" Processing lake fid_1={fid}")

        # Set up storage
        lake_rasters = {}

        ref_shape, ref_transform, ref_crs = None, None, None

        # --- Load raster variables ---
        for var in variables:
            matching_files = [f for f in os.listdir(raster_folder) if date_str in f and var in f and f.endswith(".tif")]
            if not matching_files:
                print(f"    Missing file for {var} on {date_str}")
                continue
            try:
                file_path = os.path.join(raster_folder, matching_files[0])
                arr, trans, crs = clip_and_resample(file_path, geom, band=1)
                if ref_shape is None:
                    ref_shape, ref_transform, ref_crs = arr.shape, trans, crs
                else:
                    arr, _, _ = clip_and_resample(file_path, geom, ref_shape, ref_transform, ref_crs, band=1)
                lake_rasters[var] = arr
                print(f"   Loaded {var}")
            except Exception as e:
                print(f"  Failed {var}: {e}")

        if len(lake_rasters) != len(variables):
            print(f"   Skipping lake {fid} — missing base rasters")
            continue
        

        # --- Load reflectance bands needed for NDCI and MCI ---
        needed_rtoa_bands = ['rtoa8', 'rtoa10', 'rtoa11', 'rtoa12']
        rtoa_data = {}

        for band_name in needed_rtoa_bands:
            matching_files = [f for f in os.listdir(raster_folder) if date_str in f and band_name in f and f.endswith(".tif")]
            if not matching_files:
                print(f"   Missing {band_name} for NDCI/MCI on {date_str}")
                break
            try:
                file_path = os.path.join(raster_folder, matching_files[0])
                arr, _, _ = clip_and_resample(file_path, geom, ref_shape, ref_transform, ref_crs, band=1)
                rtoa_data[band_name] = arr
            except Exception as e:
                print(f"    Failed {band_name}: {e}")
                break

        # --- Compute derived variables: NDCI and MCI ---
        if all(b in rtoa_data for b in ['rtoa8', 'rtoa11']):
            numerator = rtoa_data['rtoa8'] - rtoa_data['rtoa11']
            denominator = rtoa_data['rtoa8'] + rtoa_data['rtoa11']
            with np.errstate(divide='ignore', invalid='ignore'):
                ndci = np.where(denominator != 0, numerator / denominator, 0.0)
            lake_rasters['NDCI'] = ndci
            print(f"  Computed NDCI")

        if all(b in rtoa_data for b in ['rtoa10', 'rtoa11', 'rtoa12']):
            mci = rtoa_data['rtoa11'] - rtoa_data['rtoa12'] - (rtoa_data['rtoa10'] - rtoa_data['rtoa12']) * ((708.75 - 753.75) / (681.25 - 753.75))
            lake_rasters['MCI'] = mci
            print(f"  Computed MCI")


        # --- Precipitation stack (5-day) ---
        pr_stack = []
        for i in range(0, 5):
            past_date = date - timedelta(days=i)
            past_str = past_date.strftime("%Y%m%d")
            past_file = os.path.join(daily_climate_folder, f"{past_str}.tif")
            if not os.path.exists(past_file):
                print(f"   Missing precip for {past_str}")
                break
            with rasterio.open(past_file) as src:
                band = src.read(1)
                band[band == src.nodata] = 0.0
                resampled = np.zeros(ref_shape, dtype=np.float32)
                reproject(
                    source=band,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
                pr_stack.append(resampled)
        if len(pr_stack) < 5:
            print(f"   Skipping lake {fid} — not enough precip data")
            continue
        lake_rasters['pr5'] = np.sum(pr_stack, axis=0)

        # --- Today's tmmx and vs ---
        today_file = os.path.join(daily_climate_folder, f"{date_str}.tif")
        if not os.path.exists(today_file):
            print(f"   Missing today’s climate file")
            continue
        with rasterio.open(today_file) as src:
            for band_name, band_num in [('tmmx', 5), ('vs', 6)]:
                band = src.read(band_num)
                band[band == src.nodata] = 0.0
                resampled = np.zeros(ref_shape, dtype=np.float32)
                reproject(
                    source=band,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
                lake_rasters[band_name] = resampled

        # --- Temperature from CSV ---
        temp_day = temp_csv[(temp_csv['date'] == date) & (temp_csv['fid_1'] == fid) & (temp_csv['Kelvin'] > 0)]
        if not temp_day.empty:
            temp_value = float(temp_day['Kelvin'].values[0])
            temp_band = np.full(ref_shape, temp_value, dtype=np.float32)
            lake_rasters['tempK'] = temp_band
            print(f"   Temperature added: {temp_value}K")
        else:
            print(f"    ⏭️ Skipping lake {fid} — no temperature for {date_str}")
            continue  # Skip this lake if no temp


        # --- Write raster ---
        all_vars = ['CI', 'Chla.y', 'NDCI', 'MCI', 'PAR', 'Kd490', 'ADG443', 'TSM', 'pr5', 'water_temp_K', 'vs']
        # Rename keys in lake_rasters to match model expectations
        if 'chla' in lake_rasters:
            lake_rasters['Chla.y'] = lake_rasters.pop('chla')
        if 'adg443' in lake_rasters:
            lake_rasters['ADG443'] = lake_rasters.pop('adg443')
        if 'tempK' in lake_rasters:
            lake_rasters['water_temp_K'] = lake_rasters.pop('tempK')

        stack = np.stack([lake_rasters[v] for v in all_vars])

        meta = {
            'driver': 'GTiff',
            'height': ref_shape[0],
            'width': ref_shape[1],
            'count': len(all_vars),
            'dtype': 'float32',
            'crs': ref_crs,
            'transform': ref_transform,
            'nodata': np.nan
        }

        output_path = os.path.join(output_folder, f"lake_{fid}_{date_str}.tif")
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i, var in enumerate(all_vars):
                dst.write(stack[i], i + 1)
                dst.set_band_description(i + 1, var)

        print(f" Saved: {output_path}")
