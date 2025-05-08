import os
import glob
import joblib
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask

# --- Configuration ---
plot_vis = True
input_folder = 'per_lake_rasters/'
output_folder = 'model_output_tifs/'
shapefile_path = 'ALL_DRMs/attachments/Jillians_DRMs_polygons.shp'
model_path = 'etr_final.joblib'
os.makedirs(output_folder, exist_ok=True)

# --- Load trained model and shapefile ---
model = joblib.load(model_path)
clip_gdf = gpd.read_file(shapefile_path)

def extract_fid_from_filename(filename):
    """Extracts fid_1 from filenames like lake_12.0_20240610.tif or lake_12_20240610.tif"""
    base = os.path.basename(filename)
    try:
        parts = base.replace('.tif', '').split('_')
        # Convert float strings like '12.0' to int
        return int(float(parts[1]))  # e.g., '12.0' -> 12
    except (IndexError, ValueError):
        return None

def ml_raster(raster_path, output_path):
    with rasterio.open(raster_path) as src:
        fid = extract_fid_from_filename(raster_path)
        if fid is None:
            print(f"Skipping {raster_path} — could not parse fid_1.")
            return

        # Match geometry by fid_1
        lake_geom = clip_gdf[clip_gdf['fid_1'] == fid]
        if lake_geom.empty:
            print(f"No matching geometry for fid_1={fid} — skipping.")
            return

        # Clip raster to lake geometry
        geom = lake_geom.geometry.values[0]
        try:
            clipped_data, clipped_transform = mask(src, [geom], crop=True)
            masked_data, _ = mask(src, [geom], crop=True, filled=False)
        except Exception as e:
            print(f" Masking error for fid_1={fid}: {e}")
            return

        bands, height, width = clipped_data.shape
        reshaped_data = clipped_data.reshape(bands, -1).T

        nodata = src.nodata if src.nodata is not None else -9999
        geometry_mask = np.ma.getmaskarray(masked_data)[0].flatten()
        nodata_mask = np.any(clipped_data == nodata, axis=0).flatten()
        combined_mask = geometry_mask | nodata_mask

        valid_data = reshaped_data[~combined_mask]
        if valid_data.size == 0:
            print(f" No valid pixels to predict for fid_1={fid}.")
            return

        # Predict using the model
        predictions = model.predict(valid_data)

        # Build prediction raster
        prediction_raster = np.full((height * width,), nodata, dtype='float32')
        prediction_raster[~combined_mask] = predictions
        prediction_raster = prediction_raster.reshape((height, width))

        # Save prediction raster
        profile = src.profile.copy()
        profile.update({
            'height': height,
            'width': width,
            'transform': clipped_transform,
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw',
            'nodata': nodata
        })

        print(f" Writing prediction for lake {fid} to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction_raster, 1)

        # Optional plotting
        if plot_vis:
            plt.figure(figsize=(8, 6))
            plt.imshow(np.ma.masked_equal(prediction_raster, nodata), cmap='viridis')
            plt.title(f'Prediction — Lake {fid}')
            plt.colorbar(label='Prediction')
            plt.axis('off')
            plt.show()


# --- Run over all rasters ---
raster_files = glob.glob(os.path.join(input_folder, '*.tif'))
if not raster_files:
    print(" No rasters found in input folder.")

for raster_file in raster_files:
    filename = os.path.basename(raster_file)
    output_path = os.path.join(output_folder, filename.replace('.tif', '_pred.tif'))
    ml_raster(raster_file, output_path)

