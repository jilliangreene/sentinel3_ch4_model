import os
import glob
import joblib
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask

# Configuration
plot_vis = True  # Set to False to disable plotting
input_folder = 'multiband_rasters/'
output_folder = 'output_tifs/'
shapefile_path = 'ALL_DRMs/ALL_DRMs.shp'
os.makedirs(output_folder, exist_ok=True)

# Load trained model
model = joblib.load('etr_example.joblib')

# Load and reproject shapefile to match raster
clip_gdf = gpd.read_file(shapefile_path)

def ml_raster(raster_path, output_path):
    with rasterio.open(raster_path) as src:
        if clip_gdf.crs != src.crs:
            print("Reprojecting shapefile to match raster CRS...")
            gdf_local = clip_gdf.to_crs(src.crs)
        else:
            gdf_local = clip_gdf

        clip_shapes = [feature["geometry"] for feature in gdf_local.__geo_interface__['features']]
        
        print(f"Clipping and predicting for {os.path.basename(raster_path)}...")
        clipped_data, clipped_transform = mask(src, clip_shapes, crop=True)
        masked_data, _ = mask(src, clip_shapes, crop=True, filled=False)

        bands, height, width = clipped_data.shape
        reshaped_data = clipped_data.reshape(bands, -1).T

        nodata = src.nodata if src.nodata is not None else -9999
        geometry_mask = np.ma.getmaskarray(masked_data)[0].flatten()
        nodata_mask = np.any(clipped_data == nodata, axis=0).flatten()
        combined_mask = geometry_mask | nodata_mask

        valid_data = reshaped_data[~combined_mask]
        print(f"Valid pixels: {valid_data.shape[0]}")

        if valid_data.shape[0] == 0:
            print("No valid pixels to predict — skipping.")
            return

        predictions = model.predict(valid_data)

        prediction_raster = np.full((height * width,), nodata, dtype='float32')
        prediction_raster[~combined_mask] = predictions
        prediction_raster = prediction_raster.reshape((height, width))

        profile = src.profile
        profile.update({
            'height': height,
            'width': width,
            'transform': clipped_transform,
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw',
            'nodata': nodata
        })

        print(f"Writing output to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction_raster, 1)

        if plot_vis:
            plt.figure(figsize=(8, 6))
            show_data = np.ma.masked_equal(prediction_raster, nodata)
            plt.imshow(show_data, cmap='viridis')
            plt.title(f'Prediction - {os.path.basename(raster_path)} (Masked to Shape)')
            plt.colorbar(label='Prediction')
            plt.axis('off')
            plt.show()


# Process all rasters
raster_files = glob.glob(os.path.join(input_folder, '*.tif'))

if not raster_files:
    print("No rasters found!")

for raster_file in raster_files:
    filename = os.path.basename(raster_file)
    output_path = os.path.join(output_folder, filename)
    ml_raster(raster_file, output_path)
