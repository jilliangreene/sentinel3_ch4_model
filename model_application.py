#import libraries
import joblib
import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from datetime import datetime, timedelta
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping
from shapely.ops import unary_union

# def inspect_overlap(raster_path, shapefile_path):

#     with rasterio.open(raster_path) as src:
#         raster_bounds = box(*src.bounds)
#         raster_crs = src.crs

#         print(f"\n Raster: {os.path.basename(raster_path)}")
#         print(f"  CRS: {raster_crs}")
#         print(f"  Bounds: {src.bounds}")

#         gdf = gpd.read_file(shapefile_path).to_crs(raster_crs)
#         shape_bounds = gdf.total_bounds
#         shape_box = box(*shape_bounds)

#         print(f"Shapefile bounds (in raster CRS): {shape_bounds}")

#         if not raster_bounds.intersects(shape_box):
#             print(" No spatial overlap!")
#         else:
#             print("Shapefile and raster DO overlap.")

# inspect_overlap("testing/adg443_DRMs_20240610160106.tif", "ALL_DRMs/ALL_DRMs.shp")


# --- Paths ---
raster_folder = 'testing/'
daily_climate_folder = 'testing/gridmet_test/'
output_folder = 'output_tif/'
shapefile_path = 'ALL_DRMs/ALL_DRMs.shp'

# --- Variables ---
variables = ['CI', 'chla', 'PAR', 'Kd490', 'adg443', 'TSM']

# --- Date Range ---
start_date = datetime.strptime("20240610", "%Y%m%d")
end_date = datetime.strptime("20240619", "%Y%m%d")
date_list = [(start_date + timedelta(days=x)) for x in range((end_date - start_date).days + 1)]

# --- Utility Function ---
def clip_and_resample_to_template(raster_path, band=1, template_shape=None, template_transform=None, template_crs=None):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(shapefile_path).to_crs(src.crs)
        buffered_geom = unary_union(gdf.geometry).buffer(0.0001)
        shapes = [mapping(buffered_geom)]

        out_image, out_transform = mask(src, shapes, crop=True, indexes=band, filled=False, all_touched=True)

        if out_image.ndim == 2:
            out_image = out_image[np.newaxis, :, :]
        masked_band = out_image[0]

        if np.ma.is_masked(masked_band) and masked_band.mask.all():
            raise ValueError("All values are nodata after clipping.")

        clipped_array = masked_band.filled(0.0)

        if template_shape is None:
            return clipped_array, out_transform, src.crs

        resampled = np.zeros(template_shape, dtype=np.float32)
        reproject(
            source=clipped_array,
            destination=resampled,
            src_transform=out_transform,
            src_crs=src.crs,
            dst_transform=template_transform,
            dst_crs=template_crs,
            resampling=Resampling.nearest
        )
        return resampled, template_transform, template_crs

# --- Main Loop ---
for date in date_list:
    date_str = date.strftime("%Y%m%d")
    print(f"\n Processing date: {date_str}")
    date_rasters = {}

    # Set reference (chla) raster for resampling
    ref_shape = None
    ref_transform = None
    ref_crs = None

    # Load and clip all variables
    for var in variables:
        matching_files = [f for f in os.listdir(raster_folder) if date_str in f and var in f and f.endswith(".tif")]
        if not matching_files:
            print(f"  Missing file for {var} on {date_str}")
            continue

        file_path = os.path.join(raster_folder, matching_files[0])
        try:
            if var == 'CI' and ref_shape is None:
                arr, ref_transform, ref_crs = clip_and_resample_to_template(file_path, band=1)
                ref_shape = arr.shape
                print(f" Reference set from chla — shape: {ref_shape}")
            else:
                arr, _, _ = clip_and_resample_to_template(
                    file_path, band=1,
                    template_shape=ref_shape,
                    template_transform=ref_transform,
                    template_crs=ref_crs
                )
            date_rasters[var] = arr
            print(f"  Loaded {var} — shape: {arr.shape}")
        except Exception as e:
            print(f"  Failed to clip {var}: {e}")

    if len(date_rasters) != len(variables):
        print(f" Skipping {date_str} — missing base raster(s)")
        continue

    # Compute 5-day precipitation sum
    pr_stack = []
    for i in range(1, 6):
        past_date = date - timedelta(days=i)
        past_str = past_date.strftime("%Y%m%d")
        past_file = os.path.join(daily_climate_folder, f"{past_str}.tif")
        if not os.path.exists(past_file):
            print(f" Missing climate file for {past_str}")
            break

        with rasterio.open(past_file) as src:
            pr_band = src.read(1)
            pr_band[pr_band == src.nodata] = 0.0
            pr_resampled = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=pr_band,
                destination=pr_resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest
            )
            pr_stack.append(pr_resampled)

    if len(pr_stack) < 5:
        print(f"  Skipping {date_str} — insufficient precip history")
        continue

    date_rasters['pr5'] = np.sum(pr_stack, axis=0)

    # Step 4: Add tmmx and vs from today’s climate raster
    today_climate_file = os.path.join(daily_climate_folder, f"{date_str}.tif")
    if not os.path.exists(today_climate_file):
        print(f" Missing today's climate file: {date_str}")
        continue

    with rasterio.open(today_climate_file) as src:
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
            date_rasters[band_name] = resampled
            print(f" Loaded {band_name} — shape: {resampled.shape}")

    # Step 5: Write multiband raster
    all_vars = variables + ['pr5', 'tmmx', 'vs']
    stack = np.stack([date_rasters[v] for v in all_vars])

    meta = {
        'driver': 'GTiff',
        'height': ref_shape[0],
        'width': ref_shape[1],
        'count': len(all_vars),
        'dtype': 'float32',
        'crs': ref_crs,
        'transform': ref_transform
    }

    output_path = os.path.join(output_folder, f"multiband_{date_str}.tif")
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, var in enumerate(all_vars):
            dst.write(stack[i], i + 1)
            dst.set_band_description(i + 1, var)

    print(f" Saved: {output_path}")


# # --- Settings ---
# raster_folder = 'testing/'
# daily_climate_folder = 'testing/gridmet_test/'
# output_folder = 'output_tif/'
# shapefile_path = 'ALL_DRMs/ALL_DRMs.shp' 

# variables = ['CI', 'chla', 'PAR', 'Kd490', 'adg443', 'TSM']
# start_date = datetime.strptime("20240610", "%Y%m%d")
# end_date = datetime.strptime("20240619", "%Y%m%d")
# date_list = [(start_date + timedelta(days=x)) for x in range((end_date - start_date).days + 1)]

# # --- show plot to debug clip ---
# def show_raster_and_shapefile(src, gdf):
#     import matplotlib.pyplot as plt
#     from rasterio.plot import show

#     fig, ax = plt.subplots(figsize=(8, 8))
#     show((src, 1), ax=ax, title="Raster + Shapefile Overlay")
#     gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2)
#     plt.show()

# # --- Function to clip a raster to shapefile ---
# def clip_raster_to_shapefile(raster_path, band=1, debug_plot=False):
#     import geopandas as gpd
#     from rasterio.mask import mask
#     from shapely.geometry import mapping
#     from shapely.ops import unary_union
#     import rasterio
#     import numpy as np
#     import matplotlib.pyplot as plt

#     with rasterio.open(raster_path) as src:
#         if band > src.count:
#             raise ValueError(f"Band {band} requested, but {raster_path} has only {src.count} bands.")

#         gdf = gpd.read_file(shapefile_path).to_crs(src.crs)
#         buffered_geom = unary_union(gdf.geometry).buffer(0.0001)
#         shapes = [mapping(buffered_geom)]

#         try:
#             out_image, out_transform = mask(
#                 src, shapes, crop=True, indexes=band,
#                 filled=False, all_touched=True
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to clip {os.path.basename(raster_path)}: {e}")

#         # Normalize output to 3D if needed
#         if out_image.ndim == 2:
#             out_image = out_image[np.newaxis, :, :]

#         masked_band = out_image[0]
#         if np.ma.is_masked(masked_band) and masked_band.mask.all():
#             if debug_plot:
#                 print(f"All values are nodata after clipping {raster_path}")
#                 show_raster_and_shapefile(src, gdf)
#             raise ValueError(f"Clipped image has only nodata (masked) values: {raster_path}")

#         out_meta = src.meta.copy()
#         out_meta.update({
#             "height": out_image.shape[1],
#             "width": out_image.shape[2],
#             "transform": out_transform
#         })

#         return masked_band.filled(0.0), out_meta





# # --- Main loop ---
# for date in date_list:
#     date_str = date.strftime("%Y%m%d")
#     print(f"Processing date: {date_str}")

#     date_rasters = {}
#     meta = None

#     # Load and clip individual variable rasters
#     for var in variables:
#         matching_files = [f for f in os.listdir(raster_folder)
#                           if date_str in f and var in f and f.endswith(".tif")]

#         if not matching_files:
#             print(f"  Missing variable '{var}' on {date_str}")
#             continue

#         file_path = os.path.join(raster_folder, matching_files[0])
#         try:
#             data, clipped_meta = clip_raster_to_shapefile(file_path)
#             print(f"  Loaded and clipped {var} — shape: {data.shape}")
#             date_rasters[var] = data

#             if meta is None:
#                 meta = clipped_meta
#         except Exception as e:
#             print(f"  Failed to clip {var}: {e}")

#     if len(date_rasters) != len(variables):
#         print(f"  Skipping {date_str} — missing base raster(s)")
#         continue

#     # Sum 5-day precipitation (band 1)
#     pr_stack = []
#     for i in range(1, 6):
#         past_date = date - timedelta(days=i)
#         past_str = past_date.strftime("%Y%m%d")
#         past_file = os.path.join(daily_climate_folder, f"{past_str}.tif")

#         if not os.path.exists(past_file):
#             print(f"  Missing climate file for {past_str}")
#             break

#         try:
#             pr_band, _ = clip_raster_to_shapefile(past_file, band=1)
#             pr_stack.append(pr_band)
#         except Exception as e:
#             print(f"  Failed to clip pr from {past_str}: {e}")
#             break

#     if len(pr_stack) < 5:
#         print(f"  Skipping {date_str} — insufficient precip history")
#         continue

#     date_rasters['pr5'] = np.sum(pr_stack, axis=0)
#     print(f"  Computed pr5 — shape: {date_rasters['pr5'].shape}")

#     # Add today's tmmx (band 5) and vs (band 6)
#     today_climate_file = os.path.join(daily_climate_folder, f"{date_str}.tif")
#     if not os.path.exists(today_climate_file):
#         print(f"  Missing climate file: {date_str}")
#         continue

#     try:
#         date_rasters['tmmx'], _ = clip_raster_to_shapefile(today_climate_file, band=5)
#         date_rasters['vs'], _ = clip_raster_to_shapefile(today_climate_file, band=6)
#         print(f"  Loaded and clipped tmmx — shape: {date_rasters['tmmx'].shape}")
#         print(f"  Loaded and clipped vs — shape: {date_rasters['vs'].shape}")
#     except Exception as e:
#         print(f"  Failed to clip tmmx/vs: {e}")
#         continue

#     # Stack and write to multiband raster
#     all_vars = variables + ['pr5', 'tmmx', 'vs']
#     stack = np.stack([date_rasters[v] for v in all_vars])
#     print(f"  Final stack shape: {stack.shape}")

#     meta.update({
#         'count': stack.shape[0],
#         'dtype': stack.dtype
#     })

#     output_path = os.path.join(output_folder, f"multiband_{date_str}.tif")
#     with rasterio.open(output_path, 'w', **meta) as dst:
#         for i, var in enumerate(all_vars):
#             dst.write(stack[i], i + 1)
#             dst.set_band_description(i + 1, var)

#     print(f"  Saved: {output_path}")
