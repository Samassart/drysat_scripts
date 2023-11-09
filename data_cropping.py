"""
This script consists in processing z-score based on S1 SSM for the study area
of the Drysat project
"""

# %% packages and static parameters
import warnings
# Filter out the Proj-related warning
warnings.filterwarnings("ignore", message="PROJ: internal_proj_identify")
from pathlib import Path
import geopandas as gpd
import xarray as xr
import rioxarray as rio
from datetime import datetime
import numpy as np
import pyproj

# %% clipping scene function
def clipping_scene(
        list_of_scenes:list,
        list_of_dates:list,
        shapefile:gpd.GeoDataFrame,
        output_time_name:str = "Time",
        output_var_name:str = "Default",
        nodata_handling:(int | float) = - 9999
    ) -> xr.Dataset:
    """
    This function a list of geotiffs paths as input
    Iteratively open them, crop them and store them into xarray.Dataset

    Parameters:
    list_of_scenes (List): list of the scenes to crop
    list_of_dates (List): list of assiocated dates
    shapefile (gpd.gpd.GeoDataFrame): Shapefile used to crop our data
    output_time_name (str): Name of the time dimension for the output dataset
    output_var_name (str): Name of the variable for the output dataset

    Returns:
    Output_dataset (xr.Dataset): Dataset containing the cropped scenes
    """
    # list dimension check
    if len(list_of_dates) != len(list_of_scenes):
        raise ValueError(f"Cannot match scenes number ({len(list_of_scenes)})"
                        f" to dates number ({len(list_of_dates)})!")
    combined_layers = None

    for scene, date in zip(list_of_scenes, list_of_dates):
        print(f"Processing of {shapefile.NAME_2.values[0]} at date {date}")
        # load via rio and crop
        scene_xr = rio.open_rasterio(scene)

        # check first if the pyproj is up to date
        if "custom_proj" not in locals():
            # take care of the equi7grid custom proj
            custom_proj = pyproj.Proj(scene_xr.spatial_ref.crs_wkt)

        cropped_scene_xr = scene_xr.rio.clip(
            shapefile.geometry.values, 
            shapefile.crs, 
            drop=True, 
            invert=False,
            )

        # set-up filenaming and metadata
        allnan = np.all(cropped_scene_xr.values == nodata_handling)
        cropped_scene_xr.name = output_var_name

        if allnan:
            continue

        else:
            cropped_scene_xr = cropped_scene_xr.where(
                cropped_scene_xr != nodata_handling, np.nan
                )
            cropped_scene_xr[output_time_name] = date
            if combined_layers is None:
                combined_layers = cropped_scene_xr

            else:
                combined_layers =  xr.concat(
                    [combined_layers, cropped_scene_xr], dim=output_time_name
                    )
    
    combined_layers = combined_layers.isel(band=0)
    return combined_layers


# %% actually run the code 
if __name__ == "__main__":

    # static variables
    regions = [
        "Massinga",
        "Mabote",
        "Chokwé",
        "Govuro",
        "Buzi",
        "Muanza"]

    for region_analyzed in regions:
        # tiling 
        if region_analyzed == "Chokwé":
            tile = 'E066N018T6'
        else:
            tile = 'E066N024T6'

        # pathings
        shp_directory = Path(f"/data/Drysat/Datasets/shapefiles/{region_analyzed}.shp")
        s1_directory = Path(f"/data/Drysat/Datasets/SSM_PROCESSING_V2/SSM_EXTR/V1M1R1/AF500M/{tile}")
        gpd_shp = gpd.read_file(shp_directory)

        list_of_s1_scene = sorted(list(s1_directory.glob("*.tif")))
        datetime_list = [
            datetime.strptime(
                Path(file).stem.split('_')[1], "%Y%m%dT%H%M%S"
                ) for file in list_of_s1_scene]



        combined_layers = clipping_scene(
            list_of_s1_scene,
            datetime_list,
            gpd_shp,
            output_time_name="date",
            output_var_name="S1SSM"
        )
        combined_layers.to_netcdf(f"/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_S1SSM_scenes/{region_analyzed}_SSM.nc")
# %%
