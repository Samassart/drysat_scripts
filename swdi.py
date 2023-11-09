# %% packages
import xarray as xr
from datetime import datetime
from typing import Tuple
import rioxarray as rio
import rasterio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# %% functions
def swdi_ssm(
        input_layers:xr.Dataset,
        wilting_point:xr.DataArray,
        field_capacity:xr.DataArray,
        porosity:xr.DataArray | float,
        time_range:Tuple[datetime, datetime],
        xr_var_name: str = "S1SSM",
        xr_time_name: str = "date",
        scaling:int | float = 100,
        smoothing: bool = True,
        ) -> xr.Dataset:
        """
        This function takes an xarray dataset as input layers. 
        In particular SSM data and apply a SWDI based approach 
        on the layer. The returned layer is the Soil Water Deficiency Index for the timeperiod.

        Parameters:
        input_layers (xr.Dataset): Contains every soil moisture layers stored, time dimension name should be "date"
        wilting_point (xr.DataArray): The layer of SSM when the wilting point is reached
        field_capacity (xr.DataArray | float): The layer of SSM when the field capacity is reached, if float, set as constant
        porosity (xr.DataArray): The porosity layer to go from relative SSM to absolute SSM,
        time_range (Tuple[datetime, datetime]): Time period to process the SWDI value,
        xr_var_name (str): Name of the variable within the xr.Dataset (default to S1SSM),
        scaling (int | float): scaling of the input data,
        smoothing (bool): Does the dataset require smoothing (21 days as default) 

        Returns:
        SWDI_layers (xr.Dataset): Dataset containing the resulting SWDI values
        """
        # Filter the dataset for values within the time range
        filtered_dataset = input_layers.sel(date=slice(time_range[0], time_range[1]))

        # set-up the SSM layer to a state to be used, store everything in the input layer

        if smoothing:
            filtered_dataset[xr_var_name] = filtered_dataset[xr_var_name].rolling(date=21, min_periods=1).mean()

        # porosity can be float or entire dataframe
        filtered_dataset["S1SSM_relative"] = (filtered_dataset[xr_var_name]/scaling) * porosity

        # process the SWDI and only store the values below 0
        filtered_dataset["SWDI"] = (
            filtered_dataset["S1SSM_relative"]*100 -  field_capacity) / (field_capacity - wilting_point)
        filtered_dataset["SWDI"] = filtered_dataset["SWDI"].where(filtered_dataset["SWDI"] < 0, 0.01)
        filtered_dataset["SWDI"] = filtered_dataset["SWDI"].where(filtered_dataset["SWDI"] != 0, np.nan)
        return filtered_dataset["SWDI"]

# %% process
if __name__ == "__main__":  
    regions = [
        "Massinga",
        "Mabote",
        "Chokwé",
        "Govuro",
        "Buzi",
        "Muanza"]
    for region in regions:
        # paths
        sg_dir = Path("/home/smassart/shares/radar/Projects/DrySat/07_data/CGLS_grid_standard/EAST_AFRICA/STATIC/SOILGRID_drysat")
        wilting_point_layer = rio.open_rasterio(sg_dir / "R20190101_--------_WV1500---_0-5CM---------_---_-----_------_EASTAFRICA.tiff")
        field_capacity_layer = rio.open_rasterio(sg_dir / "R20190101_--------_WV0010---_0-5CM---------_---_-----_------_EASTAFRICA.tiff")
        porosity_layer = rio.open_rasterio(sg_dir / "R20190101_--------_PORO-26--_0-5CM---------_---_-----_------_EASTAFRICA.tiff")
        # load the relevant tile 
        target_crs = rasterio.crs.CRS.from_string(wilting_point_layer.spatial_ref.crs_wkt)
        if region == "Chokwé":
            tile = 'E066N018T6'
        else:
            tile = 'E066N024T6'

        print(f"Processing of {region}")
        input_layers = xr.open_dataset(
            f"/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_S1SSM_scenes/{region}_SSM.nc",
            )
        
        # crop the layers for swdi
        wilting_point_layers_region = wilting_point_layer.rio.reproject_match(input_layers).isel(band=0)/10
        field_capacity_layers_region = field_capacity_layer.rio.reproject_match(input_layers).isel(band=0)/10
        porosity_layer_region = porosity_layer.rio.reproject_match(input_layers).isel(band=0)/100

        # set-up function
        test = swdi_ssm(
            input_layers=input_layers,
            wilting_point=wilting_point_layers_region,
            field_capacity=field_capacity_layers_region,
            porosity=porosity_layer_region,
            time_range=[datetime(2016, 1, 1), datetime(2022, 12, 31)],
            xr_var_name ="S1SSM",
            xr_time_name="date",
            scaling=100,
            smoothing=True
            )

        output_xr = Path("/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_DROUGHT_scenes")
        test.to_netcdf(output_xr / f"{region}_swdi.nc")
