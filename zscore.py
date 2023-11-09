# %% packages
import xarray as xr
from datetime import datetime
from typing import Tuple
from cmcrameri import cm
import matplotlib.pyplot as plt
from pathlib import Path
# Filter out RuntimeWarnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
def zscore_ssm(
        input_layers:xr.Dataset,
        time_range:Tuple[datetime, datetime],
        xr_var_name: str = "S1SSM",
        xr_time_name: str = "date",
        smoothing: bool = True,
        ) -> xr.Dataset:
        """
        This function takes an xarray dataset as input layers. 
        In particular SSM data and apply a SWDI based approach 
        on the layer. The returned layer is the Soil Water Deficiency Index for the timeperiod.

        Parameters:
        input_layers (xr.Dataset): Contains every soil moisture layers stored, time dimension name should be "date"
        time_range (Tuple[datetime, datetime]): Time period to process the Z-score value,
        xr_var_name (str): Name of the variable within the xr.Dataset (default to S1SSM),
        xr_time_name (str): Name of the temporal variable within the xr.Dataset
        scaling (int | float): scaling of the input data,
        smoothing (bool): Does the dataset require smoothing (21 days as default),

        Returns:
        SWDI_layers (xr.Dataset): Dataset containing the resulting SWDI values
        """
        # check if the variables are correct
        assert xr_var_name in list(input_layers.keys()), (
            f"Variable name '{xr_var_name}' not in the dataset"
            )
        assert xr_time_name in list(input_layers.coords), (
            f"Temporal name '{xr_time_name}' in the dataset"
            )

        # initialize z-score placeholder
        z_score_cb_ds = None
        # Filter the dataset for values within the time range
        filtered_dataset = input_layers.sel(date=slice(time_range[0], time_range[1]))

        # If required, smooth
        if smoothing:
            filtered_dataset = filtered_dataset.rolling(date=21, min_periods=1).mean()

        # add the doy for calculation of monthly std
        filtered_dataset['day_of_year'] = filtered_dataset[xr_time_name].dt.dayofyear

        # get the monthly z-score
        for month in range(1, 13):
            monthly_dataset = filtered_dataset.where(
                filtered_dataset.date.dt.month.isin([month]), 
                drop=True
                )
            # statistics
            print(f"For month {month}: Number of observations=")
            print(len(monthly_dataset.date.values))
            monthly_std_xr = monthly_dataset[xr_var_name].std(dim=xr_time_name)
            monthly_mean_xr = monthly_dataset[xr_var_name].mean(dim=xr_time_name)

            # process z-score
            z_score_ds = (monthly_dataset[xr_var_name] 
                          - monthly_mean_xr)/monthly_std_xr

            # combine
            if z_score_cb_ds is None:
                 z_score_cb_ds = z_score_ds
            else:
                 z_score_cb_ds = xr.concat(
                      [z_score_cb_ds, z_score_ds],
                      dim=xr_time_name, compat="identical")
                      

        # Assuming your DataArray is named 'data_array'
        z_score_cb_ds.name = "Z-score"
        return z_score_cb_ds

# %%
if __name__ == "__main__":  
    regions = [
        "Massinga",
        "Mabote",
        "Chokwé",
        "Govuro",
        "Buzi",
        "Muanza"]
# check out results
for region in regions:
    print(f"Processing of {region}")
    input_layers = xr.open_dataset(
        f"/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_S1SSM_scenes/{region}_SSM.nc",
        )
    
    test = zscore_ssm(
        input_layers=input_layers,
        time_range=[datetime(2016, 1, 1), datetime(2022, 12, 31)],
        smoothing=True,
        xr_var_name = "S1SSM",
        xr_time_name = "date")

    output_dump = Path("/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/temporary_dump")
    output_xr = Path("/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_DROUGHT_scenes")
    # save
    test.to_netcdf(output_xr / f"{region}_zscore.nc")
