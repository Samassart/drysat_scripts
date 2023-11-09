# %%
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple
import pandas as pd
from pathlib import Path
from equi7grid.equi7grid import Equi7Grid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def plot_timeseries(
    drought_pd:pd.DataFrame,
    ssm_pd:pd.DataFrame,
    time_range:Tuple[datetime, datetime],
  
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        This function cleanly plots timeseries merging 
        drought indicators and general timeseries dynamics

        Parameters:
        drought_pd (pd.DataFrame): Contain the given drought indicators (in espg:4326)
        ssm_pd (pd.DataFrame): Contain the original ssm product
        time_range (Tuple[datetime, datetime]): Time period to process the Z-score value,

        Returns:
        AxesSubplot of clean looking results
        """
        drought_pd[drought_pd > 0] = 0
        drought_var_name = drought_pd.columns[0]
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(
                ssm_pd.index, 
                ssm_pd, 
                color='tab:blue', 
                label='SSM from Sentinel-1', 
                linewidth=0.9
                )
        ax.set_xlabel('Date')
        ax.set_ylabel('Soil Moisture', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.legend(loc='upper left')

        ax1 = ax.twinx()
        ax1.bar(
                drought_pd.index,
                drought_pd[drought_var_name],
                color="tab:red",
                alpha=0.5,
                width=1.5
                )
        ax1.invert_yaxis()
        ax1.set_ylabel(drought_var_name, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')


        return fig, ax




def df_extraction(
    input_layers:xr.Dataset,
    lat: float,
    lon: float  
    ) -> pd.DataFrame:
        """
        This convert a xarray dataset into a panda dataframe (in )
        for a given set of coordinate

        Parameters:
        input_layers (xr.Dataset): given Dataset
        lat (float): longitude
        lon (float): longitude

        Returns:
        pd.DataFrame for the given lat/lon
        """
        e7 = Equi7Grid(500)
        ijtile = e7.lonlat2ij_in_tile(lon, lat, lowerleft=True)
        tile = ijtile[0].split("_")[1]
        x_val = float(ijtile[1]) * 500 
        y_val = float(ijtile[2]) * 500 

        e_val = float(tile[2:4] + "00000")
        n_val = float(tile[6:8] + "00000")
        # actual pixel value
        x_point = x_val + e_val
        y_point = y_val + n_val

        # select nearest point
        filtered_ds = input_layers.sel(x = x_point, y = y_point, method = 'nearest')
        var_name = list(filtered_ds.keys())[0]
        variable_values = filtered_ds[var_name].values
        temporal_values = filtered_ds.date.values

        # make the pd.dataframe
        pd_to_return = pd.DataFrame(
                {var_name: variable_values},
                index=temporal_values
        )
        return pd_to_return





def mean_monthly(
        input_layers:xr.Dataset,     
        ) -> pd.DataFrame:
        monthly_means = {}
        for date in input_layers['date']:
                month_year = f"{date.dt.month.item():02d}-{date.dt.year.item()}"
                # Create a new dataset containing data for the specified 'month_year'.
                filtered_ds = input_layers.sel(date=input_layers['date'].dt.strftime('%m-%Y') == month_year)
                name_var = list(filtered_ds.keys())[0]
                if month_year not in monthly_means:
                        values_to_avg = filtered_ds[name_var].values
                        values_to_avg[values_to_avg > 0] = np.nan
                        values_to_avg[values_to_avg < -10] = np.nan
                        mean_month = np.nanmean(values_to_avg)
                        dt_month_year = datetime.strptime(month_year, "%m-%Y")
                        monthly_means[dt_month_year] = mean_month
        
        data_to_return = pd.DataFrame({"mean_drought": monthly_means.values()}, index=monthly_means.keys())
        #data_to_return.replace([np.inf, -np.inf], 0, inplace=True)
        return data_to_return





if __name__ == "__main__":
        drought_type = "zscore"
        test_points = {
            "Massinga": [34.8205,-22.6407],
            "Mabote": [33.784,-21.883],
            "Chokwé": [33.0523,-24.6636],
            "Govuro": [34.7481,-21.2771],
            "Buzi": [34.227,-20.250],
            "Muanza": [35.1657,-19.2203],
            }
        
        categorical_drought_swdi = {
                (-5, -1): 4,
                (-1, -0.75): 3,
                (-0.75, -0.45): 2,
                (-0.45, -0.3): 1,
                (-0.3, 5): 0,
        }
        categorical_drought_zscore = {
                (-5, -1.4): 4,
                (-1.4, -1): 3,
                (-1, -0.7): 2,
                (-0.7, -0.5): 1,
                (-0.5, 5): 0,
        }

        for region in test_points:
                # Load the xarray dataset of the region
                drought_idx_path = Path("/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_DROUGHT_scenes")
                ssm_idx_path = Path("/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/cropped_S1SSM_scenes")
                output_figures = Path("/data/Drysat/ASCAT_SEASONALITY/clean_code_for_droughts/figures/")
                dataset_drought_name = f"{region}_{drought_type}.nc"
                dataset_ssm_name = f"{region}_SSM.nc"

                dataset_drought = xr.open_dataset(drought_idx_path / dataset_drought_name)
                dataset_ssmoist = xr.open_dataset(ssm_idx_path / dataset_ssm_name)
                
                lon, lat = test_points[region][0], test_points[region][1]

                drought_pd = df_extraction(
                        dataset_drought,
                        lat,
                        lon
                        )
                
                ssm_pd = df_extraction(
                        dataset_ssmoist,
                        lat,
                        lon
                )
                
                # now use the plotting function
                fig, ax = plot_timeseries(
                        drought_pd = drought_pd.dropna(),
                        ssm_pd = ssm_pd.dropna(),
                        time_range=[datetime(2016, 1, 1), datetime(2022, 12, 31)]
                    )
                plt.title(f"Random point for {region} - {lat, lon}")
                plt.savefig(output_figures / "timeseries_region" / f"{region}_timeseries_{drought_type}.png")
                plt.close()
                # make a drought heatmap
                drought_heat = mean_monthly(
                        dataset_drought
                        )
                
                drought_heat["month"] = drought_heat.index.month
                drought_heat["year"] = drought_heat.index.year


                drought = (
                        drought_heat
                        .pivot(index="year", columns="month", values="mean_drought")
                        )
                
                
                # Draw a heatmap with the numeric values in each cell
                f, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(drought, annot=True, linewidths=.5, ax=ax)
                plt.title(f"Monthly avg of {drought_type} for {region}")
                plt.savefig(output_figures / "heatmap_region" / f"{region}_heatmap_continuous_{drought_type}.png")


                # same but categorical
                if drought_type == "swdi":
                        # Convert the dataframe to categorical
                        for col in drought.columns:
                                bin_edges = [-100] + [range_[1] for range_ in categorical_drought_swdi.keys()]
                                
                                labels = list(categorical_drought_swdi.values())
                                drought[col] = pd.cut(
                                        drought[col], 
                                        bins=bin_edges,
                                        labels=labels,
                                        include_lowest=True, right=False).astype(int)
                                        
                        print(drought)
                        # Draw a heatmap with the numeric values in each cell
                        f, ax = plt.subplots(figsize=(12, 6))
                        sns.heatmap(drought, annot=True, linewidths=.5, ax=ax, cmap='Reds')

                        plt.title(f"Monthly avg of {drought_type}-categorized for {region}")
                        plt.savefig(output_figures / "heatmap_region" / f"{region}_heatmap_category_{drought_type}.png")


                # same but categorical
                if drought_type == "zscore":
                        # Convert the dataframe to categorical
                        for col in drought.columns:
                                bin_edges = [-100] + [range_[1] for range_ in categorical_drought_zscore.keys()]
                                
                                labels = list(categorical_drought_zscore.values())
                                drought[col] = pd.cut(
                                        drought[col], 
                                        bins=bin_edges,
                                        labels=labels,
                                        include_lowest=True, right=False).astype(int)
                                        
                        print(drought)
                        # Draw a heatmap with the numeric values in each cell
                        f, ax = plt.subplots(figsize=(12, 6))
                        sns.heatmap(drought, annot=True, linewidths=.5, ax=ax, cmap='Reds')

                        plt.title(f"Monthly avg of {drought_type}-categorized for {region}")
                        plt.savefig(output_figures / "heatmap_region" / f"{region}_heatmap_category_{drought_type}.png")

 # %%
