import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xa
from utils import read_ssta, is_in_index_region
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



def heatmap_of_edges(file_path=None, model=None, min_weight=0.1, reader=None, data_dir=None, thresh=(0, 0),
                     save_to=None, plot_heatmap=True, from_or_to_ONI="from", set_title=True, region="ONI"):
    """
    The adaptively learnt edges by MTGNN are unidirectional!
    :param file_path: path to saved torch model
    :param data_dir if None, args.dir is used!
    :param min_weight: threshold for the adjacency matrix edge weights to be plotted
    :param from_or_to_ONI: from, towards, or both
    :return:
    """
    import cartopy.crs as ccrs
    import torch
    assert file_path is not None or model is not None
    if model is None:
        model = torch.load(file_path, map_location='cpu')
    model.eval()
    args = model.args

    if reader is not None:
        ssta = read_ssta(args.index, data_dir, get_mask=False, stack_lon_lat=True, resolution=args.resolution,
                         reader=reader)
        coordinates = ssta.indexes["cord"]
        lats, lons = ssta.attrs["Lats"], ssta.attrs["Lons"]
    else:
        from utils import load_cnn_data
        data_dir = data_dir or args.data_dir
        _, _, GODAS, coordinates = load_cnn_data(window=args.window, lead_months=args.horizon, lon_min=args.lon_min,
                                                 lon_max=args.lon_max, lat_min=args.lat_min, lat_max=args.lat_max,
                                                 data_dir=data_dir, use_heat_content=args.use_heat_content,
                                                 return_new_coordinates=True)
        lats, lons = GODAS[0].attrs["Lats"], GODAS[0].attrs["Lons"]

    adj = model.adj_matrix.numpy()
    print("# Nonzero Edges:", np.count_nonzero(adj))

    lat_len, lon_len = len(lats), len(lons)

    incoming_edge_heat = xa.DataArray(np.zeros((lat_len, lon_len)), coords=[("lat", lats), ("lon", lons)])
    outgoing_edge_heat = xa.DataArray(np.zeros((lat_len, lon_len)), coords=[("lat", lats), ("lon", lons)])
    for i, (lat_i, lon_i) in enumerate(coordinates):
        incoming_edge_heat.loc[lat_i, lon_i] = np.sum(adj[:, i])
        outgoing_edge_heat.loc[lat_i, lon_i] = np.sum(adj[i, :])


    incoming_edge_heat = xa.where(incoming_edge_heat < thresh[0], min_weight, incoming_edge_heat)
    outgoing_edge_heat = xa.where(outgoing_edge_heat < thresh[1], min_weight, outgoing_edge_heat)
    fig = plt.figure()
    cm = 180
    from_or_to_ONI = from_or_to_ONI.lower()
    if from_or_to_ONI != "both":
        ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
    else:
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=cm))

    minlon = -180 + cm
    maxlon = +179 + cm
    ax1.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())

    if region.lower() in ["all", "world"]:
        title1 = f"Heatmap of summed incoming edge weights"
        title2 = f"Heatmap of summed outgoing edge weights"
    else:
        title1 = f"Heatmap of summed edge weights that come from {region} region"
        title2 = f"Heatmap of summed edge weights that point towards {region} region"
    if set_title:
        ax1.set_title(title1) if from_or_to_ONI in ["both", "from"] else ax1.set_title(title2)

    if from_or_to_ONI == "from":
        loop_over = zip([ax1], [incoming_edge_heat])
    elif from_or_to_ONI == "towards":
        loop_over = zip([ax1], [outgoing_edge_heat])
    else:
        ax2 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree(central_longitude=cm))
        ax2.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())
        if set_title:
            ax2.set_title(title2)
        loop_over = zip([ax1, ax2], [incoming_edge_heat, outgoing_edge_heat])

    for ax, heat in loop_over:
        if plot_heatmap:
            im = ax.pcolormesh(lons, lats, heat, cmap="Reds", transform=ccrs.PlateCarree())

            fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01) if from_or_to_ONI else fig.colorbar(im, ax=ax, pad=0.01)
        else:
            im = ax.contourf(lons, lats, heat, transform=ccrs.PlateCarree(), alpha=0.85, cmap="Reds", levels=100)
            fig.colorbar(im, ax=ax, pad=0.01)

        '''        map = Basemap(projection='cyl', llcrnrlat=-55, urcrnrlat=60, resolution='c', llcrnrlon=0, urcrnrlon=380, ax=ax)
        map.drawcoastlines(linewidth=0.2)
        map.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=6.5, color='grey', linewidth=0.2)
        map.drawmeridians(np.arange(0., 380., 60.), labels=[0, 0, 0, 1], fontsize=6.5, color='grey', linewidth=0.2)
        '''

    ax1.coastlines()
    if from_or_to_ONI == "both":
        ax2.coastlines()


    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def create_filepath(prefix="", suffix="", **kwargs):
    if prefix is None:
        return None
    string = prefix
    for key, name in kwargs.items():
        string += f"_{name}{key}"
    return string + suffix


def plot_time_series(data, *args, labels=["timeseries"], time_steps=None, data_std=None, linewidth=2,
                     timeaxis="time", ylabel="Nino3.4 index", plot_months=False, show=True, save_to=None):
    if time_steps is not None:
        time = time_steps
    elif isinstance(data, xa.DataArray):
        time = data.get_index(timeaxis)
    else:
        time = np.arange(0, data.shape[0], 1)
    series = np.array(data)
    plt.figure()
    plt.plot(time, series, label=labels[0], linewidth=linewidth)
    if data_std is not None:
        plt.fill_between(time, series - data_std, series + data_std, alpha=0.25)
    minimum, maximum = np.min(data), np.max(data)
    for i, arg in enumerate(args, 1):
        minimum, maximum = min(minimum, np.min(arg)), max(maximum, np.max(np.max(arg)))
        try:
            plt.plot(time, arg, label=labels[i], linewidth=linewidth)
        except ValueError as e:
            raise ValueError("Please align the timeseries to the same time axis.", e)
        except IndexError:
            raise IndexError("You must pass as many entries in labels, as there are time series to plot")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.yticks(np.arange(np.round(minimum - 0.5, 0), np.round(maximum + 0.51, 0), 0.5))
    nth_month = 10
    if plot_months and isinstance(time[0], pd._libs.tslibs.Timestamp):
        xticks, year_mon = time[::nth_month][:-1], [f"{date.year}-{date.month}" for date in time[::nth_month][:-1]]
        xticks = xticks.append(pd.Index([time[-1]]))
        year_mon.append(f"{time[-1].year}-{time[-1].month}")  # add last month
        plt.xticks(ticks=xticks, labels=year_mon, rotation=20)
    plt.legend()
    plt.grid()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')

    if show:
        plt.show()
    return time

