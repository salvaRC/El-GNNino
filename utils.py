import numpy as np
import xarray as xa
from ninolearn.IO.read_processed import data_reader
from sklearn.metrics import mean_squared_error


def rmse(y, preds):
    """
    The root-mean-squarred error (RMSE) for a given observation and prediction.

    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :rtype: float
    :return: The RMSE value
    """
    return np.sqrt(mean_squared_error(y, preds))


def season_to_int(season):
    """
    :param season: eg DJF for Dezember, January, February, etc.
    """
    dictio = {'DJF': 12,
              'JFM': 1,
              'FMA': 2,
              'MAM': 3,
              'AMJ': 4,
              'MJJ': 5,
              'JJA': 6,
              'JAS': 7,
              'ASO': 8,
              'SON': 9,
              'OND': 10,
              'NDJ': 11,
              "all": "all"
              }
    return dictio[season]


def cord_mask(data: xa.DataArray, is_flattened=False, flattened_too=False, lat=(-5, 5), lon=(190, 240)):
    """
    :param data:
    :param dim:
    :return:
    """
    oni_mask = {'time': slice(None), 'lat': slice(lat[0], lat[1]), 'lon': slice(lon[0], lon[1])}
    if flattened_too:
        flattened = data.copy() if is_flattened else data.stack(cord=['lat', 'lon']).copy()
        flattened[:, :] = 0
        flattened.loc[oni_mask] = 1  # Masked (ONI) region has 1 as value
        flattened_mask = (flattened[0, :] == 1)
        # print(np.count_nonzero(flattened_mask), '<<<<<<<<<<<<<<<<<')
        # flattened.sel(oni_mask) == flattened.loc[:, flattened_mask]
        return oni_mask, flattened_mask
    return oni_mask


def get_index_mask(data, index, flattened_too=False, is_data_flattened=False):
    """
    Get a mask to mask out the region used for  the ONI/El Nino3.4 or ICEN index.
    :param data:
    :param index: ONI or Nino3.4 or ICEN
    :return:
    """
    lats, lons = get_region_bounds(index)
    return cord_mask(data, lat=lats, lon=lons, flattened_too=flattened_too, is_flattened=is_data_flattened)


def get_region_bounds(index):
    if index.lower() in ["nino3.4", "oni"]:
        return (-5, 5), (190, 240)  # 170W-120W
    elif index.lower() == "icen":
        return (-10, 0), (270, 280)  # 90W-80W
    elif index.lower() in ["all", "world"]:
        return (-60, 60), (0, 360)  #
    else:
        raise ValueError("Unknown region/index")


def is_in_index_region(lat, lon, index="ONI"):
    lat_bounds, lon_bounds = get_region_bounds(index=index)
    if lat_bounds[0] <= lat <= lat_bounds[1]:
        if lon_bounds[0] <= lon <= lon_bounds[1]:
            return True
    return False


def check_chosen_coordinates(index, lon_min=190, lon_max=240, lat_min=-5, lat_max=5, ):
    if index in ["Nino3.4", "ONI"]:
        assert lat_min <= -5 and lat_max >= 5
        assert lon_min <= 190 and lon_max >= 240  # 170W-120W
    elif index == "ICEN":
        assert lat_min <= -10 and lat_max >= 0
        assert lon_min <= 270 and lon_max >= 280  # 90W-80W
    elif index[-3:] == "mon":
        pass
    else:
        raise ValueError("Unknown index")


def read_ssta(index, data_dir, get_mask=False, stack_lon_lat=True, resolution=2.5, dataset="ERSSTv5", fill_nan=0,
              start_date='1871-01', end_date='2019-12',
              lon_min=190, lon_max=240,
              lat_min=-5, lat_max=5,
              reader=None):
    """

    :param index: choose target index (e.g. ONI, Nino3.4, ICEN)
    :param start_date:
    :param end_date:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param reader: If a data_reader is passed, {start,end}_date and {lat, lon}_{min, max} will be ignored.
    :return:
    """
    if index in ["Nino3.4", "ONI"]:
        k = 5 if index == "Nino3.4" else 3
    elif index == "ICEN":
        k = 3
    elif index[-3:] == "mon":
        k = int(index[-4])  # eg 1mon
    else:
        raise ValueError("Unknown index")

    if reader is None:
        reader = data_reader(data_dir=data_dir,
                             startdate=start_date, enddate=end_date,
                             lon_min=lon_min, lon_max=lon_max,
                             lat_min=lat_min, lat_max=lat_max)
        check_chosen_coordinates(index, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

    resolution_suffix = f"{resolution}x{resolution}"
    ssta = reader.read_netcdf('sst', dataset=dataset, processed='anom', suffix=resolution_suffix)
    ssta = ssta.rolling(time=k).mean()[k - 1:]  # single months SSTAs --> rolling mean over k months SSTAs

    if stack_lon_lat:
        lats, lons = ssta.get_index('lat'), ssta.get_index('lon')
        ssta = ssta.stack(cord=['lat', 'lon'])
        ssta.attrs["Lons"] = lons
        ssta.attrs["Lats"] = lats
    if fill_nan is not None:
        if fill_nan == "trim":
            ssta_old_index = ssta.get_index('cord')
            ssta = ssta.dropna(dim='cord')
            print(f"Dropped {len(ssta_old_index) - len(ssta.get_index('cord'))} nodes.")
            # print("Dropped coordinates:", set(ssta_old_index).difference(set(ssta.get_index("cord"))))
            # print(flattened_ssta.loc["1970-01", (0, 290)]) --> will raise error
        else:
            ssta = ssta.fillna(fill_nan)

    if get_mask:
        index_mask, train_mask = get_index_mask(ssta, index=index, flattened_too=True, is_data_flattened=stack_lon_lat)
        train_mask = np.array(train_mask)
        return ssta, train_mask
    return ssta


def get_filename(args, transfer=False):
    args.save += args.target_month
    args.save += f"{args.horizon}lead_{args.index}" \
                 f"_{args.lat_min}-{args.lat_max}lats" \
                 f"_{args.lon_min}-{args.lon_max}lons" \
                 f"_{args.window}w{args.layers}L{args.gcn_depth}gcnDepth{args.dilation_exponential}dil" \
                 f"_{args.batch_size}bs{args.dropout}d{args.normalize}normed"
    args.save += "_prelu" if args.prelu else ""
    args.save += "_withHC" if args.use_heat_content else ""
    args.save += "_CNN_DATA_TRANSFER.pt" if transfer else ".pt"
    return args.save
