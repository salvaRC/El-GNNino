from os.path import join
import pandas as pd
import xarray as xr
import numpy as np
import gc


# TODO: Write a routine that generates this list
csv_vars = ['nino3.4M', 'nino3.4S', 'wwv']


def generateFileName(variable, dataset, processed='', suffix=None):
    """
    generates a file name
    """
    filenamelist = [variable, dataset, processed]

    # remove ''  entries from list
    filenamelist = list(filter(lambda a: a != '', filenamelist))

    filename = '_'.join(filenamelist)

    if suffix is not None:
        filename = '.'.join([filename, suffix])

    return filename


class data_reader(object):
    def __init__(self, data_dir, startdate='1980-01', enddate='2020-08',
                 lon_min=120, lon_max=280, lat_min=-30, lat_max=30):
        """
        Data reader for different kind of El Nino related data.

        :param startdate:year and month from which on data should be loaded
        :param enddate: year and month to which data should be loaded
        :lon_min: eastern boundary of data set in degrees east
        :lon_max: western boundary of data set in degrees east
        :lat_min: southern boundary of data set in degrees north
        :lat_max: northern boundary of data set in degrees north
        """
        self.startdate = pd.to_datetime(startdate)
        self.enddate = pd.to_datetime(enddate) + pd.tseries.offsets.MonthEnd(0)
        self.data_dir = data_dir + "processed/"

        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def __del__(self):
        gc.collect()

    def shift_window(self, month=1):
        self.startdate = self.startdate + pd.DateOffset(months=month)
        self.enddate = self.enddate + pd.DateOffset(months=month) \
                       + pd.tseries.offsets.MonthEnd(0)

    def split(self, data, train_end_date, val_end_date, flatten=True):
        train_end_date = pd.to_datetime(train_end_date) + pd.tseries.offsets.MonthEnd(0)
        val_end_date = pd.to_datetime(val_end_date) + pd.tseries.offsets.MonthEnd(0)
        train = data.loc[self.startdate:train_end_date, :, :]
        val = data.loc[train_end_date:val_end_date, :, :]
        test = data.loc[val_end_date:self.enddate, :, :]
        if flatten:
            train, val, test \
                = train.stack(cord=['lat', 'lon']), val.stack(cord=['lat', 'lon']), test.stack(cord=['lat', 'lon'])
        train, val, test = np.array(train.fillna(0)), np.array(val.fillna(0)), np.array(test.fillna(0))
        return train, val, test


    def read_netcdf(self, variable, dataset='', processed='', suffix="", chunks=None):
        """
        wrapper for xarray.open_dataarray.

        :param variable: the name of the variable
        :param dataset: the name of the dataset
        :param processed: the postprocessing that was applied
        :param chunks: same as for xarray.open_dataarray
        """
        filename = generateFileName(variable, dataset,
                                    processed=suffix + processed, suffix="nc")

        data = xr.open_dataarray(join(self.data_dir, filename), chunks=chunks)

        regrided = ['GODAS', 'ERSSTv5', 'ORAS4', 'NODC', 'NCAR']

        if processed == 'meanclim':
            return data

        else:
            self._check_dates(data, f'{filename[:-3]}')
            if dataset not in regrided and dataset != 'ORAP5' and dataset != 'GFDL-CM3':
                return data.loc[self.startdate:self.enddate,
                       self.lat_max:self.lat_min,
                       self.lon_min:self.lon_max]

            elif dataset in regrided or dataset == 'GFDL-CM3':
                return data.loc[self.startdate:self.enddate,
                       self.lat_min:self.lat_max,
                       self.lon_min:self.lon_max]
            elif dataset == 'ORAP5':
                return data.loc[self.startdate: self.enddate, :, :].where(
                    (data.nav_lat > self.lat_min) &
                    (data.nav_lat < self.lat_max) &
                    (data.nav_lon > self.lon_min) &
                    (data.nav_lon < self.lon_max),
                    drop=True)


    def _check_dates(self, data, name):
        """
        Checks if provided start and end date are in the bounds of the data
        that should be read.
        """
        if isinstance(data, xr.DataArray):
            if self.startdate < data.time.values.min():
                raise IndexError("The startdate is out of\
                                 bounds for %s data!" % name)
            if self.enddate > pd.to_datetime(data.time.values.max()) + pd.tseries.offsets.MonthEnd(0):
                print(data.time.values.max())
                print(self.enddate)
                raise IndexError("The enddate is out of bounds for %s data!" % name)

        if isinstance(data, pd.DataFrame):
            if self.startdate < data.index.values.min():
                msg = f"The startdate is out of bounds for {name} data!"
                raise IndexError(msg)
            if self.enddate > pd.to_datetime(data.index.values.max()) + pd.tseries.offsets.MonthEnd(0):
                print(self.enddate)
                print(data.index.values.max())
                raise IndexError("The enddate is out of bounds for %s data!" % name)


if __name__ == "__main__":
    reader = data_reader(startdate="1981-01", enddate='2018-12',
                         lon_min=120, lon_max=380, lat_min=-30, lat_max=30)

    data = reader.read_netcdf('sshg', dataset='GODAS', processed='anom')
    data2 = reader.read_netcdf('zos', dataset='GFDL-CM3', processed='anom')
