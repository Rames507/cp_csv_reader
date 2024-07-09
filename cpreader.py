import csv
import os
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CPReader:
    def __init__(self, file: str | os.PathLike, gcc: float = 0.85, ccc: float = 1.0, tcc: float = 1.0):
        """
        # TODO
        Important: Internally time is in seconds * 10 to avoid floating point values.
        :param file: Path to input file.
            A CSV file with 3 columns: time, cycle_id, angle_id (e.g.: "1.5,17,6943")
        """
        self.gcc = gcc
        self.ccc = ccc
        self.tcc = tcc

        with open(file, "r") as f:
            reader = csv.reader(f)
            rows = [[float(row[0]) * 10, *map(int, row[1:3])] for row in reader]

        self.raw_data: pd.DataFrame = pd.DataFrame(
            rows, columns=("time", "cycle_id", "angle_id"), dtype=np.uint32
        )
        self.table: pd.DataFrame = self.range1(self.raw_data)
        self.table["range2"] = self.range2(self.table["range1"])

        # the slice is from 3 to 7 (both inclusive) since the first 3 values are always 'NaN'
        self.irange = np.mean(self.table.loc[3:7, "range2"])  # noqa

        self.table = self.table.merge(
            # TODO: what range is used for calculation of CA?
            self.clot_amplitude(self.table["range2"]),
            left_index=True,
            right_index=True,
        )

        pos = 20
        print(self.table[pos:pos+10])
        print(self.irange)

    @staticmethod
    def get_cycles(raw_data: pd.DataFrame) -> Iterable[pd.DataFrame]:
        """
        Splits the  dataframe into valid cycles of 50 elements (from cycle_id 0 to 49).
        Will ignore incomplete cycles.
        :param raw_data: A dataframe with "cycle_id" column
        """
        # determines the starting point of the first cycle
        # (important if input file doesn't start at cycle_id == 0)
        cycle_starting_points: pd.DataFrame = raw_data.index[raw_data["cycle_id"] == 0]
        start_idx: np.int64 = cycle_starting_points[0]

        for i in range(start_idx, len(raw_data), 50):
            cycle = raw_data.iloc[i: i + 50]
            if len(cycle) == 50:
                yield cycle

    def range1(self, raw_data_df: pd.DataFrame) -> pd.DataFrame:
        columns = ("time", "range1")
        range1_dict = {key: [] for key in columns}
        for cycle in self.get_cycles(raw_data_df):
            cycle_id = cycle["cycle_id"]
            median_values1 = cycle.loc[(7 <= cycle_id) & (cycle_id <= 19)]
            median_values2 = cycle.loc[(32 <= cycle_id) & (cycle_id <= 44)]
            median1 = np.median(median_values1["angle_id"])
            median2 = np.median(median_values2["angle_id"])
            range1 = np.abs(median1 - median2)

            range1_dict["range1"].append(range1)
            range1_dict["time"].append(cycle.iloc[0]["time"])

        range1_df = pd.DataFrame(range1_dict, dtype=np.uint32)
        return range1_df

    @staticmethod
    def range2(series: pd.Series) -> pd.Series:
        """
        Range2:
        Smoothes a series by taking the median of 7 consecutive values (3 preceeding, 3 following).
        First and last 3 values will be 'NaN'.
        :param series:
        :return:
        """
        new_series = pd.Series()
        for idx, item in enumerate(series):
            # the first and last 3 entries don't get a value
            if idx <= 2 or idx >= (len(series) - 3):
                val = np.nan
            else:
                val = np.median(series[idx - 3: idx + 4])

            new_series[idx] = val
        return new_series

    def clot_amplitude(self, range_: pd.Series) -> pd.DataFrame:
        """
        The clot amplitude is calculated as follows (CA(t) represents the CA at timepoint (t)):
        CA(t) = (iRange-Range(t))  / iRange * GCC * CCC * TCC

        :param range_: A series of Range values to calculate the clot amplitude from
        :return: A dataframe with different clot amplitude values calculated.
        """
        # CA(t) = (iRange-Range(t))  / iRange *GCC *CCC*TCC
        result_df = pd.DataFrame()
        result_df["ca_raw"] = (self.irange - range_) * 100 / self.irange
        # TODO: the * 100 is not mentioned in the excel doc
        result_df["ca_gcc"] = result_df["ca_raw"] * self.gcc
        result_df["ca_ccc"] = result_df["ca_gcc"] * self.ccc
        result_df["ca_tcc"] = result_df["ca_ccc"] * self.tcc
        result_df["amplitude"] = result_df["ca_tcc"]

        return result_df

    def plot(self):
        ...
        # TODO
        # plt.xlabel("time")
        #
        # columns_to_plot = ("range1", "range2")
        # added_lines = plt.plot(self.range_df["time"], self.range_df.loc[:, columns_to_plot])
        # plt.legend(added_lines, columns_to_plot)
        #
        # plt.show()


if __name__ == "__main__":
    t = CPReader(pathlib.Path("./data/2024-03-25 14.53.14 Ch.1 EX-test fibrinolysis.csv"))
    # t = CPReader(pathlib.Path("./data/2024-04-01 09.02.28 Ch.4 IN-test heparin 1 u.csv"))
    # print(t.df)
    # t.plot()
