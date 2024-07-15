import csv
import os
import pathlib
from collections import defaultdict
from typing import Iterable

import matplotlib.axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CPReader:
    def __init__(
        self,
        file: str | os.PathLike[str],
        gcc: float = 0.85,
        ccc: float = 1.0,
        tcc: float = 1.0,
    ):
        """
        # TODO add documentation
        :param file: Path to input file.
            A CSV file with 3 columns: time, cycle_id, angle_id (e.g.: "1.5,17,6943")
        """
        self.gcc = gcc
        self.ccc = ccc
        self.tcc = tcc

        with open(file, "r") as f:
            reader = csv.reader(f)
            rows = [[float(row[0]), *map(int, row[1:3])] for row in reader]

        self.raw_data: pd.DataFrame = pd.DataFrame(rows, columns=("time", "cycle_id", "angle_id"))
        self.table: pd.DataFrame = self.range1()
        self.table["range2"] = self.range2()

        # the slice is from 3 to 7 (both inclusive) since the first 3 values are always 'NaN'
        self.irange = np.mean(self.table.loc[3:7, "range2"])  # noqa

        self.table = self.table.merge(
            self.clot_amplitude(),
            left_index=True,
            right_index=True,
        )

        self.single_values = self.calculate_single_values()
        mcf_idx, mcf = self.calculate_mcf()
        self.single_values["MCF"] = mcf

        self.table["lysis"] = self.calculate_lysis(mcf_idx)
        self.table["ML"] = self.table["lysis"].cummax()

        self.single_values.update(self.calculate_lysis_values())

        pd.set_option("display.max_rows", 30)
        pd.set_option("display.max_columns", 15)
        pd.set_option("display.width", None)
        pos = 20
        print(self.table.loc[pos : pos + 20, :])
        print(self.irange)
        from pprint import pprint

        pprint(self.single_values)

    def get_cycles(self) -> Iterable[pd.DataFrame]:
        """
        Splits the  dataframe into valid cycles of 50 elements,
        25 elements apart (from cycle_id 0 to 49 and 25 to 24 respectively).
        Will ignore incomplete cycles.
        """
        # determines the starting point of the first cycle
        # (important if input file doesn't start at cycle_id == 0)
        # TODO: maybe allow cycles to start at id 25?
        cycle_starting_points: pd.DataFrame = self.raw_data.index[self.raw_data["cycle_id"] == 0]
        start_idx: np.int64 = cycle_starting_points[0]

        for i in range(start_idx, len(self.raw_data), 25):
            cycle = self.raw_data.iloc[i : i + 50]
            if len(cycle) == 50:
                yield cycle

    def range1(self) -> pd.DataFrame:
        """
        Calculates the range and range1 values from raw data.
        :return:
        """
        range1_dict = defaultdict(list)
        for cycle in self.get_cycles():
            cycle_id = cycle["cycle_id"]
            median_values1: pd.DataFrame = cycle.loc[(7 <= cycle_id) & (cycle_id <= 19)]
            median_values2: pd.DataFrame = cycle.loc[(32 <= cycle_id) & (cycle_id <= 44)]
            # noinspection PyTypeChecker
            values: pd.Series = pd.concat(
                [
                    median_values1["angle_id"],
                    median_values2["angle_id"],
                ]
            )
            max_value = values.max()
            min_value = values.min()
            range_ = np.abs(max_value - min_value)

            median1 = np.median(median_values1["angle_id"])
            median2 = np.median(median_values2["angle_id"])
            range1 = np.abs(median1 - median2)

            range1_dict["time"].append(cycle.iloc[0]["time"])
            range1_dict["range"].append(range_)
            range1_dict["range1"].append(range1)

        range1_df = pd.DataFrame(range1_dict)
        return range1_df

    def range2(self) -> pd.Series:
        """
        Range2:
        Smoothes a series by taking the median of 7 consecutive values (3 preceeding, 3 following).
        First and last 3 values will be 'NaN'.
        :return:
        """
        series = self.table["range1"]
        new_series = pd.Series()
        for idx, item in enumerate(series):
            # the first and last 3 entries don't get a value
            if idx <= 2 or idx >= (len(series) - 3):
                val = np.nan
            else:
                val = np.median(series[idx - 3 : idx + 4])

            new_series[idx] = val
        return new_series

    def clot_amplitude(self) -> pd.DataFrame:
        """
        The clot amplitude is calculated as follows (CA(t) represents the CA at timepoint (t)):
        CA(t) = (iRange-Range(t))  / iRange * GCC * CCC * TCC

        :return: A dataframe with different clot amplitude values calculated.
        """
        # CA(t) = (iRange-Range(t))  / iRange *GCC *CCC*TCC
        result_df = pd.DataFrame()
        result_df["ca_raw"] = (self.irange - self.table["range"]) * 100 / self.irange
        result_df["ca_gcc"] = result_df["ca_raw"] * self.gcc
        result_df["ca_ccc"] = result_df["ca_gcc"] * self.ccc
        result_df["ca_tcc"] = result_df["ca_ccc"] * self.tcc
        result_df["amplitude"] = result_df["ca_tcc"]

        return result_df

    def calculate_single_values(self) -> dict:
        single_values = {
            "initial_CT": self.table[self.table["amplitude"] > 2].iloc[0]["time"],
            "definite_CT": (ct := self.table[self.table["amplitude"] > 4].iloc[0]["time"]),
            "A5": self.table[self.table["time"] == ct + 60 * 5]["amplitude"].iloc[0],
            "A10": self.table[self.table["time"] == ct + 60 * 10]["amplitude"].iloc[0],
            "A20": self.table[self.table["time"] == ct + 60 * 20]["amplitude"].iloc[0],
            "CFT": self.table[self.table["amplitude"] > 20].iloc[0]["time"] - ct,
        }
        return single_values

    def calculate_lysis_values(self) -> dict:
        ct = self.single_values["definite_CT"]
        lysis_values = {
            "ML": min(self.table["lysis"].max(), 100),
            "LOT": self.table[self.table["lysis"] > 15].iloc[0]["time"] - ct,
            "LT": self.table[self.table["lysis"] > 50].iloc[0]["time"] - ct,
        }

        try:
            lysis_values["ML30"] = self.table[self.table["time"] == ct + 60 * 30]["ML"].item()
        except ValueError:
            lysis_values["ML30"] = np.nan
        try:
            lysis_values["ML45"] = self.table[self.table["time"] == ct + 60 * 45]["ML"].item()
        except ValueError:
            lysis_values["ML45"] = np.nan
        try:
            lysis_values["ML60"] = self.table[self.table["time"] == ct + 60 * 60]["ML"].item()
        except ValueError:
            lysis_values["ML60"] = np.nan

        return lysis_values

    def calculate_mcf(self) -> tuple[int, float]:
        amplitudes = self.table["amplitude"]
        highest_ca: float = amplitudes[0]
        amplitude: float
        for idx, amplitude in enumerate(amplitudes):
            if (
                (
                    # The MCF is finalized either (whichever scenario is achieved first)
                    # •	When 3 consecutive values are lower than the highest CA recorded prior to these 3 values
                    sum(amplitudes.iloc[idx : idx + 4] < highest_ca) == 3
                    # •	When a CA of at least 20 mm is reached and the current CA is less than 0.5 mm larger
                    #   than the CA of the value 10 lines before
                    or (amplitude >= 20 and 0 < (amplitude - amplitudes.iloc[idx - 10]) < 0.5)
                )
                # mcf is only shown from the point of the definite CT (defined as the point where "amplitude > 4")
                # It doesn't really make sense to be able to finalize it before that point
                and amplitude > 4
            ):
                # MCF is finalized
                break

            if amplitude > highest_ca:
                highest_ca = amplitude

        # noinspection PyUnboundLocalVariable
        return idx, highest_ca

    def calculate_lysis(self, mcf_idx: int) -> pd.Series:
        # since the MCF is currently not working, we use the max values instead (testing only!)
        # TODO: remove overwriting of mcf index
        mcf_idx = self.table["amplitude"].idxmax()

        mcf = self.table.iloc[mcf_idx]

        lysis = pd.Series(np.zeros(len(self.table)))
        lysis.iloc[:mcf_idx] = np.nan  # no lysis value before mcf is finalized

        max_clot_firmness: float = mcf["amplitude"].item()
        table_subset = self.table.iloc[mcf_idx:]
        lysis.iloc[mcf_idx:] = (1 - (table_subset["amplitude"] / max_clot_firmness)) * 100
        return lysis


if __name__ == "__main__":
    t = CPReader(pathlib.Path("./data/2024-03-25 14.53.14 Ch.1 EX-test fibrinolysis.csv"))
    # t = CPReader(pathlib.Path("./data/2024-04-01 09.02.28 Ch.4 IN-test heparin 1 u.csv"))
    # print(t.df)
    # t.plot()

    """
    old plot method:
    
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
    """
