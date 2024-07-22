import csv
import logging
import os
import pathlib
from collections import defaultdict
from typing import Iterable, Callable

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        self.table: pd.DataFrame = self._range1()
        self.table["range2"] = self._range2()

        # the slice is from 3 to 7 (both inclusive) since the first 3 values are always 'NaN'
        self.irange = np.mean(self.table.loc[3:7, "range2"])  # noqa

        self.table = self.table.merge(
            self._clot_amplitude(),
            left_index=True,
            right_index=True,
        )

        self.single_values = self._calculate_single_values()

        mcf_idx, mcf = self._calculate_mcf()
        self.single_values["MCF"] = mcf
        self.table["lysis"] = self._calculate_lysis(mcf_idx)
        self.table["ML"] = self.table["lysis"].cummax()

        self.single_values.update(self._calculate_lysis_values())

        # TODO: remove those, just for testing
        pd.set_option("display.max_rows", 30)
        pd.set_option("display.max_columns", 15)
        pd.set_option("display.width", None)
        pos = 20
        print(self.table.loc[pos : pos + 20, :])
        from pprint import pprint

        pprint(self.single_values)

    def _get_cycles(self) -> Iterable[pd.DataFrame]:
        """
        Splits the  dataframe into valid cycles of 50 elements,
        25 elements apart (from cycle_id 0 to 49 and 25 to 24 respectively).
                                         1 to 0  and 26 to 25
                                         yeah this doesn't make much sense but that's how it is defined
                                         # TODO: cycle_ID
        Will ignore incomplete cycles.
        """
        # determines the starting point of the first cycle
        # (important if input file doesn't start at cycle_id == 0)
        # TODO: maybe allow first cycle to start at id 25?
        cycle_starting_points: pd.DataFrame = self.raw_data.index[self.raw_data["cycle_id"] == 0]
        start_idx: np.int64 = cycle_starting_points[0]

        for i in range(start_idx, len(self.raw_data), 25):
            idx = i + 1
            cycle = self.raw_data.iloc[idx : idx + 50]

            if len(cycle) == 50:
                yield cycle

    def _range1(self) -> pd.DataFrame:
        """
        Calculates the range and range1 values from raw data.
        :return:
        """
        range1_dict = defaultdict(list)
        for cycle in self._get_cycles():
            cycle_id = cycle["cycle_id"]
            median_values1: pd.DataFrame = cycle.loc[(7 <= cycle_id) & (cycle_id <= 19)]
            median_values2: pd.DataFrame = cycle.loc[(32 <= cycle_id) & (cycle_id <= 44)]
            # noinspection PyTypeChecker
            values: pd.Series = pd.concat([
                median_values1["angle_id"],
                median_values2["angle_id"],
            ])
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

    def _range2(self) -> pd.Series:
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
                val = np.median(series.iloc[idx - 3 : idx + 4])

            new_series[idx] = val
        return new_series

    def _clot_amplitude(self) -> pd.DataFrame:
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

    @staticmethod
    def __catch_exceptions_for_single_value_calcs(name: str, value_func: Callable):
        """
        Catches exceptions and logs them to the global logger.
        Will return NaN if an exception occurs.
        :param name: Name of the value to be calculated. Only used for logging.
        :param value_func: A function that takes no arguments.
        """
        # noinspection PyBroadException
        try:
            return value_func()
        except Exception:
            logger.warning(f"Could not calculate '{name}', setting no NaN instead.")
            return np.nan

    def _calculate_single_values(self) -> dict:
        single_values = dict()

        single_values["initial_CT"] = self.__catch_exceptions_for_single_value_calcs(
            "initial_CT", lambda: self.table[self.table["amplitude"] > 2].iloc[0]["time"]
        )

        single_values["definite_CT"] = (
            ct := self.__catch_exceptions_for_single_value_calcs(
                "definite_CT", lambda: self.table[self.table["amplitude"] > 4].iloc[0]["time"]
            )
        )

        single_values["A5"] = self.__catch_exceptions_for_single_value_calcs(
            "A5", lambda: self.table[self.table["time"] == ct + 60 * 5]["amplitude"].iloc[0]
        )
        single_values["A10"] = self.__catch_exceptions_for_single_value_calcs(
            "A10", lambda: self.table[self.table["time"] == ct + 60 * 10]["amplitude"].iloc[0]
        )
        single_values["A20"] = self.__catch_exceptions_for_single_value_calcs(
            "A20", lambda: self.table[self.table["time"] == ct + 60 * 20]["amplitude"].iloc[0]
        )
        single_values["CFT"] = self.__catch_exceptions_for_single_value_calcs(
            "CFT", lambda: self.table[self.table["amplitude"] > 20].iloc[0]["time"] - ct
        )
        return single_values

    def _calculate_lysis_values(self) -> dict:
        ct = self.single_values["definite_CT"]
        lysis_values = dict()

        lysis_values["ML"] = min(self.table["lysis"].max(), 100)
        lysis_values["LOT"] = self.__catch_exceptions_for_single_value_calcs(
            "LOT", lambda: self.table[self.table["lysis"] > 15].iloc[0]["time"] - ct
        )
        lysis_values["LT"] = self.__catch_exceptions_for_single_value_calcs(
            "LT", lambda: self.table[self.table["lysis"] > 50].iloc[0]["time"] - ct
        )
        lysis_values["ML30"] = self.__catch_exceptions_for_single_value_calcs(
            "ML30", lambda: self.table[self.table["time"] == ct + 60 * 30]["ML"].item()
        )
        lysis_values["ML45"] = self.__catch_exceptions_for_single_value_calcs(
            "ML45", lambda: self.table[self.table["time"] == ct + 60 * 45]["ML"].item()
        )
        lysis_values["ML60"] = self.__catch_exceptions_for_single_value_calcs(
            "ML60", lambda: self.table[self.table["time"] == ct + 60 * 60]["ML"].item()
        )

        return lysis_values

    def _calculate_mcf(self) -> tuple[int, float]:
        amplitudes: pd.Series = self.table["amplitude"]
        highest_ca: float = 0.0
        amplitude: float
        for idx, amplitude in enumerate(amplitudes):
            # FIXME: this breaks too early.
            if (
                (
                    # The MCF is finalized either (whichever scenario is achieved first)
                    # •	When 3 consecutive values are lower than the highest CA recorded prior to these 3 values
                    sum(amplitudes.iloc[idx : idx + 3] < highest_ca) == 3
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

    def _calculate_lysis(self, mcf_idx: int) -> pd.Series:
        # since the MCF is currently not working, we use the max values instead (testing only!)
        # TODO: remove overwriting of mcf index
        mcf_idx = self.table["amplitude"].idxmax()  # <-- remove this

        mcf = self.table.iloc[mcf_idx]

        lysis = pd.Series(np.zeros(len(self.table)))
        lysis.iloc[:mcf_idx] = np.nan  # no lysis value before mcf is finalized

        max_clot_firmness: float = mcf["amplitude"]
        table_subset = self.table.iloc[mcf_idx:]
        lysis_subset = (1 - (table_subset["amplitude"] / max_clot_firmness)) * 100
        lysis.iloc[mcf_idx:] = lysis_subset.astype(np.float64)
        return lysis

    def plot(self):
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        fig, ax = plt.subplots()

        amplitudes = self.table["amplitude"].copy()  # copy() to avoid writing data back to self.table
        amplitudes[amplitudes < 2] = 2  # for plotting only round up all amplitude to at least 2

        time = self.table["time"]

        bar_width = time[1] - time[0]  # make the bars fill up the space between the indices.
        max_value = amplitudes.max()
        ax.set_ylim(max_value / -1.6, max_value / 1.6)

        ax.set_axisbelow(True)
        ax.set_xticks(np.arange(0, round(max(time), -1), 10))
        ax.grid(linestyle="--", alpha=0.8)

        # plot the green part before the initial CT
        ct = self.single_values["initial_CT"]
        ct_idx = time[time == ct].index.item()
        amplitudes_plt = amplitudes.iloc[:ct_idx]
        time_plt = time.iloc[:ct_idx] / 60
        ax.bar(time_plt, height=amplitudes_plt, width=bar_width, bottom=amplitudes_plt / -2, color="#5D8C41")

        # plot the purple part until CA reaches 10mm
        try:
            cft_idx = amplitudes[amplitudes > 20].index[0]
        except IndexError:
            # if it never reaches 20mm it will plot until the end
            cft_idx = len(amplitudes)
        amplitudes_plt = amplitudes.iloc[ct_idx:cft_idx]
        time_plt = time.iloc[ct_idx:cft_idx] / 60
        ax.bar(time_plt, height=amplitudes_plt, width=bar_width, bottom=amplitudes_plt / -2, color="#EA37F7")

        # plot the rest
        amplitudes_plt = amplitudes.iloc[cft_idx:]
        time_plt = time.iloc[cft_idx:] / 60
        ax.bar(time_plt, height=amplitudes_plt, width=bar_width, bottom=amplitudes_plt / -2, color="#394B90")

        plt.show()


if __name__ == "__main__":
    logging.basicConfig()
    # t = CPReader(pathlib.Path("./data/2024-03-25 14.53.14 Ch.1 EX-test fibrinolysis.csv"))
    t = CPReader(pathlib.Path("./data/2024-04-01 09.02.28 Ch.4 IN-test heparin 1 u.csv"))
    t.plot()
