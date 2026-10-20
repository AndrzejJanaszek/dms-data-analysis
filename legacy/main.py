import sqlite3
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt

import sys
from qtpy.QtWidgets import QApplication
from plotpy.builder import make
from plotpy.plot import PlotDialog, PlotOptions

import datetime
from qwt import QwtScaleDraw, QwtText

def load_data_sqlite():
    # Load data from .db
    conn = sqlite3.connect('data/measurements.db')
    df = pd.read_sql_query("SELECT * FROM measurements WHERE value > 1000", conn)
    conn.close()

    print(df.head())
    print(len(df))

    # timestump shift
    target = datetime.datetime(2026, 7, 22, 12, 27, tzinfo=datetime.timezone.utc)
    target_ts = target.timestamp()

    offset = target_ts - df['timestamp'].max()
    df['timestamp'] = df['timestamp'] + offset

    return df

def plot_and_save(df, day_start, day_end):
    start = datetime.datetime(2026, 7, day_start, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    end   = datetime.datetime(2026, 7, day_end, 0, 0, tzinfo=datetime.timezone.utc).timestamp()

    df_range = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    plt.figure(figsize=(13, 7))
    plt.plot(pd.to_datetime(df_range['timestamp'], unit='s'), df_range['value'])
    plt.xlabel('Czas')
    plt.ylabel('Value')
    plt.title(f'Pomiary {day_start:02d}-{day_end:02d}.07.2026')
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f'plots/wykres_{day_start:02d}_{day_end:02d}_07_2026.png'
    plt.savefig(filename)
    plt.close()

    print(f'Zapisano: {filename}')

def show_gui(df):
    app = QApplication(sys.argv)

    x = df['timestamp'].to_numpy()
    y = df['value'].to_numpy()

    win = PlotDialog(
        title="Analiza pomiarów",
        options=PlotOptions(type="curve"),
        toolbar=True,
    )
    plot = win.get_plot()
    curve = make.curve(x, y, color="w")
    plot.add_item(curve)

    plot.setAxisScaleDraw(plot.xBottom, DateTimeScaleDraw())
    plot.set_axis_title("bottom", "Czas")
    plot.set_axis_title("left", "Value")

    win.exec_()

def show_comparison(df, df_smoothed, df_derivative = None, df_sec_derivative = None, intervals = None):
    app = QApplication(sys.argv)

    win = PlotDialog(
        title="Porównanie: dane / wygładzone",
        options=PlotOptions(type="curve"),
        toolbar=True,
    )
    plot = win.get_plot()

    curve_raw = make.curve(
        df['timestamp'].to_numpy(), df['value'].to_numpy(),
        color="blue", title="Surowe dane"
    )
    plot.add_item(curve_raw)

    curve_smoothed = make.curve(
        df_smoothed['timestamp'].to_numpy(), df_smoothed['value'].to_numpy(),
        color="white", title="Wygładzone"
    )
    plot.add_item(curve_smoothed)

    if df_derivative is not None: 
        curve_derivative = make.curve(
            df_derivative['timestamp'].to_numpy(), df_derivative['value'].to_numpy(),
            color="red", title="Pochodna",
            yaxis="right"
        )
        plot.add_item(curve_derivative)

    if df_sec_derivative is not None: 
            curve_sec_derivative = make.curve(
                df_sec_derivative['timestamp'].to_numpy(), df_sec_derivative['value'].to_numpy(),
                color="green", title="druga Pochodna",
                yaxis="right"
            )
            plot.add_item(curve_sec_derivative)


    if intervals is not None:
        for xmin, xmax in intervals:
            range_item = make.range(xmin, xmax)
            plot.add_item(range_item)

    plot.setAxisScaleDraw(plot.xBottom, DateTimeScaleDraw())
    plot.set_axis_title("bottom", "Czas")
    plot.set_axis_title("left", "Value")
    plot.set_axis_title("right", "Pochodna")

    # włączenie prawej osi (domyślnie bywa ukryta)
    plot.enableAxis(plot.yRight, True)

    # legenda, żeby rozróżnić krzywe
    legend = make.legend("TR")
    plot.add_item(legend)

    win.exec_()

class DateTimeScaleDraw(QwtScaleDraw):
    def label(self, value):
        try:
            dt = datetime.datetime.fromtimestamp(value)
            text = dt.strftime("%H:%M:%S\n%d-%m-%Y")
        except (ValueError, OSError, OverflowError):
            text = ""
        return QwtText(text)

# split df if two series overlap
def split_by_timestamp_continuity(df):
    last_timestamp = 0
    split_idx = -1
    for idx, row in df.iterrows():
        if row['timestamp'] > last_timestamp:
            last_timestamp = row['timestamp']
        else:
            split_idx = idx
            break

    if split_idx > 0 :
        df_1 = df.iloc[:split_idx].reset_index(drop=True)
        df_2 = df.iloc[split_idx:].reset_index(drop=True)
        return df_1, df_2
    else :
        return df

def load_data_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_by_day(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date

    return [
        group.drop(columns=['date']).reset_index(drop=True)
        for _, group in df.groupby('date')
    ]

def df_smooth(df, window_width):
    df_smoothed = df.copy()
    df_smoothed['value'] = df['value'].rolling(window=window_width, center=True, min_periods=1).mean()
    return df_smoothed

def df_derivative_gradient(df):
    df_deriv = df.copy()
    df_deriv['value'] = np.gradient(df['value'].to_numpy(), df['timestamp'].to_numpy())
    return df_deriv

def find_intervals_below_threshold(df_derivative, threshold):
    mask = (df_derivative['value'] < threshold).to_numpy()
    timestamps = df_derivative['timestamp'].to_numpy()

    # wykryj miejsca, gdzie maska się zmienia (False->True, True->False)
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1   # początki przedziałów True
    ends = np.where(diff == -1)[0] + 1    # końce przedziałów True

    # obsłuż przypadek, gdy szereg zaczyna/kończy się w stanie True
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    intervals = [
        (timestamps[s], timestamps[e - 1])
        for s, e in zip(starts, ends)
    ]
    return intervals

def get_values_from_intervals(df, intervals):
    results = []
    for start, end in intervals:
        value_start = df.loc[df["timestamp"] == start, 'value'].iloc[0]
        value_end = df.loc[df["timestamp"] == end, 'value'].iloc[0]
        results.append({
            "s_value": value_start,
            "e_value": value_start,
            "s_timestamp": start,
            "e_timestamp": end,
            "dxdy": (value_end - value_start) / (end - start),
            "dx": value_end - value_start,
            "dy": end - start,
        })

    return results
        # results.append((start, value_start, end, value_end, dxdy))
def main():
    # df = load_data_sqlite()


    # ###### LOAD AND SPLIT BY DAY ######
    # df = load_data_csv("data/df_2.csv")=
    # df_list = split_by_day(df)

    # for e in df_list:
    #     datetime: pd.Timestamp = e["datetime"][0]
    #     date = datetime.date().strftime("%Y-%m-%d")
    #     filename = "df_data_from_" + date + ".csv"
    #     e.to_csv("data/days/" + filename, index=False)


    # df_1, df_2 = split_by_timestamp_continuity(df)

    df = load_data_csv("data/days/df_data_from_2026-07-16.csv")

    df_smoothed = df_smooth(df, 10)
    # df_smoothed_100 = df_smooth(df, 100)
    df_deriv = df_derivative_gradient(df_smoothed)
    df_smoothed_deriv = df_smooth(df_deriv, 10)
    df_sec_deriv = df_derivative_gradient(df_smoothed_deriv)

    intervals = find_intervals_below_threshold(df_deriv, -5)

    intervals_cuted = []
    for s, e in intervals:
        if (e - s) > 0 and (e - s) < 6:
            intervals_cuted.append((s,e))

    value_intervals = get_values_from_intervals(df_smoothed, intervals_cuted)

    # dxdy_values = [r["dy"] for r in value_intervals]
    # plt.hist(dxdy_values, bins=20, color="steelblue", edgecolor="black")
    # plt.xlabel("dxdy")
    # plt.ylabel("Liczba wystąpień")
    # plt.title("Histogram dxdy")
    # plt.grid(alpha=0.3)
    # plt.show()

    # show_gui(df)
    show_comparison(df=df,
                    df_smoothed=df_smoothed,
                    df_derivative=df_deriv,
                    intervals=intervals_cuted)
    # show_comparison(df_smoothed_100, df_smoothed, df_deriv, intervals)
    # plot_and_save(df, 11, 12)
    # plot_and_save(df, 12, 13)
    # plot_and_save(df, 19, 20)
    pass

if(__name__ == "__main__"):
    main()