import sqlite3
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

def main():
    # df = load_data_sqlite()


    # ###### LOAD AND SPLIT BY DAY ######
    # df = load_data_csv("data/df_2.csv")
    # df_list = split_by_day(df)

    # for e in df_list:
    #     datetime: pd.Timestamp = e["datetime"][0]
    #     date = datetime.date().strftime("%Y-%m-%d")
    #     filename = "df_data_from_" + date + ".csv"
    #     e.to_csv("data/days/" + filename, index=False)


    # df_1, df_2 = split_by_timestamp_continuity(df)

    df = load_data_csv("data/days/df_data_from_2026-07-19.csv")

    show_gui(df)
    # plot_and_save(df, 11, 12)
    # plot_and_save(df, 12, 13)
    # plot_and_save(df, 19, 20)
    pass

if(__name__ == "__main__"):
    main()