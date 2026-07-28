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

def load_data():
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
    curve = make.curve(x, y, color="b")
    plot.add_item(curve)

    plot.setAxisScaleDraw(plot.xBottom, DateTimeScaleDraw())
    plot.set_axis_title("bottom", "Czas")
    plot.set_axis_title("left", "Value")

    win.exec_()

class DateTimeScaleDraw(QwtScaleDraw):
    def label(self, value):
        dt = datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
        return QwtText(dt.strftime('%H:%M:%S\n%d-%m-%Y'))

def main():
    df = load_data()
    show_gui(df)
    # plot_and_save(df, 11, 12)
    # plot_and_save(df, 12, 13)
    # plot_and_save(df, 19, 20)
    pass

if(__name__ == "__main__"):
    main()