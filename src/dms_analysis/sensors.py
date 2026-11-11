from dataclasses import dataclass

import pandas as pd

ERROR_CODES = (85.0, -127.0)
ADDRESS_LENGTH = 16


@dataclass(frozen=True)
class Sensor:
    number: int
    sides: str


SENSORS = {
    "28A8F126AB240B1C": Sensor(1, "południe, wschód"),
    "28989298AA240B81": Sensor(2, "zachód, południe"),
    "287C36E9AB240BD4": Sensor(3, "zachód"),
    "2833D8EDAB240BF3": Sensor(4, "wschód"),
}


def valid_readings(temperatures: pd.DataFrame) -> pd.DataFrame:
    full_address = temperatures["sensor_address"].str.len() == ADDRESS_LENGTH
    error_code = temperatures["value"].isin(ERROR_CODES)
    return temperatures[full_address & ~error_code]


def with_sensor_numbers(temperatures: pd.DataFrame) -> pd.DataFrame:
    numbers = {address: sensor.number for address, sensor in SENSORS.items()}
    return temperatures.assign(sensor=temperatures["sensor_address"].map(numbers).astype("Int64"))


def sensors_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sensor_address": address, "sensor": sensor.number, "sides": sensor.sides}
            for address, sensor in SENSORS.items()
        ]
    )
