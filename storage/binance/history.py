from __future__ import annotations

import enum
import logging
import time
from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


SPOT_BASE_URL: Final = 'https://api.binance.com/api/v3/klines'
FUTURES_BASE_URL: Final = 'https://fapi.binance.com/fapi/v1/klines'


class HTTPException(Exception):

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def get_base_url(
        market: str,
):
    if market.upper() == 'SPOT':
        return SPOT_BASE_URL
    elif 'FUTURE' in market.upper():
        return FUTURES_BASE_URL
    else:
        raise ValueError(f'Invalid market type: {market}')


def get_first_candle_timestamp_ms(
        symbol: str,
        interval: str,
        market: str,
) -> int:
    """Returns the first available candle data timestamp
    on binance_util `symbol`, `interval`, `market` in milliseconds."""
    return int(requests.get(
        get_base_url(market), params=dict(symbol=symbol.upper(), interval=interval, limit=1, startTime=0)
    ).json()[-1][0])


def _get_candles(
        symbol: str,
        interval: str,
        market: str,
        limit=1000,
        start_time_ms: float | datetime = None,
        end_time_ms: float | datetime = None
) -> pd.DataFrame:
    if isinstance(start_time_ms, datetime):
        start_time_ms = start_time_ms.timestamp() * 1000
    if isinstance(end_time_ms, datetime):
        end_time_ms = end_time_ms.timestamp() * 1000

    if isinstance(start_time_ms, float):
        start_time_ms = int(start_time_ms)
    if isinstance(end_time_ms, float):
        end_time_ms = int(end_time_ms)

    response = requests.get(
        get_base_url(market),
        params=dict(
            symbol=symbol.upper(),
            interval=interval,
            limit=limit,
            startTime=start_time_ms,
            endTime=end_time_ms,
        )
    )

    if response.status_code >= 300:
        raise HTTPException(**response.json())

    df = pd.DataFrame(
        data=response.json(),
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore"
        ], dtype="float64"
    )

    return df


def get_last_in_hdf(path: str, key: str) -> pd.DataFrame:
    """Returns last row from HDF store as a pandas DataFrame."""
    with pd.HDFStore(path, mode='r') as s:
        n_rows = s.get_storer(key).nrows
        return s.select(key, start=n_rows - 1, stop=n_rows)


class TimeFormat(enum.Enum):
    DATE_TIME = 'DATE_TIME'
    SECONDS = 'SECONDS'
    MILLISECONDS = 'MILLISECONDS'


def _build_where(start_time_ms: float | None, end_time_ms: float | None):
    """
    >>> _build_where(None, None)
    ''

    >>> _build_where(0, None)
    ''

    >>> _build_where(1, None)
    'open_time >= 1'

    >>> _build_where(1, 2)
    'open_time >= 1 and close_time <= 2'

    >>> _build_where(None, 2)
    'close_time <= 2'
    """
    where = ''
    if start_time_ms:
        where += f'open_time >= {start_time_ms}'

    if end_time_ms:
        if where:
            where += ' and '

        where += f'close_time <= {end_time_ms}'

    return where


SEC_MAP = {
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
    'M': 2629800,
    'Y': 31557600,
}


def split_interval(interval: str):
    """
    Decomposes interval to time value and time unit.
    :return: tuple - time value (int), time unit (str)
    :examples:
    >>> split_interval('15m')
    (15, 'm')
    >>> split_interval('1h')
    (1, 'h')
    >>> split_interval('200d')
    (200, 'd')
    """

    timeframe = interval[len(interval) - 1]
    value = interval[0: len(interval) - 1]

    return int(value), timeframe


def interval_to_seconds(interval: str) -> int:
    """
    Converts interval (str) to seconds (int)
    :examples:
    >>> interval_to_seconds('1m')
    60
    >>> interval_to_seconds('15m')
    900
    >>> interval_to_seconds('1h')
    3600
    >>> interval_to_seconds('2h')
    7200
    """

    value, timeframe = split_interval(str(interval))
    return value * SEC_MAP[timeframe]


def get_n_candles(
        symbol: str,
        interval: str,
        market: str,
        start_time_ms: float,
        end_time_ms: float = None,
):
    """Calculates the number of storage between start_time_ms and end_time_ms."""

    if (start_time_ms and end_time_ms) and (start_time_ms >= end_time_ms):
        raise ValueError('End time must be greater than start time.')

    if start_time_ms > datetime.now().timestamp() * 1000:
        raise ValueError('Start time must not be greater than ')

    first_time_ms = get_first_candle_timestamp_ms(symbol, interval, market)
    if start_time_ms > first_time_ms:
        first_time_ms = start_time_ms

    last_time_ms = int(datetime.now().timestamp() * 1000)
    if end_time_ms and end_time_ms < last_time_ms:
        last_time_ms = end_time_ms

    total_items = (last_time_ms - first_time_ms) / interval_to_seconds(interval) / 1000

    return total_items


def validate_candles(df: pd.DataFrame):
    def log(validation_flag: bool, pass_msg: str, fail_msg):
        if validation_flag:
            logging.info(f'Validation PASS: {pass_msg}')
        else:
            logging.warning(f'Validation FAILED: {fail_msg}')

    def validate_spacing(series: pd.Series):
        diffs = np.diff(series.to_numpy())
        return np.all(diffs == diffs[0])

    open_time_spacing = validate_spacing(df['open_time'])
    close_time_spacing = validate_spacing(df['close_time'])

    log(open_time_spacing, 'open_time series is equally spaced!', 'open_time series is not equally spaced!')
    log(close_time_spacing, 'close_time series is equally spaced!', 'close_time series is not equally spaced!')

    return all([open_time_spacing, close_time_spacing])


def get_candles(
        symbol: str,
        interval: str,
        market: str,
        hdf_path: str,
        hdf_key: str = None,
        limit=1000,
        wait_sec=1.0,
        start_time_ms: float | datetime = None,
        end_time_ms: float | datetime = None,
        time_format=TimeFormat.MILLISECONDS,
        validation=True,
) -> pd.DataFrame:
    """Recursively fetches storage from Binance from
    start_time to end_time and stores storage in a HDF5 store.

    Returned DataFrame column schema (Binance):
      * "open_time"
      * "open"
      * "high"
      * "low"
      * "close"
      * "volume"
      * "close_time"
      * "quote_asset_volume"
      * "number_of_trades"
      * "taker_buy_base_asset_volume"
      * "taker_buy_quote_asset_volume"
      * "ignore"

    Note: open_time and close_time is returned in seconds instead of milliseconds if date_time is False.

    :param hdf_path: Storage path to append downloaded storage (File is created if not exists).
    :param hdf_key: Identifier for the group in the store. Default value: f'{symbol.lower()}_{interval}_{market.lower()}'
    :param limit: Maximum number of downloaded storage per iteration (maximum allowed value is 1000).
    :param wait_sec: Number of seconds to wait between each iteration.
    :param start_time_ms: UTC timestamp in milliseconds or datetime object.
    :param end_time_ms: UTC timestamp in milliseconds or datetime object.
    :param time_format: Applies formatting to open_time and close_time columns (default: MILLISECONDS).
    :param validation: Does some validation on the dataset. Log messages to console.
    :return: Returns a pandas DataFrame with all storage from start_time to end_time.
    """

    def get_latest_close(candles: pd.DataFrame):
        return candles['close_time'].iat[-1]

    def convert_time_to_ms(val: float | datetime | None):
        return val.timestamp() * 1000 if isinstance(val, datetime) else val

    start_time_ms = convert_time_to_ms(start_time_ms)
    end_time_ms = convert_time_to_ms(end_time_ms)

    next_time_ms = start_time_ms if start_time_ms else .0

    if hdf_key is None:
        hdf_key = f'{symbol.lower()}_{interval}_{market.lower()}'

    try:
        last_df = get_last_in_hdf(path=hdf_path, key=hdf_key)
        next_time_ms = get_latest_close(last_df)
    except (KeyError, FileNotFoundError):
        pass

    n_candles = get_n_candles(
        symbol=symbol,
        interval=interval,
        market=market,
        start_time_ms=next_time_ms,
        end_time_ms=end_time_ms
    )

    with tqdm(total=int(n_candles)) as pbar:
        while True:
            next_df = _get_candles(
                symbol=symbol, interval=interval, market=market,
                limit=limit, start_time_ms=next_time_ms, end_time_ms=end_time_ms
            )

            # Remove last candle because it is still an open candle
            if not next_df.empty and len(next_df) < limit:
                next_df = next_df[:-1]

            next_df.to_hdf(
                hdf_path,
                key=hdf_key,
                mode='a',
                append=True,
                format='table',
                data_columns=['open_time', 'close_time'],
                index=False,
            )
            pbar.update(len(next_df))

            if len(next_df) < limit:
                pbar.close()
                break

            time.sleep(wait_sec)
            next_time_ms = get_latest_close(next_df)

    df: pd.DataFrame = pd.read_hdf(
        path_or_buf=hdf_path,
        key=hdf_key,
        where=_build_where(start_time_ms, end_time_ms),
    )

    if validation:
        if validate_candles(df):
            logging.info('All data validation passed!')
        else:
            logging.warning('Data validation failed!')

    if time_format is TimeFormat.DATE_TIME:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    elif time_format is TimeFormat.SECONDS:
        df['open_time'] /= 1000
        df['close_time'] /= 1000

    return df
