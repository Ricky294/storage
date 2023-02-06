from storage.binance.history import get_candles, TimeFormat
from datetime import datetime, timezone
import pandas as pd

pd_t = pd.to_datetime(1672531200, unit='s')

# Make sure that time zone is set to UTC!
# By default, datetime.fromtimestamp uses local timestamp, so convert it to UTC.
dt_t = datetime.fromtimestamp(1672531200, tz=timezone.utc)

candles = get_candles(
    'adausdt',
    '1d',
    'SPOT',
    'binance_candles.hdf',
    wait_sec=1,
    start_time_ms=pd_t,     # or dt_t
    time_format=TimeFormat.DATE_TIME
)
