"""
Abstract API -- compatible with the original talib.abstract module.

Usage:
    from talib.abstract import SMA, RSI, MACD

    # Call with a dict of arrays (pandas DataFrame also works)
    output = SMA({'close': close_array}, timeperiod=25)

    # Or use the Function class directly
    func = Function('SMA')
    func.parameters = {'timeperiod': 25}
    output = func(input_arrays)
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import talib._talib as _talib
from talib._talib import get_functions, get_function_groups  # noqa: F401

# ---------------------------------------------------------------------------
# Internal metadata registry
# ---------------------------------------------------------------------------
# Each entry maps FUNC_NAME -> dict with:
#   group        : str
#   display_name : str
#   input_names  : OrderedDict  {key: list_of_price_series}
#   parameters   : OrderedDict  {param: default}
#   output_names : list[str]
#   output_flags : dict[str, list[str]]   (TA-Lib compat flags)
#
# The "input_names" use the TA-Lib convention:
#   'price'   -> ['close']   (single real-valued series)
#   'prices'  -> ['open', 'high', 'low', 'close']
#   'price0'/'price1' -> two independent series
#   Specific keys like 'high', 'low', etc. for explicit inputs
# ---------------------------------------------------------------------------

# Flags matching TA-Lib output flag constants
_LINE = ['Line']
_DASHED_LINE = ['Dashed Line']
_DOT = ['Dot']
_HISTOGRAM = ['Histogram']
_PATTERN_BOOL = ['Pattern Bool']
_PATTERN_BULL_BEAR = ['Pattern Bull/Bear']
_PATTERN_STRENGTH = ['Pattern Strength']
_POSITIVE = ['Positive']
_NEGATIVE = ['Negative']
_ZERO = ['Zero']
_UPPER_LIMIT = ['Upper Limit']
_LOWER_LIMIT = ['Lower Limit']

def _make_info(name, group, display_name, input_names, parameters, output_names,
               output_flags=None):
    """Helper to build a function info dict."""
    if output_flags is None:
        output_flags = {n: _LINE for n in output_names}
    return {
        'name': name,
        'group': group,
        'display_name': display_name,
        'input_names': OrderedDict(input_names),
        'parameters': OrderedDict(parameters),
        'output_names': list(output_names),
        'output_flags': OrderedDict(output_flags),
    }


def _close():
    return [('price', ['close'])]

def _close_volume():
    return [('price', ['close']), ('price1', ['volume'])]

def _high_low():
    return [('prices', ['high', 'low'])]

def _high_low_close():
    return [('prices', ['high', 'low', 'close'])]

def _ohlc():
    return [('prices', ['open', 'high', 'low', 'close'])]

def _ohlcv():
    return [('prices', ['open', 'high', 'low', 'close']), ('price1', ['volume'])]

def _hlcv():
    return [('prices', ['high', 'low', 'close']), ('price1', ['volume'])]

def _two_series():
    return [('price0', ['close']), ('price1', ['close'])]

def _close_periods():
    return [('price', ['close']), ('periods', ['periods'])]


# Master metadata table -- one entry per function
_FUNC_INFO: Dict[str, dict] = {}

def _reg(name, group, display_name, input_names, parameters, output_names,
         output_flags=None):
    _FUNC_INFO[name] = _make_info(name, group, display_name, input_names,
                                  parameters, output_names, output_flags)

# ======================== Overlap Studies ========================

_reg('SMA', 'Overlap Studies', 'Simple Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('EMA', 'Overlap Studies', 'Exponential Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('WMA', 'Overlap Studies', 'Weighted Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('DEMA', 'Overlap Studies', 'Double Exponential Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('TEMA', 'Overlap Studies', 'Triple Exponential Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('TRIMA', 'Overlap Studies', 'Triangular Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('KAMA', 'Overlap Studies', 'Kaufman Adaptive Moving Average',
     _close(), [('timeperiod', 30)], ['real'])

_reg('T3', 'Overlap Studies', 'Triple Exponential Moving Average (T3)',
     _close(), [('timeperiod', 5), ('vfactor', 0.7)], ['real'])

_reg('MAMA', 'Overlap Studies', 'MESA Adaptive Moving Average',
     _close(), [('fastlimit', 0.5), ('slowlimit', 0.05)],
     ['mama', 'fama'])

_reg('BBANDS', 'Overlap Studies', 'Bollinger Bands',
     _close(), [('timeperiod', 5), ('nbdevup', 2.0), ('nbdevdn', 2.0), ('matype', 0)],
     ['upperband', 'middleband', 'lowerband'])

_reg('SAR', 'Overlap Studies', 'Parabolic SAR',
     _high_low(), [('acceleration', 0.02), ('maximum', 0.2)], ['real'])

_reg('SAREXT', 'Overlap Studies', 'Parabolic SAR - Extended',
     _high_low(),
     [('startvalue', 0.0), ('offsetonreverse', 0.0),
      ('accelerationinitlong', 0.02), ('accelerationlong', 0.02), ('accelerationmaxlong', 0.2),
      ('accelerationinitshort', 0.02), ('accelerationshort', 0.02), ('accelerationmaxshort', 0.2)],
     ['real'])

_reg('MIDPOINT', 'Overlap Studies', 'MidPoint over period',
     _close(), [('timeperiod', 14)], ['real'])

_reg('MIDPRICE', 'Overlap Studies', 'Midpoint Price over period',
     _high_low(), [('timeperiod', 14)], ['real'])

_reg('MAVP', 'Overlap Studies', 'Moving average with variable period',
     _close_periods(),
     [('minperiod', 2), ('maxperiod', 30), ('matype', 0)],
     ['real'])

_reg('HT_TRENDLINE', 'Overlap Studies', 'Hilbert Transform - Instantaneous Trendline',
     _close(), [], ['real'])

# ======================== Momentum Indicators ========================

_reg('RSI', 'Momentum Indicators', 'Relative Strength Index',
     _close(), [('timeperiod', 14)], ['real'])

_reg('MACD', 'Momentum Indicators', 'Moving Average Convergence/Divergence',
     _close(), [('fastperiod', 12), ('slowperiod', 26), ('signalperiod', 9)],
     ['macd', 'macdsignal', 'macdhist'],
     {'macd': _LINE, 'macdsignal': _DASHED_LINE, 'macdhist': _HISTOGRAM})

_reg('MACDEXT', 'Momentum Indicators', 'MACD with controllable MA type',
     _close(),
     [('fastperiod', 12), ('fastmatype', 1), ('slowperiod', 26), ('slowmatype', 1),
      ('signalperiod', 9), ('signalmatype', 1)],
     ['macd', 'macdsignal', 'macdhist'],
     {'macd': _LINE, 'macdsignal': _DASHED_LINE, 'macdhist': _HISTOGRAM})

_reg('MACDFIX', 'Momentum Indicators', 'Moving Average Convergence/Divergence Fix 12/26',
     _close(), [('signalperiod', 9)],
     ['macd', 'macdsignal', 'macdhist'],
     {'macd': _LINE, 'macdsignal': _DASHED_LINE, 'macdhist': _HISTOGRAM})

_reg('STOCH', 'Momentum Indicators', 'Stochastic',
     _high_low_close(),
     [('fastk_period', 5), ('slowk_period', 3), ('slowk_matype', 0),
      ('slowd_period', 3), ('slowd_matype', 0)],
     ['slowk', 'slowd'],
     {'slowk': _DASHED_LINE, 'slowd': _DASHED_LINE})

_reg('STOCHF', 'Momentum Indicators', 'Stochastic Fast',
     _high_low_close(),
     [('fastk_period', 5), ('fastd_period', 3), ('fastd_matype', 0)],
     ['fastk', 'fastd'],
     {'fastk': _DASHED_LINE, 'fastd': _DASHED_LINE})

_reg('STOCHRSI', 'Momentum Indicators', 'Stochastic Relative Strength Index',
     _close(),
     [('timeperiod', 14), ('fastk_period', 5), ('fastd_period', 3), ('fastd_matype', 0)],
     ['fastk', 'fastd'],
     {'fastk': _DASHED_LINE, 'fastd': _DASHED_LINE})

_reg('ADX', 'Momentum Indicators', 'Average Directional Movement Index',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('ADXR', 'Momentum Indicators', 'Average Directional Movement Index Rating',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('CCI', 'Momentum Indicators', 'Commodity Channel Index',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('MOM', 'Momentum Indicators', 'Momentum',
     _close(), [('timeperiod', 10)], ['real'])

_reg('ROC', 'Momentum Indicators', 'Rate of change',
     _close(), [('timeperiod', 10)], ['real'])

_reg('ROCP', 'Momentum Indicators', 'Rate of change Percentage',
     _close(), [('timeperiod', 10)], ['real'])

_reg('ROCR', 'Momentum Indicators', 'Rate of change ratio',
     _close(), [('timeperiod', 10)], ['real'])

_reg('ROCR100', 'Momentum Indicators', 'Rate of change ratio 100 scale',
     _close(), [('timeperiod', 10)], ['real'])

_reg('WILLR', 'Momentum Indicators', "Williams' %R",
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('APO', 'Momentum Indicators', 'Absolute Price Oscillator',
     _close(), [('fastperiod', 12), ('slowperiod', 26), ('matype', 0)], ['real'])

_reg('PPO', 'Momentum Indicators', 'Percentage Price Oscillator',
     _close(), [('fastperiod', 12), ('slowperiod', 26), ('matype', 0)], ['real'])

_reg('BOP', 'Momentum Indicators', 'Balance Of Power',
     _ohlc(), [], ['real'])

_reg('CMO', 'Momentum Indicators', 'Chande Momentum Oscillator',
     _close(), [('timeperiod', 14)], ['real'])

_reg('AROON', 'Momentum Indicators', 'Aroon',
     _high_low(), [('timeperiod', 14)],
     ['aroondown', 'aroonup'],
     {'aroondown': _DASHED_LINE, 'aroonup': _DASHED_LINE})

_reg('AROONOSC', 'Momentum Indicators', 'Aroon Oscillator',
     _high_low(), [('timeperiod', 14)], ['real'])

_reg('MFI', 'Momentum Indicators', 'Money Flow Index',
     _hlcv(), [('timeperiod', 14)], ['real'])

_reg('TRIX', 'Momentum Indicators', 'Triple Smooth EMA 1-day Rate-Of-Change',
     _close(), [('timeperiod', 30)], ['real'])

_reg('ULTOSC', 'Momentum Indicators', 'Ultimate Oscillator',
     _high_low_close(),
     [('timeperiod1', 7), ('timeperiod2', 14), ('timeperiod3', 28)],
     ['real'])

_reg('DX', 'Momentum Indicators', 'Directional Movement Index',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('PLUS_DI', 'Momentum Indicators', 'Plus Directional Indicator',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('MINUS_DI', 'Momentum Indicators', 'Minus Directional Indicator',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('PLUS_DM', 'Momentum Indicators', 'Plus Directional Movement',
     _high_low(), [('timeperiod', 14)], ['real'])

_reg('MINUS_DM', 'Momentum Indicators', 'Minus Directional Movement',
     _high_low(), [('timeperiod', 14)], ['real'])

# ======================== Volatility ========================

_reg('ATR', 'Volatility Indicators', 'Average True Range',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('NATR', 'Volatility Indicators', 'Normalized Average True Range',
     _high_low_close(), [('timeperiod', 14)], ['real'])

_reg('TRANGE', 'Volatility Indicators', 'True Range',
     _high_low_close(), [], ['real'])

# ======================== Volume ========================

_reg('AD', 'Volume Indicators', 'Chaikin A/D Line',
     [('prices', ['high', 'low', 'close']), ('price1', ['volume'])],
     [], ['real'])

_reg('ADOSC', 'Volume Indicators', 'Chaikin A/D Oscillator',
     [('prices', ['high', 'low', 'close']), ('price1', ['volume'])],
     [('fastperiod', 3), ('slowperiod', 10)], ['real'])

_reg('OBV', 'Volume Indicators', 'On Balance Volume',
     _close_volume(), [], ['real'])

# ======================== Price Transform ========================

_reg('AVGPRICE', 'Price Transform', 'Average Price',
     _ohlc(), [], ['real'])

_reg('MEDPRICE', 'Price Transform', 'Median Price',
     _high_low(), [], ['real'])

_reg('TYPPRICE', 'Price Transform', 'Typical Price',
     _high_low_close(), [], ['real'])

_reg('WCLPRICE', 'Price Transform', 'Weighted Close Price',
     _high_low_close(), [], ['real'])

# ======================== Statistic Functions ========================

_reg('STDDEV', 'Statistic Functions', 'Standard Deviation',
     _close(), [('timeperiod', 5), ('nbdev', 1.0)], ['real'])

_reg('VAR', 'Statistic Functions', 'Variance',
     _close(), [('timeperiod', 5), ('nbdev', 1.0)], ['real'])

_reg('BETA', 'Statistic Functions', 'Beta',
     _two_series(), [('timeperiod', 5)], ['real'])

_reg('CORREL', 'Statistic Functions', 'Pearson Correlation Coefficient',
     _two_series(), [('timeperiod', 30)], ['real'])

_reg('LINEARREG', 'Statistic Functions', 'Linear Regression',
     _close(), [('timeperiod', 14)], ['real'])

_reg('LINEARREG_SLOPE', 'Statistic Functions', 'Linear Regression Slope',
     _close(), [('timeperiod', 14)], ['real'])

_reg('LINEARREG_INTERCEPT', 'Statistic Functions', 'Linear Regression Intercept',
     _close(), [('timeperiod', 14)], ['real'])

_reg('LINEARREG_ANGLE', 'Statistic Functions', 'Linear Regression Angle',
     _close(), [('timeperiod', 14)], ['real'])

_reg('TSF', 'Statistic Functions', 'Time Series Forecast',
     _close(), [('timeperiod', 14)], ['real'])

# ======================== Math Transform ========================

for _mt_name, _mt_display in [
    ('ACOS', 'Vector Trigonometric ACos'),
    ('ASIN', 'Vector Trigonometric ASin'),
    ('ATAN', 'Vector Trigonometric ATan'),
    ('CEIL', 'Vector Ceil'),
    ('COS', 'Vector Trigonometric Cos'),
    ('COSH', 'Vector Trigonometric Cosh'),
    ('EXP', 'Vector Arithmetic Exp'),
    ('FLOOR', 'Vector Floor'),
    ('LN', 'Vector Log Natural'),
    ('LOG10', 'Vector Log10'),
    ('SIN', 'Vector Trigonometric Sin'),
    ('SINH', 'Vector Trigonometric Sinh'),
    ('SQRT', 'Vector Square Root'),
    ('TAN', 'Vector Trigonometric Tan'),
    ('TANH', 'Vector Trigonometric Tanh'),
]:
    _reg(_mt_name, 'Math Transform', _mt_display, _close(), [], ['real'])

# ======================== Math Operators ========================

_reg('ADD', 'Math Operators', 'Vector Arithmetic Add',
     _two_series(), [], ['real'])

_reg('SUB', 'Math Operators', 'Vector Arithmetic Sub',
     _two_series(), [], ['real'])

_reg('MULT', 'Math Operators', 'Vector Arithmetic Mult',
     _two_series(), [], ['real'])

_reg('DIV', 'Math Operators', 'Vector Arithmetic Div',
     _two_series(), [], ['real'])

_reg('MAX', 'Math Operators', 'Highest value over a specified period',
     _close(), [('timeperiod', 30)], ['real'])

_reg('MAXINDEX', 'Math Operators', 'Index of highest value over a specified period',
     _close(), [('timeperiod', 30)], ['integer'])

_reg('MIN', 'Math Operators', 'Lowest value over a specified period',
     _close(), [('timeperiod', 30)], ['real'])

_reg('MININDEX', 'Math Operators', 'Index of lowest value over a specified period',
     _close(), [('timeperiod', 30)], ['integer'])

_reg('SUM', 'Math Operators', 'Summation',
     _close(), [('timeperiod', 30)], ['real'])

# ======================== Cycle Indicators ========================

_reg('HT_DCPERIOD', 'Cycle Indicators', 'Hilbert Transform - Dominant Cycle Period',
     _close(), [], ['real'])

_reg('HT_DCPHASE', 'Cycle Indicators', 'Hilbert Transform - Dominant Cycle Phase',
     _close(), [], ['real'])

_reg('HT_PHASOR', 'Cycle Indicators', 'Hilbert Transform - Phasor Components',
     _close(), [], ['inphase', 'quadrature'])

_reg('HT_SINE', 'Cycle Indicators', 'Hilbert Transform - SineWave',
     _close(), [], ['sine', 'leadsine'])

_reg('HT_TRENDMODE', 'Cycle Indicators', 'Hilbert Transform - Trend vs Cycle Mode',
     _close(), [], ['integer'])

# ======================== Pattern Recognition ========================

_CDL_NAMES = [
    ('CDLDOJI', 'Doji'),
    ('CDLHAMMER', 'Hammer'),
    ('CDLENGULFING', 'Engulfing Pattern'),
    ('CDL2CROWS', 'Two Crows'),
    ('CDL3BLACKCROWS', 'Three Black Crows'),
    ('CDL3INSIDE', 'Three Inside Up/Down'),
    ('CDL3LINESTRIKE', 'Three-Line Strike'),
    ('CDL3OUTSIDE', 'Three Outside Up/Down'),
    ('CDL3STARSINSOUTH', 'Three Stars In The South'),
    ('CDL3WHITESOLDIERS', 'Three Advancing White Soldiers'),
    ('CDLABANDONEDBABY', 'Abandoned Baby'),
    ('CDLADVANCEBLOCK', 'Advance Block'),
    ('CDLBELTHOLD', 'Belt-hold'),
    ('CDLBREAKAWAY', 'Breakaway'),
    ('CDLCLOSINGMARUBOZU', 'Closing Marubozu'),
    ('CDLCONCEALBABYSWALL', 'Concealing Baby Swallow'),
    ('CDLCOUNTERATTACK', 'Counterattack'),
    ('CDLDARKCLOUDCOVER', 'Dark Cloud Cover'),
    ('CDLDOJISTAR', 'Doji Star'),
    ('CDLDRAGONFLYDOJI', 'Dragonfly Doji'),
    ('CDLEVENINGDOJISTAR', 'Evening Doji Star'),
    ('CDLEVENINGSTAR', 'Evening Star'),
    ('CDLGAPSIDESIDEWHITE', 'Up/Down-gap side-by-side white lines'),
    ('CDLGRAVESTONEDOJI', 'Gravestone Doji'),
    ('CDLHANGINGMAN', 'Hanging Man'),
    ('CDLHARAMI', 'Harami Pattern'),
    ('CDLHARAMICROSS', 'Harami Cross Pattern'),
    ('CDLHIGHWAVE', 'High-Wave Candle'),
    ('CDLHIKKAKE', 'Hikkake Pattern'),
    ('CDLHIKKAKEMOD', 'Modified Hikkake Pattern'),
    ('CDLHOMINGPIGEON', 'Homing Pigeon'),
    ('CDLIDENTICAL3CROWS', 'Identical Three Crows'),
    ('CDLINNECK', 'In-Neck Pattern'),
    ('CDLINVERTEDHAMMER', 'Inverted Hammer'),
    ('CDLKICKING', 'Kicking'),
    ('CDLKICKINGBYLENGTH', 'Kicking - bull/bear determined by the longer marubozu'),
    ('CDLLADDERBOTTOM', 'Ladder Bottom'),
    ('CDLLONGLEGGEDDOJI', 'Long Legged Doji'),
    ('CDLLONGLINE', 'Long Line Candle'),
    ('CDLMARUBOZU', 'Marubozu'),
    ('CDLMATCHINGLOW', 'Matching Low'),
    ('CDLMATHOLD', 'Mat Hold'),
    ('CDLMORNINGDOJISTAR', 'Morning Doji Star'),
    ('CDLMORNINGSTAR', 'Morning Star'),
    ('CDLONNECK', 'On-Neck Pattern'),
    ('CDLPIERCING', 'Piercing Pattern'),
    ('CDLRICKSHAWMAN', 'Rickshaw Man'),
    ('CDLRISEFALL3METHODS', 'Rising/Falling Three Methods'),
    ('CDLSEPARATINGLINES', 'Separating Lines'),
    ('CDLSHOOTINGSTAR', 'Shooting Star'),
    ('CDLSHORTLINE', 'Short Line Candle'),
    ('CDLSPINNINGTOP', 'Spinning Top'),
    ('CDLSTALLEDPATTERN', 'Stalled Pattern'),
    ('CDLSTICKSANDWICH', 'Stick Sandwich'),
    ('CDLTAKURI', 'Takuri (Dragonfly Doji with very long lower shadow)'),
    ('CDLTASUKIGAP', 'Tasuki Gap'),
    ('CDLTHRUSTING', 'Thrusting Pattern'),
    ('CDLTRISTAR', 'Tristar Pattern'),
    ('CDLUNIQUE3RIVER', 'Unique 3 River'),
    ('CDLUPSIDEGAP2CROWS', 'Upside Gap Two Crows'),
    ('CDLXSIDEGAP3METHODS', 'Upside/Downside Gap Three Methods'),
]

for _cdl_name, _cdl_display in _CDL_NAMES:
    _reg(_cdl_name, 'Pattern Recognition', _cdl_display,
         _ohlc(), [], ['integer'],
         {'integer': _PATTERN_BULL_BEAR})


# ---------------------------------------------------------------------------
# Helpers to resolve input arrays from a dict/DataFrame
# ---------------------------------------------------------------------------

def _get_array(input_arrays, key):
    """Extract a numpy array from input_arrays (dict or DataFrame) by key."""
    if hasattr(input_arrays, 'to_numpy'):
        # pandas Series
        return np.asarray(input_arrays[key], dtype=np.float64)
    if hasattr(input_arrays, '__getitem__'):
        val = input_arrays[key]
        if hasattr(val, 'to_numpy'):
            return np.asarray(val, dtype=np.float64)
        return np.asarray(val, dtype=np.float64)
    raise TypeError(f"Cannot extract '{key}' from input_arrays of type {type(input_arrays)}")


def _resolve_inputs(info, input_arrays):
    """
    Resolve the ordered positional input arrays required to call the underlying
    C function, based on the metadata input_names.

    Returns a list of numpy arrays in the order expected by _talib.<FUNC>.
    """
    arrays = []
    for _key, price_series in info['input_names'].items():
        for col in price_series:
            arrays.append(_get_array(input_arrays, col))
    return arrays


# ---------------------------------------------------------------------------
# Function class -- the core of the Abstract API
# ---------------------------------------------------------------------------

class Function:
    """
    TA-Lib abstract Function wrapper.

    Provides a higher-level interface that accepts dicts / DataFrames of
    price data and supports named parameters.

    Compatible with the original ``talib.abstract.Function`` API:

        >>> sma = Function('SMA')
        >>> sma.parameters
        OrderedDict([('timeperiod', 30)])
        >>> sma.parameters = {'timeperiod': 20}
        >>> result = sma({'close': close_array})
    """

    def __new__(cls, function_name, *args, **kwargs):
        # When called as Function('SMA', ...) we want to return a Function
        # *instance*, not a bare Rust function.  __new__ + __init__ is the
        # standard pattern.
        return super().__new__(cls)

    def __init__(self, function_name, func_object=None, **kwargs):
        self._name = function_name.upper()
        if self._name not in _FUNC_INFO:
            raise ValueError(f"Invalid function name: {self._name}")

        # Deep-copy so mutations don't affect the global registry
        self._info = copy.deepcopy(_FUNC_INFO[self._name])

        # The actual callable in the Rust extension
        self._func = func_object or getattr(_talib, self._name)

        # Allow setting initial parameters via kwargs
        if kwargs:
            self.set_parameters(kwargs)

    # ---- Properties ----

    @property
    def info(self) -> dict:
        """Return the full info dict for this function."""
        return self._info

    @property
    def function_flags(self) -> list:
        """Return function flags (TA-Lib compat, typically empty for us)."""
        return self._info.get('function_flags', [])

    @property
    def input_names(self) -> OrderedDict:
        """Ordered dict mapping input group keys to their price series names."""
        return self._info['input_names']

    @input_names.setter
    def input_names(self, new_names):
        self._info['input_names'] = OrderedDict(new_names)

    @property
    def parameters(self) -> OrderedDict:
        """Current parameter values."""
        return self._info['parameters']

    @parameters.setter
    def parameters(self, new_params):
        """Set one or more parameters by dict."""
        self.set_parameters(new_params)

    @property
    def output_names(self) -> list:
        return self._info['output_names']

    @property
    def output_flags(self) -> OrderedDict:
        return self._info['output_flags']

    @property
    def lookback(self) -> int:
        """
        Compute the lookback period for the current parameter settings.

        This calls the underlying function with a minimal array and measures
        how many leading NaN values are produced.
        """
        # Use a reasonably sized dummy array
        size = 200
        dummy = np.ones(size, dtype=np.float64)

        # Build dummy input_arrays
        input_arrays = {}
        for _key, series_names in self._info['input_names'].items():
            for col in series_names:
                if col not in input_arrays:
                    input_arrays[col] = dummy

        try:
            result = self._call(input_arrays)
            # result may be a single array or tuple of arrays
            if isinstance(result, tuple):
                arr = np.asarray(result[0])
            else:
                arr = np.asarray(result)
            # Count leading NaN values
            lookback = 0
            for val in arr:
                if np.isnan(val):
                    lookback += 1
                else:
                    break
            return lookback
        except Exception:
            return 0

    # ---- Parameter helpers ----

    def set_parameters(self, new_params: dict):
        """Update parameters, ignoring unknown keys."""
        for key, value in new_params.items():
            if key in self._info['parameters']:
                self._info['parameters'][key] = value

    def set_function_args(self, *args, **kwargs):
        """
        Set input arrays (positional) and parameters (keyword).
        Returns self for chaining.
        """
        if args:
            self._input_arrays = args[0]
        # Remaining kwargs are parameters
        self.set_parameters(kwargs)
        return self

    # ---- Calling ----

    def _call(self, input_arrays, **kwargs):
        """Internal: build args and call the Rust function."""
        # Merge any override kwargs into parameters
        params = OrderedDict(self._info['parameters'])
        for k, v in kwargs.items():
            if k in params:
                params[k] = v

        # Resolve input arrays
        pos_args = _resolve_inputs(self._info, input_arrays)

        # Append parameter values in order
        all_args = pos_args + list(params.values())

        return self._func(*all_args)

    def run(self, input_arrays, **kwargs):
        """
        Run the function with the given input arrays and optional parameter
        overrides.  Same as calling the instance directly.
        """
        return self(input_arrays, **kwargs)

    def __call__(self, input_arrays=None, **kwargs):
        """
        Call the function.

        Parameters
        ----------
        input_arrays : dict or DataFrame
            Mapping of column names to numpy arrays.
        **kwargs
            Parameter overrides for this call only.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
        """
        if input_arrays is None:
            input_arrays = getattr(self, '_input_arrays', None)
        if input_arrays is None:
            raise TypeError("input_arrays is required")

        return self._call(input_arrays, **kwargs)

    # ---- Representation ----

    def __repr__(self):
        return f"Function({self._name!r})"

    def __str__(self):
        return f"{self._name}({', '.join(f'{k}={v}' for k, v in self._info['parameters'].items())})"


# ---------------------------------------------------------------------------
# Module-level attribute access: ``from talib.abstract import SMA``
# ---------------------------------------------------------------------------
# We create pre-instantiated Function objects for every registered function
# and expose them as module attributes.
# ---------------------------------------------------------------------------

def _make_func(name):
    """Create a module-level callable Function instance."""
    return Function(name)


# Populate module namespace so that ``from talib.abstract import SMA`` works.
_this_module = __import__(__name__)
for _fname in _FUNC_INFO:
    globals()[_fname] = _make_func(_fname)


def __getattr__(name):
    """
    Fallback for attribute access on the module.
    Allows ``talib.abstract.SMA`` even if not yet in globals
    (e.g. after dynamic registration).
    """
    upper = name.upper()
    if upper in _FUNC_INFO:
        func = _make_func(upper)
        globals()[name] = func
        return func
    raise AttributeError(f"module 'talib.abstract' has no attribute {name!r}")
