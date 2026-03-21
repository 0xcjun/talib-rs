use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ta_rs_core as core;

use crate::conversion::to_py_array;

// 辅助宏: 将 TaError 转为 PyValueError
macro_rules! ta_err {
    ($expr:expr) => {
        $expr.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    };
}

// ============================================================
// Overlap Studies
// ============================================================

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn SMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::sma(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn EMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::ema(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn WMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::wma(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn DEMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::dema(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn TEMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::tema(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn TRIMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::trima(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn KAMA(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::kama(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=5, vfactor=0.7))]
pub fn T3(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize, vfactor: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::t3(input.as_slice()?, timeperiod, vfactor))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, fastlimit=0.5, slowlimit=0.05))]
pub fn MAMA(py: Python<'_>, input: PyReadonlyArray1<f64>, fastlimit: f64, slowlimit: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (mama, fama) = ta_err!(core::overlap::mama(input.as_slice()?, fastlimit, slowlimit))?;
    Ok((to_py_array(py, mama), to_py_array(py, fama)))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0))]
pub fn BBANDS(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize, nbdevup: f64, nbdevdn: f64, matype: i32) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let ma = core::MaType::try_from(matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let (upper, middle, lower) = ta_err!(core::overlap::bbands(input.as_slice()?, timeperiod, nbdevup, nbdevdn, ma))?;
    Ok((to_py_array(py, upper), to_py_array(py, middle), to_py_array(py, lower)))
}

#[pyfunction]
#[pyo3(signature = (high, low, acceleration=0.02, maximum=0.2))]
pub fn SAR(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, acceleration: f64, maximum: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::sar(high.as_slice()?, low.as_slice()?, acceleration, maximum))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, startvalue=0.0, offsetonreverse=0.0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2))]
pub fn SAREXT(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, startvalue: f64, offsetonreverse: f64, accelerationinitlong: f64, accelerationlong: f64, accelerationmaxlong: f64, accelerationinitshort: f64, accelerationshort: f64, accelerationmaxshort: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::sar_ext(high.as_slice()?, low.as_slice()?, startvalue, offsetonreverse, accelerationinitlong, accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn MIDPOINT(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::midpoint(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod=14))]
pub fn MIDPRICE(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::midprice(high.as_slice()?, low.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, periods, minperiod=2, maxperiod=30, matype=0))]
pub fn MAVP(py: Python<'_>, input: PyReadonlyArray1<f64>, periods: PyReadonlyArray1<f64>, minperiod: usize, maxperiod: usize, matype: i32) -> PyResult<Py<PyArray1<f64>>> {
    let ma = core::MaType::try_from(matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = ta_err!(core::overlap::mavp(input.as_slice()?, periods.as_slice()?, minperiod, maxperiod, ma))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn HT_TRENDLINE(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::overlap::ht_trendline(input.as_slice()?))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Momentum Indicators
// ============================================================

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn RSI(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::rsi(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, fastperiod=12, slowperiod=26, signalperiod=9))]
pub fn MACD(py: Python<'_>, input: PyReadonlyArray1<f64>, fastperiod: usize, slowperiod: usize, signalperiod: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (m, s, h) = ta_err!(core::momentum::macd(input.as_slice()?, fastperiod, slowperiod, signalperiod))?;
    Ok((to_py_array(py, m), to_py_array(py, s), to_py_array(py, h)))
}

#[pyfunction]
#[pyo3(signature = (input, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9, signalmatype=1))]
pub fn MACDEXT(py: Python<'_>, input: PyReadonlyArray1<f64>, fastperiod: usize, fastmatype: i32, slowperiod: usize, slowmatype: i32, signalperiod: usize, signalmatype: i32) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let fmt = core::MaType::try_from(fastmatype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let smt = core::MaType::try_from(slowmatype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let sigmt = core::MaType::try_from(signalmatype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let (m, s, h) = ta_err!(core::momentum::macd_ext(input.as_slice()?, fastperiod, fmt, slowperiod, smt, signalperiod, sigmt))?;
    Ok((to_py_array(py, m), to_py_array(py, s), to_py_array(py, h)))
}

#[pyfunction]
#[pyo3(signature = (input, signalperiod=9))]
pub fn MACDFIX(py: Python<'_>, input: PyReadonlyArray1<f64>, signalperiod: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (m, s, h) = ta_err!(core::momentum::macd_fix(input.as_slice()?, signalperiod))?;
    Ok((to_py_array(py, m), to_py_array(py, s), to_py_array(py, h)))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0))]
pub fn STOCH(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, fastk_period: usize, slowk_period: usize, slowk_matype: i32, slowd_period: usize, slowd_matype: i32) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let skm = core::MaType::try_from(slowk_matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let sdm = core::MaType::try_from(slowd_matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let (k, d) = ta_err!(core::momentum::stoch(high.as_slice()?, low.as_slice()?, close.as_slice()?, fastk_period, slowk_period, skm, slowd_period, sdm))?;
    Ok((to_py_array(py, k), to_py_array(py, d)))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0))]
pub fn STOCHF(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, fastk_period: usize, fastd_period: usize, fastd_matype: i32) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let fdm = core::MaType::try_from(fastd_matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let (k, d) = ta_err!(core::momentum::stochf(high.as_slice()?, low.as_slice()?, close.as_slice()?, fastk_period, fastd_period, fdm))?;
    Ok((to_py_array(py, k), to_py_array(py, d)))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0))]
pub fn STOCHRSI(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize, fastk_period: usize, fastd_period: usize, fastd_matype: i32) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let fdm = core::MaType::try_from(fastd_matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let (k, d) = ta_err!(core::momentum::stochrsi(input.as_slice()?, timeperiod, fastk_period, fastd_period, fdm))?;
    Ok((to_py_array(py, k), to_py_array(py, d)))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn ADX(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::adx(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn ADXR(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::adxr(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn CCI(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::cci(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=10))]
pub fn MOM(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::mom(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=10))]
pub fn ROC(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::roc(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=10))]
pub fn ROCP(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::rocp(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=10))]
pub fn ROCR(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::rocr(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=10))]
pub fn ROCR100(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::rocr100(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn WILLR(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::willr(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, fastperiod=12, slowperiod=26, matype=0))]
pub fn APO(py: Python<'_>, input: PyReadonlyArray1<f64>, fastperiod: usize, slowperiod: usize, matype: i32) -> PyResult<Py<PyArray1<f64>>> {
    let ma = core::MaType::try_from(matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = ta_err!(core::momentum::apo(input.as_slice()?, fastperiod, slowperiod, ma))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, fastperiod=12, slowperiod=26, matype=0))]
pub fn PPO(py: Python<'_>, input: PyReadonlyArray1<f64>, fastperiod: usize, slowperiod: usize, matype: i32) -> PyResult<Py<PyArray1<f64>>> {
    let ma = core::MaType::try_from(matype).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = ta_err!(core::momentum::ppo(input.as_slice()?, fastperiod, slowperiod, ma))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn BOP(py: Python<'_>, open: PyReadonlyArray1<f64>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::bop(open.as_slice()?, high.as_slice()?, low.as_slice()?, close.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn CMO(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::cmo(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod=14))]
pub fn AROON(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (down, up) = ta_err!(core::momentum::aroon(high.as_slice()?, low.as_slice()?, timeperiod))?;
    Ok((to_py_array(py, down), to_py_array(py, up)))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod=14))]
pub fn AROONOSC(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::aroon_osc(high.as_slice()?, low.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, timeperiod=14))]
pub fn MFI(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, volume: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::mfi(high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn TRIX(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::trix(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28))]
pub fn ULTOSC(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod1: usize, timeperiod2: usize, timeperiod3: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::ultosc(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod1, timeperiod2, timeperiod3))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn DX(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::dx(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn PLUS_DI(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::plus_di(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn MINUS_DI(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::minus_di(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod=14))]
pub fn PLUS_DM(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::plus_dm(high.as_slice()?, low.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod=14))]
pub fn MINUS_DM(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::momentum::minus_dm(high.as_slice()?, low.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Volatility
// ============================================================

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn ATR(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::volatility::atr(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod=14))]
pub fn NATR(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::volatility::natr(high.as_slice()?, low.as_slice()?, close.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn TRANGE(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::volatility::trange(high.as_slice()?, low.as_slice()?, close.as_slice()?))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Volume
// ============================================================

#[pyfunction]
pub fn AD(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, volume: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::volume::ad(high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, fastperiod=3, slowperiod=10))]
pub fn ADOSC(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, volume: PyReadonlyArray1<f64>, fastperiod: usize, slowperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::volume::adosc(high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?, fastperiod, slowperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn OBV(py: Python<'_>, close: PyReadonlyArray1<f64>, volume: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::volume::obv(close.as_slice()?, volume.as_slice()?))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Price Transform
// ============================================================

#[pyfunction]
pub fn AVGPRICE(py: Python<'_>, open: PyReadonlyArray1<f64>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::price_transform::avgprice(open.as_slice()?, high.as_slice()?, low.as_slice()?, close.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn MEDPRICE(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::price_transform::medprice(high.as_slice()?, low.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn TYPPRICE(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::price_transform::typprice(high.as_slice()?, low.as_slice()?, close.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn WCLPRICE(py: Python<'_>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::price_transform::wclprice(high.as_slice()?, low.as_slice()?, close.as_slice()?))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Statistic Functions
// ============================================================

#[pyfunction]
#[pyo3(signature = (input, timeperiod=5, nbdev=1.0))]
pub fn STDDEV(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize, nbdev: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::stddev(input.as_slice()?, timeperiod, nbdev))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=5, nbdev=1.0))]
pub fn VAR(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize, nbdev: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::var(input.as_slice()?, timeperiod, nbdev))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input0, input1, timeperiod=5))]
pub fn BETA(py: Python<'_>, input0: PyReadonlyArray1<f64>, input1: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::beta(input0.as_slice()?, input1.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input0, input1, timeperiod=30))]
pub fn CORREL(py: Python<'_>, input0: PyReadonlyArray1<f64>, input1: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::correl(input0.as_slice()?, input1.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn LINEARREG(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::linearreg(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn LINEARREG_SLOPE(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::linearreg_slope(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn LINEARREG_INTERCEPT(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::linearreg_intercept(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn LINEARREG_ANGLE(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::linearreg_angle(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=14))]
pub fn TSF(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::statistic::tsf(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Math Transform
// ============================================================

macro_rules! math_transform_py {
    ($name:ident, $func:path) => {
        #[pyfunction]
        pub fn $name(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
            let result = $func(input.as_slice()?);
            Ok(to_py_array(py, result))
        }
    };
}

math_transform_py!(ACOS, core::math_transform::acos);
math_transform_py!(ASIN, core::math_transform::asin);
math_transform_py!(ATAN, core::math_transform::atan);
math_transform_py!(CEIL, core::math_transform::ceil);
math_transform_py!(COS, core::math_transform::cos);
math_transform_py!(COSH, core::math_transform::cosh);
math_transform_py!(EXP, core::math_transform::exp);
math_transform_py!(FLOOR, core::math_transform::floor);
math_transform_py!(LN, core::math_transform::ln);
math_transform_py!(LOG10, core::math_transform::log10);
math_transform_py!(SIN, core::math_transform::sin);
math_transform_py!(SINH, core::math_transform::sinh);
math_transform_py!(SQRT, core::math_transform::sqrt);
math_transform_py!(TAN, core::math_transform::tan);
math_transform_py!(TANH, core::math_transform::tanh);

// ============================================================
// Math Operators
// ============================================================

#[pyfunction]
pub fn ADD(py: Python<'_>, input0: PyReadonlyArray1<f64>, input1: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::add(input0.as_slice()?, input1.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn SUB(py: Python<'_>, input0: PyReadonlyArray1<f64>, input1: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::sub(input0.as_slice()?, input1.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn MULT(py: Python<'_>, input0: PyReadonlyArray1<f64>, input1: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::mult(input0.as_slice()?, input1.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn DIV(py: Python<'_>, input0: PyReadonlyArray1<f64>, input1: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::div(input0.as_slice()?, input1.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn MAX(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::max(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn MAXINDEX(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::maxindex(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn MIN(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::min(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn MININDEX(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::minindex(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
#[pyo3(signature = (input, timeperiod=30))]
pub fn SUM(py: Python<'_>, input: PyReadonlyArray1<f64>, timeperiod: usize) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::math_operator::sum(input.as_slice()?, timeperiod))?;
    Ok(to_py_array(py, result))
}

// ============================================================
// Cycle Indicators
// ============================================================

#[pyfunction]
pub fn HT_DCPERIOD(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::cycle::ht_dcperiod(input.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn HT_DCPHASE(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = ta_err!(core::cycle::ht_dcphase(input.as_slice()?))?;
    Ok(to_py_array(py, result))
}

#[pyfunction]
pub fn HT_PHASOR(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (inphase, quadrature) = ta_err!(core::cycle::ht_phasor(input.as_slice()?))?;
    Ok((to_py_array(py, inphase), to_py_array(py, quadrature)))
}

#[pyfunction]
pub fn HT_SINE(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (sine, leadsine) = ta_err!(core::cycle::ht_sine(input.as_slice()?))?;
    Ok((to_py_array(py, sine), to_py_array(py, leadsine)))
}

#[pyfunction]
pub fn HT_TRENDMODE(py: Python<'_>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<i32>>> {
    let result = ta_err!(core::cycle::ht_trendmode(input.as_slice()?))?;
    Ok(crate::conversion::to_py_array_i32(py, result))
}

// ============================================================
// Pattern Recognition (Candlestick)
// ============================================================

macro_rules! cdl_pyfunction {
    ($name:ident, $func:path) => {
        #[pyfunction]
        pub fn $name(
            py: Python<'_>,
            open: PyReadonlyArray1<f64>,
            high: PyReadonlyArray1<f64>,
            low: PyReadonlyArray1<f64>,
            close: PyReadonlyArray1<f64>,
        ) -> PyResult<Py<PyArray1<i32>>> {
            let result = ta_err!($func(
                open.as_slice()?,
                high.as_slice()?,
                low.as_slice()?,
                close.as_slice()?,
            ))?;
            Ok(crate::conversion::to_py_array_i32(py, result))
        }
    };
}

// 所有 K 线形态绑定
cdl_pyfunction!(CDLDOJI, core::pattern::cdl_doji);
cdl_pyfunction!(CDLHAMMER, core::pattern::cdl_hammer);
cdl_pyfunction!(CDLENGULFING, core::pattern::cdl_engulfing);
cdl_pyfunction!(CDL2CROWS, core::pattern::cdl_2crows);
cdl_pyfunction!(CDL3BLACKCROWS, core::pattern::cdl_3blackcrows);
cdl_pyfunction!(CDL3INSIDE, core::pattern::cdl_3inside);
cdl_pyfunction!(CDL3LINESTRIKE, core::pattern::cdl_3linestrike);
cdl_pyfunction!(CDL3OUTSIDE, core::pattern::cdl_3outside);
cdl_pyfunction!(CDL3STARSINSOUTH, core::pattern::cdl_3starsinsouth);
cdl_pyfunction!(CDL3WHITESOLDIERS, core::pattern::cdl_3whitesoldiers);
cdl_pyfunction!(CDLABANDONEDBABY, core::pattern::cdl_abandonedbaby);
cdl_pyfunction!(CDLADVANCEBLOCK, core::pattern::cdl_advanceblock);
cdl_pyfunction!(CDLBELTHOLD, core::pattern::cdl_belthold);
cdl_pyfunction!(CDLBREAKAWAY, core::pattern::cdl_breakaway);
cdl_pyfunction!(CDLCLOSINGMARUBOZU, core::pattern::cdl_closingmarubozu);
cdl_pyfunction!(CDLCONCEALBABYSWALL, core::pattern::cdl_concealbabyswall);
cdl_pyfunction!(CDLCOUNTERATTACK, core::pattern::cdl_counterattack);
cdl_pyfunction!(CDLDARKCLOUDCOVER, core::pattern::cdl_darkcloudcover);
cdl_pyfunction!(CDLDOJISTAR, core::pattern::cdl_dojistar);
cdl_pyfunction!(CDLDRAGONFLYDOJI, core::pattern::cdl_dragonflydoji);
cdl_pyfunction!(CDLEVENINGDOJISTAR, core::pattern::cdl_eveningdojistar);
cdl_pyfunction!(CDLEVENINGSTAR, core::pattern::cdl_eveningstar);
cdl_pyfunction!(CDLGAPSIDESIDEWHITE, core::pattern::cdl_gapsidesidewhite);
cdl_pyfunction!(CDLGRAVESTONEDOJI, core::pattern::cdl_gravestonedoji);
cdl_pyfunction!(CDLHANGINGMAN, core::pattern::cdl_hangingman);
cdl_pyfunction!(CDLHARAMI, core::pattern::cdl_harami);
cdl_pyfunction!(CDLHARAMICROSS, core::pattern::cdl_haramicross);
cdl_pyfunction!(CDLHIGHWAVE, core::pattern::cdl_highwave);
cdl_pyfunction!(CDLHIKKAKE, core::pattern::cdl_hikkake);
cdl_pyfunction!(CDLHIKKAKEMOD, core::pattern::cdl_hikkakemod);
cdl_pyfunction!(CDLHOMINGPIGEON, core::pattern::cdl_homingpigeon);
cdl_pyfunction!(CDLIDENTICAL3CROWS, core::pattern::cdl_identical3crows);
cdl_pyfunction!(CDLINNECK, core::pattern::cdl_inneck);
cdl_pyfunction!(CDLINVERTEDHAMMER, core::pattern::cdl_invertedhammer);
cdl_pyfunction!(CDLKICKING, core::pattern::cdl_kicking);
cdl_pyfunction!(CDLKICKINGBYLENGTH, core::pattern::cdl_kickingbylength);
cdl_pyfunction!(CDLLADDERBOTTOM, core::pattern::cdl_ladderbottom);
cdl_pyfunction!(CDLLONGLEGGEDDOJI, core::pattern::cdl_longleggeddoji);
cdl_pyfunction!(CDLLONGLINE, core::pattern::cdl_longline);
cdl_pyfunction!(CDLMARUBOZU, core::pattern::cdl_marubozu);
cdl_pyfunction!(CDLMATCHINGLOW, core::pattern::cdl_matchinglow);
cdl_pyfunction!(CDLMATHOLD, core::pattern::cdl_mathold);
cdl_pyfunction!(CDLMORNINGDOJISTAR, core::pattern::cdl_morningdojistar);
cdl_pyfunction!(CDLMORNINGSTAR, core::pattern::cdl_morningstar);
cdl_pyfunction!(CDLONNECK, core::pattern::cdl_onneck);
cdl_pyfunction!(CDLPIERCING, core::pattern::cdl_piercing);
cdl_pyfunction!(CDLRICKSHAWMAN, core::pattern::cdl_rickshawman);
cdl_pyfunction!(CDLRISEFALL3METHODS, core::pattern::cdl_risefall3methods);
cdl_pyfunction!(CDLSEPARATINGLINES, core::pattern::cdl_separatinglines);
cdl_pyfunction!(CDLSHOOTINGSTAR, core::pattern::cdl_shootingstar);
cdl_pyfunction!(CDLSHORTLINE, core::pattern::cdl_shortline);
cdl_pyfunction!(CDLSPINNINGTOP, core::pattern::cdl_spinningtop);
cdl_pyfunction!(CDLSTALLEDPATTERN, core::pattern::cdl_stalledpattern);
cdl_pyfunction!(CDLSTICKSANDWICH, core::pattern::cdl_sticksandwich);
cdl_pyfunction!(CDLTAKURI, core::pattern::cdl_takuri);
cdl_pyfunction!(CDLTASUKIGAP, core::pattern::cdl_tasukigap);
cdl_pyfunction!(CDLTHRUSTING, core::pattern::cdl_thrusting);
cdl_pyfunction!(CDLTRISTAR, core::pattern::cdl_tristar);
cdl_pyfunction!(CDLUNIQUE3RIVER, core::pattern::cdl_unique3river);
cdl_pyfunction!(CDLUPSIDEGAP2CROWS, core::pattern::cdl_upsidegap2crows);
cdl_pyfunction!(CDLXSIDEGAP3METHODS, core::pattern::cdl_xsidegap3methods);
