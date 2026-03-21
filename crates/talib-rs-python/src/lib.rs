#![allow(non_snake_case)]

use pyo3::prelude::*;

mod conversion;
mod func_api;
mod metadata;

/// Python 模块入口 — `talib._talib`
#[pymodule]
fn _talib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 元数据 API
    m.add_function(wrap_pyfunction!(metadata::get_functions, m)?)?;
    m.add_function(wrap_pyfunction!(metadata::get_function_groups, m)?)?;

    // ===== Overlap Studies =====
    m.add_function(wrap_pyfunction!(func_api::SMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::EMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::WMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::DEMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TEMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TRIMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::KAMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::T3, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MAMA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::BBANDS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SAREXT, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MIDPOINT, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MIDPRICE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MAVP, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::HT_TRENDLINE, m)?)?;

    // ===== Momentum Indicators =====
    m.add_function(wrap_pyfunction!(func_api::RSI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MACD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MACDEXT, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MACDFIX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::STOCH, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::STOCHF, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::STOCHRSI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ADX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ADXR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CCI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MOM, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ROC, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ROCP, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ROCR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ROCR100, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::WILLR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::APO, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::PPO, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::BOP, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CMO, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::AROON, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::AROONOSC, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MFI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TRIX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ULTOSC, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::DX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::PLUS_DI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MINUS_DI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::PLUS_DM, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MINUS_DM, m)?)?;

    // ===== Volatility =====
    m.add_function(wrap_pyfunction!(func_api::ATR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::NATR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TRANGE, m)?)?;

    // ===== Volume =====
    m.add_function(wrap_pyfunction!(func_api::AD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ADOSC, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::OBV, m)?)?;

    // ===== Price Transform =====
    m.add_function(wrap_pyfunction!(func_api::AVGPRICE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MEDPRICE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TYPPRICE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::WCLPRICE, m)?)?;

    // ===== Statistic =====
    m.add_function(wrap_pyfunction!(func_api::STDDEV, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::VAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::BETA, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CORREL, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::LINEARREG, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::LINEARREG_SLOPE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::LINEARREG_INTERCEPT, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::LINEARREG_ANGLE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TSF, m)?)?;

    // ===== Math Transform =====
    m.add_function(wrap_pyfunction!(func_api::ACOS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ASIN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::ATAN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CEIL, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::COS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::COSH, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::EXP, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::FLOOR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::LN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::LOG10, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SIN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SINH, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SQRT, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TAN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::TANH, m)?)?;

    // ===== Math Operators =====
    m.add_function(wrap_pyfunction!(func_api::ADD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SUB, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MULT, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::DIV, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MAX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MAXINDEX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MIN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::MININDEX, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::SUM, m)?)?;

    // ===== Cycle Indicators =====
    m.add_function(wrap_pyfunction!(func_api::HT_DCPERIOD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::HT_DCPHASE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::HT_PHASOR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::HT_SINE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::HT_TRENDMODE, m)?)?;

    // ===== Pattern Recognition =====
    m.add_function(wrap_pyfunction!(func_api::CDLDOJI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHAMMER, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLENGULFING, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL2CROWS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL3BLACKCROWS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL3INSIDE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL3LINESTRIKE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL3OUTSIDE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL3STARSINSOUTH, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDL3WHITESOLDIERS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLABANDONEDBABY, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLADVANCEBLOCK, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLBELTHOLD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLBREAKAWAY, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLCLOSINGMARUBOZU, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLCONCEALBABYSWALL, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLCOUNTERATTACK, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLDARKCLOUDCOVER, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLDOJISTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLDRAGONFLYDOJI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLEVENINGDOJISTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLEVENINGSTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLGAPSIDESIDEWHITE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLGRAVESTONEDOJI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHANGINGMAN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHARAMI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHARAMICROSS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHIGHWAVE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHIKKAKE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHIKKAKEMOD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLHOMINGPIGEON, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLIDENTICAL3CROWS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLINNECK, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLINVERTEDHAMMER, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLKICKING, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLKICKINGBYLENGTH, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLLADDERBOTTOM, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLLONGLEGGEDDOJI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLLONGLINE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLMARUBOZU, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLMATCHINGLOW, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLMATHOLD, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLMORNINGDOJISTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLMORNINGSTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLONNECK, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLPIERCING, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLRICKSHAWMAN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLRISEFALL3METHODS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLSEPARATINGLINES, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLSHOOTINGSTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLSHORTLINE, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLSPINNINGTOP, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLSTALLEDPATTERN, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLSTICKSANDWICH, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLTAKURI, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLTASUKIGAP, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLTHRUSTING, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLTRISTAR, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLUNIQUE3RIVER, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLUPSIDEGAP2CROWS, m)?)?;
    m.add_function(wrap_pyfunction!(func_api::CDLXSIDEGAP3METHODS, m)?)?;

    Ok(())
}
