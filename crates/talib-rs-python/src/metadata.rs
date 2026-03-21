use pyo3::prelude::*;
use std::collections::HashMap;

/// 返回所有已实现的函数名列表
#[pyfunction]
pub fn get_functions() -> Vec<String> {
    let mut funcs = Vec::new();

    // Overlap Studies
    for name in &[
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "TRIMA",
        "KAMA",
        "T3",
        "MAMA",
        "BBANDS",
        "SAR",
        "SAREXT",
        "MIDPOINT",
        "MIDPRICE",
        "MAVP",
        "HT_TRENDLINE",
    ] {
        funcs.push(name.to_string());
    }

    // Momentum Indicators
    for name in &[
        "RSI", "MACD", "MACDEXT", "MACDFIX", "STOCH", "STOCHF", "STOCHRSI", "ADX", "ADXR", "CCI",
        "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "WILLR", "APO", "PPO", "BOP", "CMO", "AROON",
        "AROONOSC", "MFI", "TRIX", "ULTOSC", "DX", "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM",
    ] {
        funcs.push(name.to_string());
    }

    // Volatility
    for name in &["ATR", "NATR", "TRANGE"] {
        funcs.push(name.to_string());
    }

    // Volume
    for name in &["AD", "ADOSC", "OBV"] {
        funcs.push(name.to_string());
    }

    // Price Transform
    for name in &["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"] {
        funcs.push(name.to_string());
    }

    // Statistic
    for name in &[
        "STDDEV",
        "VAR",
        "BETA",
        "CORREL",
        "LINEARREG",
        "LINEARREG_SLOPE",
        "LINEARREG_INTERCEPT",
        "LINEARREG_ANGLE",
        "TSF",
    ] {
        funcs.push(name.to_string());
    }

    // Math Transform
    for name in &[
        "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP", "FLOOR", "LN", "LOG10", "SIN",
        "SINH", "SQRT", "TAN", "TANH",
    ] {
        funcs.push(name.to_string());
    }

    // Math Operators
    for name in &[
        "ADD", "SUB", "MULT", "DIV", "MAX", "MAXINDEX", "MIN", "MININDEX", "SUM",
    ] {
        funcs.push(name.to_string());
    }

    // Cycle
    for name in &[
        "HT_DCPERIOD",
        "HT_DCPHASE",
        "HT_PHASOR",
        "HT_SINE",
        "HT_TRENDMODE",
    ] {
        funcs.push(name.to_string());
    }

    // Pattern Recognition
    for name in &[
        "CDLDOJI",
        "CDLHAMMER",
        "CDLENGULFING",
        "CDL2CROWS",
        "CDL3BLACKCROWS",
        "CDL3INSIDE",
        "CDL3LINESTRIKE",
        "CDL3OUTSIDE",
        "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS",
        "CDLABANDONEDBABY",
        "CDLADVANCEBLOCK",
        "CDLBELTHOLD",
        "CDLBREAKAWAY",
        "CDLCLOSINGMARUBOZU",
        "CDLCONCEALBABYSWALL",
        "CDLCOUNTERATTACK",
        "CDLDARKCLOUDCOVER",
        "CDLDOJISTAR",
        "CDLDRAGONFLYDOJI",
        "CDLEVENINGDOJISTAR",
        "CDLEVENINGSTAR",
        "CDLGAPSIDESIDEWHITE",
        "CDLGRAVESTONEDOJI",
        "CDLHANGINGMAN",
        "CDLHARAMI",
        "CDLHARAMICROSS",
        "CDLHIGHWAVE",
        "CDLHIKKAKE",
        "CDLHIKKAKEMOD",
        "CDLHOMINGPIGEON",
        "CDLIDENTICAL3CROWS",
        "CDLINNECK",
        "CDLINVERTEDHAMMER",
        "CDLKICKING",
        "CDLKICKINGBYLENGTH",
        "CDLLADDERBOTTOM",
        "CDLLONGLEGGEDDOJI",
        "CDLLONGLINE",
        "CDLMARUBOZU",
        "CDLMATCHINGLOW",
        "CDLMATHOLD",
        "CDLMORNINGDOJISTAR",
        "CDLMORNINGSTAR",
        "CDLONNECK",
        "CDLPIERCING",
        "CDLRICKSHAWMAN",
        "CDLRISEFALL3METHODS",
        "CDLSEPARATINGLINES",
        "CDLSHOOTINGSTAR",
        "CDLSHORTLINE",
        "CDLSPINNINGTOP",
        "CDLSTALLEDPATTERN",
        "CDLSTICKSANDWICH",
        "CDLTAKURI",
        "CDLTASUKIGAP",
        "CDLTHRUSTING",
        "CDLTRISTAR",
        "CDLUNIQUE3RIVER",
        "CDLUPSIDEGAP2CROWS",
        "CDLXSIDEGAP3METHODS",
    ] {
        funcs.push(name.to_string());
    }

    funcs
}

/// 返回按组分类的函数映射
#[pyfunction]
pub fn get_function_groups() -> HashMap<String, Vec<String>> {
    let mut groups = HashMap::new();

    groups.insert(
        "Overlap Studies".to_string(),
        vec![
            "SMA",
            "EMA",
            "WMA",
            "DEMA",
            "TEMA",
            "TRIMA",
            "KAMA",
            "T3",
            "MAMA",
            "BBANDS",
            "SAR",
            "SAREXT",
            "MIDPOINT",
            "MIDPRICE",
            "MAVP",
            "HT_TRENDLINE",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups.insert(
        "Momentum Indicators".to_string(),
        vec![
            "RSI", "MACD", "MACDEXT", "MACDFIX", "STOCH", "STOCHF", "STOCHRSI", "ADX", "ADXR",
            "CCI", "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "WILLR", "APO", "PPO", "BOP", "CMO",
            "AROON", "AROONOSC", "MFI", "TRIX", "ULTOSC", "DX", "PLUS_DI", "MINUS_DI", "PLUS_DM",
            "MINUS_DM",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups.insert(
        "Volatility Indicators".to_string(),
        vec!["ATR", "NATR", "TRANGE"]
            .into_iter()
            .map(String::from)
            .collect(),
    );

    groups.insert(
        "Volume Indicators".to_string(),
        vec!["AD", "ADOSC", "OBV"]
            .into_iter()
            .map(String::from)
            .collect(),
    );

    groups.insert(
        "Price Transform".to_string(),
        vec!["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
            .into_iter()
            .map(String::from)
            .collect(),
    );

    groups.insert(
        "Statistic Functions".to_string(),
        vec![
            "STDDEV",
            "VAR",
            "BETA",
            "CORREL",
            "LINEARREG",
            "LINEARREG_SLOPE",
            "LINEARREG_INTERCEPT",
            "LINEARREG_ANGLE",
            "TSF",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups.insert(
        "Math Transform".to_string(),
        vec![
            "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP", "FLOOR", "LN", "LOG10", "SIN",
            "SINH", "SQRT", "TAN", "TANH",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups.insert(
        "Math Operators".to_string(),
        vec![
            "ADD", "SUB", "MULT", "DIV", "MAX", "MAXINDEX", "MIN", "MININDEX", "SUM",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups.insert(
        "Cycle Indicators".to_string(),
        vec![
            "HT_DCPERIOD",
            "HT_DCPHASE",
            "HT_PHASOR",
            "HT_SINE",
            "HT_TRENDMODE",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups.insert(
        "Pattern Recognition".to_string(),
        vec![
            "CDLDOJI",
            "CDLHAMMER",
            "CDLENGULFING",
            "CDL2CROWS",
            "CDL3BLACKCROWS",
            "CDL3INSIDE",
            "CDL3LINESTRIKE",
            "CDL3OUTSIDE",
            "CDL3STARSINSOUTH",
            "CDL3WHITESOLDIERS",
            "CDLABANDONEDBABY",
            "CDLADVANCEBLOCK",
            "CDLBELTHOLD",
            "CDLBREAKAWAY",
            "CDLCLOSINGMARUBOZU",
            "CDLCONCEALBABYSWALL",
            "CDLCOUNTERATTACK",
            "CDLDARKCLOUDCOVER",
            "CDLDOJISTAR",
            "CDLDRAGONFLYDOJI",
            "CDLEVENINGDOJISTAR",
            "CDLEVENINGSTAR",
            "CDLGAPSIDESIDEWHITE",
            "CDLGRAVESTONEDOJI",
            "CDLHANGINGMAN",
            "CDLHARAMI",
            "CDLHARAMICROSS",
            "CDLHIGHWAVE",
            "CDLHIKKAKE",
            "CDLHIKKAKEMOD",
            "CDLHOMINGPIGEON",
            "CDLIDENTICAL3CROWS",
            "CDLINNECK",
            "CDLINVERTEDHAMMER",
            "CDLKICKING",
            "CDLKICKINGBYLENGTH",
            "CDLLADDERBOTTOM",
            "CDLLONGLEGGEDDOJI",
            "CDLLONGLINE",
            "CDLMARUBOZU",
            "CDLMATCHINGLOW",
            "CDLMATHOLD",
            "CDLMORNINGDOJISTAR",
            "CDLMORNINGSTAR",
            "CDLONNECK",
            "CDLPIERCING",
            "CDLRICKSHAWMAN",
            "CDLRISEFALL3METHODS",
            "CDLSEPARATINGLINES",
            "CDLSHOOTINGSTAR",
            "CDLSHORTLINE",
            "CDLSPINNINGTOP",
            "CDLSTALLEDPATTERN",
            "CDLSTICKSANDWICH",
            "CDLTAKURI",
            "CDLTASUKIGAP",
            "CDLTHRUSTING",
            "CDLTRISTAR",
            "CDLUNIQUE3RIVER",
            "CDLUPSIDEGAP2CROWS",
            "CDLXSIDEGAP3METHODS",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    groups
}
