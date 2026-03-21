use crate::error::{TaError, TaResult};

/// Average Price: (O + H + L + C) / 4
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = open.len();
    validate_ohlc_len(len, high, low, close)?;
    Ok((0..len)
        .map(|i| (open[i] + high[i] + low[i] + close[i]) / 4.0)
        .collect())
}

/// Median Price: (H + L) / 2
pub fn medprice(high: &[f64], low: &[f64]) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    Ok((0..len).map(|i| (high[i] + low[i]) / 2.0).collect())
}

/// Typical Price: (H + L + C) / 3
pub fn typprice(high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    Ok((0..len)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect())
}

/// Weighted Close Price: (H + L + 2*C) / 4
pub fn wclprice(high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    Ok((0..len)
        .map(|i| (high[i] + low[i] + 2.0 * close[i]) / 4.0)
        .collect())
}

fn validate_ohlc_len(len: usize, high: &[f64], low: &[f64], close: &[f64]) -> TaResult<()> {
    if len != high.len() || len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: high.len().min(low.len()).min(close.len()),
        });
    }
    Ok(())
}
