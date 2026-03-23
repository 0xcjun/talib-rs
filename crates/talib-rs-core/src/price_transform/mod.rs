use crate::error::{TaError, TaResult};

/// Average Price: (O + H + L + C) / 4
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = open.len();
    validate_ohlc_len(len, high, low, close)?;
    let mut output = vec![0.0_f64; len];
    for ((((&o, &h), &l), &c), out) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .zip(output.iter_mut())
    {
        *out = (o + h + l + c) * 0.25;
    }
    Ok(output)
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
    let mut output = vec![0.0_f64; len];
    for ((&h, &l), out) in high.iter().zip(low.iter()).zip(output.iter_mut()) {
        *out = (h + l) * 0.5;
    }
    Ok(output)
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
    let one_third = 1.0 / 3.0;
    let mut output = vec![0.0_f64; len];
    for (((&h, &l), &c), out) in high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(output.iter_mut())
    {
        *out = (h + l + c) * one_third;
    }
    Ok(output)
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
    let mut output = vec![0.0_f64; len];
    for (((&h, &l), &c), out) in high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(output.iter_mut())
    {
        *out = (h + l + c + c) * 0.25;
    }
    Ok(output)
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
