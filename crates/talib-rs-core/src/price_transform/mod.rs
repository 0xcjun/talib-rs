use crate::error::{TaError, TaResult};

/// Average Price: (O + H + L + C) / 4
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = open.len();
    validate_ohlc_len(len, high, low, close)?;
    let mut output = vec![0.0; len];
    for i in 0..len {
        unsafe {
            *output.get_unchecked_mut(i) = (*open.get_unchecked(i)
                + *high.get_unchecked(i)
                + *low.get_unchecked(i)
                + *close.get_unchecked(i))
                / 4.0;
        }
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
    let mut output = vec![0.0; len];
    for i in 0..len {
        unsafe {
            *output.get_unchecked_mut(i) =
                (*high.get_unchecked(i) + *low.get_unchecked(i)) / 2.0;
        }
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
    let mut output = vec![0.0; len];
    for i in 0..len {
        unsafe {
            *output.get_unchecked_mut(i) = (*high.get_unchecked(i)
                + *low.get_unchecked(i)
                + *close.get_unchecked(i))
                / 3.0;
        }
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
    let mut output = vec![0.0; len];
    for i in 0..len {
        unsafe {
            *output.get_unchecked_mut(i) = (*high.get_unchecked(i)
                + *low.get_unchecked(i)
                + 2.0 * *close.get_unchecked(i))
                / 4.0;
        }
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
