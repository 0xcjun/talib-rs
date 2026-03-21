use crate::error::{TaError, TaResult};
use crate::volatility::true_range_array;

/// Directional Movement Index (DX)
///
/// DX = 100 * |+DI - -DI| / (+DI + -DI)
pub fn dx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let pdi = plus_di(high, low, close, timeperiod)?;
    let mdi = minus_di(high, low, close, timeperiod)?;
    let len = high.len();
    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if !pdi[i].is_nan() && !mdi[i].is_nan() {
            let sum = pdi[i] + mdi[i];
            output[i] = if sum > 0.0 {
                100.0 * (pdi[i] - mdi[i]).abs() / sum
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

/// Plus Directional Indicator (+DI)
pub fn plus_di(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if timeperiod < 1 || len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let tr = true_range_array(high, low, close);
    let mut pdm = vec![0.0; len];
    for i in 1..len {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        if up > down && up > 0.0 {
            pdm[i] = up;
        }
    }

    // Wilder 平滑
    let mut output = vec![f64::NAN; len];
    let mut sum_tr: f64 = tr[1..=timeperiod].iter().sum();
    let mut sum_pdm: f64 = pdm[1..=timeperiod].iter().sum();

    output[timeperiod] = if sum_tr > 0.0 {
        100.0 * sum_pdm / sum_tr
    } else {
        0.0
    };

    let pf = timeperiod as f64;
    for i in (timeperiod + 1)..len {
        sum_tr = sum_tr - sum_tr / pf + tr[i];
        sum_pdm = sum_pdm - sum_pdm / pf + pdm[i];
        output[i] = if sum_tr > 0.0 {
            100.0 * sum_pdm / sum_tr
        } else {
            0.0
        };
    }

    Ok(output)
}

/// Minus Directional Indicator (-DI)
pub fn minus_di(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if timeperiod < 1 || len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let tr = true_range_array(high, low, close);
    let mut mdm = vec![0.0; len];
    for i in 1..len {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        if down > up && down > 0.0 {
            mdm[i] = down;
        }
    }

    let mut output = vec![f64::NAN; len];
    let mut sum_tr: f64 = tr[1..=timeperiod].iter().sum();
    let mut sum_mdm: f64 = mdm[1..=timeperiod].iter().sum();

    output[timeperiod] = if sum_tr > 0.0 {
        100.0 * sum_mdm / sum_tr
    } else {
        0.0
    };

    let pf = timeperiod as f64;
    for i in (timeperiod + 1)..len {
        sum_tr = sum_tr - sum_tr / pf + tr[i];
        sum_mdm = sum_mdm - sum_mdm / pf + mdm[i];
        output[i] = if sum_tr > 0.0 {
            100.0 * sum_mdm / sum_tr
        } else {
            0.0
        };
    }

    Ok(output)
}

/// Plus Directional Movement (+DM)
pub fn plus_dm(
    high: &[f64],
    low: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let mut pdm = vec![0.0; len];
    for i in 1..len {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        if up > down && up > 0.0 {
            pdm[i] = up;
        }
    }

    let mut output = vec![f64::NAN; len];
    let mut sum: f64 = pdm[1..=timeperiod].iter().sum();
    output[timeperiod] = sum;

    let pf = timeperiod as f64;
    for i in (timeperiod + 1)..len {
        sum = sum - sum / pf + pdm[i];
        output[i] = sum;
    }

    Ok(output)
}

/// Minus Directional Movement (-DM)
pub fn minus_dm(
    high: &[f64],
    low: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let mut mdm = vec![0.0; len];
    for i in 1..len {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        if down > up && down > 0.0 {
            mdm[i] = down;
        }
    }

    let mut output = vec![f64::NAN; len];
    let mut sum: f64 = mdm[1..=timeperiod].iter().sum();
    output[timeperiod] = sum;

    let pf = timeperiod as f64;
    for i in (timeperiod + 1)..len {
        sum = sum - sum / pf + mdm[i];
        output[i] = sum;
    }

    Ok(output)
}
