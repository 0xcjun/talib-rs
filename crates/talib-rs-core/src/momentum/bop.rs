use crate::error::{TaError, TaResult};

/// Balance Of Power (BOP)
///
/// BOP = (Close - Open) / (High - Low)
/// lookback = 0
pub fn bop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> TaResult<Vec<f64>> {
    let len = open.len();
    if len != high.len() || len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: high.len().min(low.len()).min(close.len()),
        });
    }

    Ok(open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .map(|(((&o, &h), &l), &c)| {
            let hl = h - l;
            if hl > 0.0 {
                (c - o) / hl
            } else {
                0.0
            }
        })
        .collect())
}
