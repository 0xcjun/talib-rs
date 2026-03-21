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

    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        let range = high[i] - low[i];
        output[i] = if range > 0.0 {
            (close[i] - open[i]) / range
        } else {
            0.0
        };
    }

    Ok(output)
}
