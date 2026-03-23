use crate::error::{TaError, TaResult};

/// Balance Of Power (BOP) — SIMD accelerated
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

    let mut output = vec![0.0_f64; len];
    crate::simd::bop_simd(open, high, low, close, &mut output);

    Ok(output)
}
