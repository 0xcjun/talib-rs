use crate::error::{TaError, TaResult};
use crate::overlap::ema::ema_core;

/// TRIX — 三重指数平滑的 ROC
///
/// TRIX = ROC(EMA(EMA(EMA(input))))
/// lookback = 3*(timeperiod-1) + 1
pub fn trix(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    let lookback = 3 * (timeperiod - 1) + 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let k = 2.0 / (timeperiod as f64 + 1.0);
    let e1 = ema_core(input, timeperiod, k)?;
    let e1v: Vec<f64> = e1.iter().copied().filter(|v: &f64| !v.is_nan()).collect();
    let e2 = ema_core(&e1v, timeperiod, k)?;
    let e2v: Vec<f64> = e2.iter().copied().filter(|v: &f64| !v.is_nan()).collect();
    let e3 = ema_core(&e2v, timeperiod, k)?;

    let mut output = vec![f64::NAN; len];
    let start = 3 * (timeperiod - 1);

    // TRIX = 1-day ROC of triple EMA
    let mut prev_e3 = f64::NAN;
    let mut e3_idx = 0;
    for i in start..len {
        let idx = i - 2 * (timeperiod - 1);
        let local_idx = idx - (timeperiod - 1);
        if local_idx < e3.len() && !e3[local_idx].is_nan() {
            if !prev_e3.is_nan() && prev_e3 != 0.0 {
                output[i] = ((e3[local_idx] - prev_e3) / prev_e3) * 100.0;
            }
            prev_e3 = e3[local_idx];
        }
    }

    Ok(output)
}
