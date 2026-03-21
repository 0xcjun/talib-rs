use crate::error::{TaError, TaResult};
use crate::overlap::ema::ema_core;

/// Triple Exponential Moving Average (TEMA)
///
/// TEMA = 3*EMA1 - 3*EMA2 + EMA3
/// 其中 EMA1 = EMA(input), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
/// lookback = 3 * (timeperiod - 1)
pub fn tema(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2 for TEMA",
        });
    }
    let len = input.len();
    let lookback = 3 * (timeperiod - 1);
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let k = 2.0 / (timeperiod as f64 + 1.0);

    // EMA1
    let ema1 = ema_core(input, timeperiod, k)?;
    let ema1_valid: Vec<f64> = ema1.iter().copied().filter(|v| !v.is_nan()).collect();

    // EMA2
    let ema2_valid = ema_core(&ema1_valid, timeperiod, k)?;
    let ema2_clean: Vec<f64> = ema2_valid.iter().copied().filter(|v| !v.is_nan()).collect();

    // EMA3
    let ema3_clean = ema_core(&ema2_clean, timeperiod, k)?;

    // 构建输出
    let mut output = vec![f64::NAN; len];
    let e2_offset = timeperiod - 1;
    let e3_offset = timeperiod - 1;

    for i in lookback..len {
        let e1 = ema1[i];
        let idx2 = i - (timeperiod - 1);
        let idx3_base = idx2.checked_sub(timeperiod - 1);
        if let Some(idx3_pos) = idx3_base {
            if idx2 < ema2_valid.len() && idx3_pos < ema3_clean.len() {
                let e2 = ema2_valid[idx2];
                let e3 = ema3_clean[idx3_pos];
                if !e2.is_nan() && !e3.is_nan() {
                    output[i] = 3.0 * e1 - 3.0 * e2 + e3;
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tema_basic() {
        let input: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = tema(&input, 5).unwrap();
        // lookback = 3*(5-1) = 12
        assert!(result[11].is_nan());
        assert!(!result[12].is_nan());
    }
}
