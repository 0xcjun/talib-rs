use crate::error::{TaError, TaResult};

/// EMA core — matches C TA-Lib formulation for optimal critical path.
///
/// C TA-Lib: `prevMA = (inReal[today] - prevMA) * k + prevMA`
/// ARM critical path: FSUB(2cy) → FMADD(4cy) = 6 cycles/iter
/// vs old `x*k + prev*(1-k)`: FMUL(3cy) → FMADD(4cy) = 7 cycles/iter
pub fn ema_core(input: &[f64], period: usize, k: f64) -> TaResult<Vec<f64>> {
    let len = input.len();
    if period == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    if len < period {
        return Err(TaError::InsufficientData {
            need: period,
            got: len,
        });
    }

    let lookback = period - 1;

    // SMA seed
    let sma_seed: f64 = crate::simd::sum_f64(&input[..period]) / period as f64;

    // Pre-allocated output (serial dependency → indexed write faster than push)
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);
    output[lookback] = sma_seed;

    // EMA recursion: k*(x - prev) + prev — C TA-Lib formulation
    // Eliminates (1-k) multiplication, shorter critical path
    let mut prev = sma_seed;
    for i in period..len {
        let val = k.mul_add(input[i] - prev, prev);
        output[i] = val;
        prev = val;
    }

    Ok(output)
}

/// Exponential Moving Average (EMA)
///
/// k = 2.0 / (timeperiod + 1), lookback = timeperiod - 1
pub fn ema(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let k = 2.0 / (timeperiod as f64 + 1.0);
    ema_core(input, timeperiod, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema(&input, 3).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_period_1() {
        let input = vec![1.0, 2.0, 3.0];
        let result = ema(&input, 1).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }
}
