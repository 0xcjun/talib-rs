use crate::error::{TaError, TaResult};

/// Kaufman Adaptive Moving Average (KAMA) — O(n) with pre-computed deltas
///
/// Pre-computes |delta| array in a vectorizable pass, then rolling sum
/// on a single compact array reduces cache pressure and abs() calls.
pub fn kama(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    let lookback = timeperiod;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    let fast_sc = 2.0 / (2.0 + 1.0);
    let slow_sc = 2.0 / (30.0 + 1.0);
    let sc_diff = fast_sc - slow_sc;

    // Phase 1: pre-compute |delta[i]| = |input[i] - input[i-1]| for i >= 1
    // This vectorizes well and removes abs() from the hot loop
    let mut abs_delta = vec![0.0_f64; len];
    for ((&cur, &prev), out) in input[1..]
        .iter()
        .zip(input[..len - 1].iter())
        .zip(abs_delta[1..].iter_mut())
    {
        *out = (cur - prev).abs();
    }

    // Phase 2: rolling sum on abs_delta[] — single array, cache-friendly
    let mut prev_kama: f64 = input[timeperiod - 1];

    let mut volatility: f64 = abs_delta[1..=timeperiod].iter().sum();

    // First output
    {
        let direction = (input[lookback] - input[0]).abs();
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc_raw = er * sc_diff + slow_sc;
        let sc = sc_raw * sc_raw;
        let kama_val = prev_kama + sc * (input[lookback] - prev_kama);
        output[lookback] = kama_val;
        prev_kama = kama_val;
    }

    // Remaining outputs — hot loop accesses: abs_delta[i], abs_delta[i-tp], input[i], input[i-tp]
    for i in (lookback + 1)..len {
        volatility += abs_delta[i] - abs_delta[i - timeperiod];

        let direction = (input[i] - input[i - timeperiod]).abs();
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc_raw = er * sc_diff + slow_sc;
        let sc = sc_raw * sc_raw;
        let kama_val = prev_kama + sc * (input[i] - prev_kama);
        output[i] = kama_val;
        prev_kama = kama_val;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kama_basic() {
        let input: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = kama(&input, 10).unwrap();
        assert!(result[9].is_nan());
        assert!(!result[10].is_nan());
    }
}
