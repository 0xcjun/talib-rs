use crate::error::{TaError, TaResult};

/// Kaufman Adaptive Moving Average (KAMA) — O(n) with slice-based bounds elision
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

    let mut prev_kama: f64 = input[timeperiod - 1];

    // Initialize volatility
    let mut volatility = 0.0_f64;
    for (&cur, &prev) in input[1..=timeperiod].iter().zip(input[..timeperiod].iter()) {
        volatility += (cur - prev).abs();
    }

    // First output
    {
        let direction = (input[lookback] - input[0]).abs();
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc_raw = er * sc_diff + slow_sc;
        let sc = sc_raw * sc_raw;
        prev_kama += sc * (input[lookback] - prev_kama);
        output[lookback] = prev_kama;
    }

    // Hot loop: 4 pre-sliced views → zero bounds checks
    // cur   = input[i]       where i in (lookback+1)..len
    // prev  = input[i-1]
    // far   = input[i-tp]
    // farp  = input[i-tp-1]
    let count = len - lookback - 1;
    let cur = &input[(lookback + 1)..];
    let prev = &input[lookback..len - 1];
    let far = &input[1..len - timeperiod];
    let farp = &input[..len - timeperiod - 1];

    for ((((&c, &p), &f), &fp), o) in cur[..count]
        .iter()
        .zip(prev[..count].iter())
        .zip(far[..count].iter())
        .zip(farp[..count].iter())
        .zip(output[(lookback + 1)..].iter_mut())
    {
        volatility += (c - p).abs() - (f - fp).abs();
        let direction = (c - f).abs();
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc_raw = er * sc_diff + slow_sc;
        let sc = sc_raw * sc_raw;
        prev_kama += sc * (c - prev_kama);
        *o = prev_kama;
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
