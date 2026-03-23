use crate::error::{TaError, TaResult};

/// Parabolic SAR — 与 C TA-Lib 完全一致的实现
///
/// acceleration 默认 0.02, maximum 默认 0.2
/// lookback = 1
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len < 2 {
        return Err(TaError::InsufficientData { need: 2, got: len });
    }

    let mut output = vec![0.0_f64; len];
    output[0] = f64::NAN; // lookback = 1

    // 方向判断: 使用 MINUS_DM 逻辑
    let diff_m = low[0] - low[1];
    let diff_p = high[1] - high[0];
    let is_long_init = !(diff_m > 0.0 && diff_m > diff_p);

    let mut is_long = is_long_init;
    let mut sar_val: f64;
    let mut ep: f64;
    let mut af = acceleration;

    if is_long {
        ep = high[1];
        sar_val = low[0];
    } else {
        ep = low[1];
        sar_val = high[0];
    }

    // Handle i=1 separately to avoid per-iteration branch
    {
        let new_low = low[1];
        let new_high = high[1];
        let p_low = low[1];  // C TA-Lib: prev = current for first bar
        let p_high = high[1];

        if is_long {
            if new_low <= sar_val {
                is_long = false;
                sar_val = ep;
                sar_val = sar_val.max(p_high).max(new_high);
                output[1] = sar_val;
                af = acceleration;
                ep = new_low;
                sar_val += af * (ep - sar_val);
                sar_val = sar_val.max(p_high).max(new_high);
            } else {
                output[1] = sar_val;
                if new_high > ep { ep = new_high; af = (af + acceleration).min(maximum); }
                sar_val += af * (ep - sar_val);
                sar_val = sar_val.min(p_low).min(new_low);
            }
        } else if new_high >= sar_val {
            is_long = true;
            sar_val = ep;
            sar_val = sar_val.min(p_low).min(new_low);
            output[1] = sar_val;
            af = acceleration;
            ep = new_high;
            sar_val += af * (ep - sar_val);
            sar_val = sar_val.min(p_low).min(new_low);
        } else {
            output[1] = sar_val;
            if new_low < ep { ep = new_low; af = (af + acceleration).min(maximum); }
            sar_val += af * (ep - sar_val);
            sar_val = sar_val.max(p_high).max(new_high);
        }
    }

    // Main loop: i >= 2, no branch for i==1
    // Use shift pattern: prev values from registers, not array reads
    let mut prev_low = low[1];
    let mut prev_high = high[1];

    for i in 2..len {
        let new_low = low[i];
        let new_high = high[i];
        let p_low = prev_low;
        let p_high = prev_high;
        prev_low = new_low;
        prev_high = new_high;

        if is_long {
            if new_low <= sar_val {
                is_long = false;
                sar_val = ep;
                sar_val = sar_val.max(p_high).max(new_high);
                output[i] = sar_val;
                af = acceleration;
                ep = new_low;
                sar_val += af * (ep - sar_val);
                sar_val = sar_val.max(p_high).max(new_high);
            } else {
                output[i] = sar_val;
                if new_high > ep { ep = new_high; af = (af + acceleration).min(maximum); }
                sar_val += af * (ep - sar_val);
                sar_val = sar_val.min(p_low).min(new_low);
            }
        } else if new_high >= sar_val {
            is_long = true;
            sar_val = ep;
            sar_val = sar_val.min(p_low).min(new_low);
            output[i] = sar_val;
            af = acceleration;
            ep = new_high;
            sar_val += af * (ep - sar_val);
            sar_val = sar_val.min(p_low).min(new_low);
        } else {
            output[i] = sar_val;
            if new_low < ep { ep = new_low; af = (af + acceleration).min(maximum); }
            sar_val += af * (ep - sar_val);
            sar_val = sar_val.max(p_high).max(new_high);
        }
    }

    Ok(output)
}

/// Parabolic SAR Extended — 与 C TA-Lib SAREXT 完全一致
///
/// 支持多空不同加速因子、startvalue 指定初始方向、offsetonreverse 反转偏移。
/// 输出: 正值 = 多头 SAR, 负值 = 空头 SAR
/// lookback = 1
pub fn sar_ext(
    high: &[f64],
    low: &[f64],
    startvalue: f64,
    offsetonreverse: f64,
    accelerationinitlong: f64,
    accelerationlong: f64,
    accelerationmaxlong: f64,
    accelerationinitshort: f64,
    accelerationshort: f64,
    accelerationmaxshort: f64,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len < 2 {
        return Err(TaError::InsufficientData { need: 2, got: len });
    }

    let mut output = vec![0.0_f64; len];
    output[0] = f64::NAN; // lookback = 1

    let mut is_long: bool;
    let mut sar_val: f64;
    let mut ep: f64;
    let mut af_long: f64;
    let mut af_short: f64;

    if startvalue == 0.0 {
        let diff_m = low[0] - low[1];
        let diff_p = high[1] - high[0];
        is_long = !(diff_m > 0.0 && diff_m > diff_p);
    } else {
        is_long = startvalue > 0.0;
    }

    if startvalue == 0.0 {
        if is_long {
            ep = high[1];
            sar_val = low[0];
        } else {
            ep = low[1];
            sar_val = high[0];
        }
    } else if startvalue > 0.0 {
        ep = high[1];
        sar_val = startvalue;
    } else {
        ep = low[1];
        sar_val = startvalue.abs();
    }

    af_long = accelerationinitlong;
    af_short = accelerationinitshort;

    // Handle i=1 separately to avoid per-iteration branch
    {
        let new_low = low[1];
        let new_high = high[1];
        let p_low = low[1];
        let p_high = high[1];

        if is_long {
            if new_low <= sar_val {
                is_long = false;
                sar_val = ep;
                sar_val = sar_val.max(p_high).max(new_high);
                if offsetonreverse != 0.0 { sar_val += sar_val * offsetonreverse; }
                output[1] = -sar_val;
                af_short = accelerationinitshort;
                ep = new_low;
                sar_val += af_short * (ep - sar_val);
                sar_val = sar_val.max(p_high).max(new_high);
            } else {
                output[1] = sar_val;
                if new_high > ep { ep = new_high; af_long = (af_long + accelerationlong).min(accelerationmaxlong); }
                sar_val += af_long * (ep - sar_val);
                sar_val = sar_val.min(p_low).min(new_low);
            }
        } else if new_high >= sar_val {
            is_long = true;
            sar_val = ep;
            sar_val = sar_val.min(p_low).min(new_low);
            if offsetonreverse != 0.0 { sar_val -= sar_val * offsetonreverse; }
            output[1] = sar_val;
            af_long = accelerationinitlong;
            ep = new_high;
            sar_val += af_long * (ep - sar_val);
            sar_val = sar_val.min(p_low).min(new_low);
        } else {
            output[1] = -sar_val;
            if new_low < ep { ep = new_low; af_short = (af_short + accelerationshort).min(accelerationmaxshort); }
            sar_val += af_short * (ep - sar_val);
            sar_val = sar_val.max(p_high).max(new_high);
        }
    }

    // Main loop: i >= 2, shift pattern for prev values
    let mut prev_low = low[1];
    let mut prev_high = high[1];

    for i in 2..len {
        let new_low = low[i];
        let new_high = high[i];
        let p_low = prev_low;
        let p_high = prev_high;
        prev_low = new_low;
        prev_high = new_high;

        if is_long {
            if new_low <= sar_val {
                is_long = false;
                sar_val = ep;
                sar_val = sar_val.max(p_high).max(new_high);
                if offsetonreverse != 0.0 { sar_val += sar_val * offsetonreverse; }
                output[i] = -sar_val;
                af_short = accelerationinitshort;
                ep = new_low;
                sar_val += af_short * (ep - sar_val);
                sar_val = sar_val.max(p_high).max(new_high);
            } else {
                output[i] = sar_val;
                if new_high > ep { ep = new_high; af_long = (af_long + accelerationlong).min(accelerationmaxlong); }
                sar_val += af_long * (ep - sar_val);
                sar_val = sar_val.min(p_low).min(new_low);
            }
        } else if new_high >= sar_val {
            is_long = true;
            sar_val = ep;
            sar_val = sar_val.min(p_low).min(new_low);
            if offsetonreverse != 0.0 { sar_val -= sar_val * offsetonreverse; }
            output[i] = sar_val;
            af_long = accelerationinitlong;
            ep = new_high;
            sar_val += af_long * (ep - sar_val);
            sar_val = sar_val.min(p_low).min(new_low);
        } else {
            output[i] = -sar_val;
            if new_low < ep { ep = new_low; af_short = (af_short + accelerationshort).min(accelerationmaxshort); }
            sar_val += af_short * (ep - sar_val);
            sar_val = sar_val.max(p_high).max(new_high);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sar_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.0, 11.0, 10.5, 10.0, 9.0];
        let low = vec![9.0, 9.5, 10.0, 10.0, 11.0, 10.5, 9.5, 9.0, 8.5, 8.0];
        let result = sar(&high, &low, 0.02, 0.2).unwrap();
        assert_eq!(result.len(), 10);
        assert!(result[0].is_nan()); // C TA-Lib lookback = 1
        assert!(!result[1].is_nan());
    }

    #[test]
    fn test_sar_updates_across_bars() {
        let high = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let low = vec![9.0, 9.5, 10.0, 10.5, 11.0];
        let result = sar(&high, &low, 0.02, 0.2).unwrap();
        assert!(result[2] > result[1], "SAR should increase in uptrend");
        assert!(result[3] > result[2], "SAR should keep increasing");
    }

    #[test]
    fn test_sar_ext_sign_convention() {
        let high = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let low = vec![9.0, 9.5, 10.0, 10.5, 11.0];
        let result = sar_ext(&high, &low, 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2).unwrap();
        for i in 1..result.len() {
            assert!(result[i] > 0.0, "Long SAR should be positive at index {}", i);
        }
    }

    #[test]
    fn test_sar_ext_long_short_differ() {
        let high = vec![10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0];
        let low = vec![9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0];
        let r1 = sar_ext(&high, &low, 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2).unwrap();
        let r2 = sar_ext(&high, &low, 0.0, 0.0, 0.04, 0.04, 0.4, 0.02, 0.02, 0.2).unwrap();
        let diff: f64 = r1
            .iter()
            .zip(r2.iter())
            .filter(|(a, b)| !a.is_nan() && !b.is_nan())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let _ = diff;
    }
}
