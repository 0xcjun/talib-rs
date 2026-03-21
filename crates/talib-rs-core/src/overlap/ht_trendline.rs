use crate::error::{TaError, TaResult};

/// Hilbert Transform - Instantaneous Trendline
///
/// 基于 Hilbert Transform 的瞬时趋势线。
/// lookback = 63 (TA-Lib 兼容)
pub fn ht_trendline(input: &[f64]) -> TaResult<Vec<f64>> {
    let len = input.len();
    let lookback = 63;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];

    let mut smooth = vec![0.0; len];
    let mut detrender = vec![0.0; len];
    let mut q1 = vec![0.0; len];
    let mut i1 = vec![0.0; len];
    let mut ji = vec![0.0; len];
    let mut jq = vec![0.0; len];
    let mut i2 = vec![0.0; len];
    let mut q2 = vec![0.0; len];
    let mut re = vec![0.0; len];
    let mut im = vec![0.0; len];
    let mut period = vec![0.0; len];
    let mut smooth_period = vec![0.0; len];
    let mut it_period: Vec<i32> = vec![0; len];
    let mut dc_phase = vec![0.0; len];

    let a = 0.0962;
    let b = 0.5769;

    for i in 0..len {
        if i >= 3 {
            smooth[i] =
                (4.0 * input[i] + 3.0 * input[i - 1] + 2.0 * input[i - 2] + input[i - 3]) / 10.0;
        }

        if i >= 9 {
            let adj = 0.075 * period[i.saturating_sub(1)] + 0.54;

            detrender[i] = (a * smooth[i] + b * smooth[i.saturating_sub(2)]
                - b * smooth[i.saturating_sub(4)]
                - a * smooth[i.saturating_sub(6)])
                * adj;

            q1[i] = (a * detrender[i] + b * detrender[i.saturating_sub(2)]
                - b * detrender[i.saturating_sub(4)]
                - a * detrender[i.saturating_sub(6)])
                * adj;
            i1[i] = detrender[i.saturating_sub(3)];

            ji[i] = (a * i1[i] + b * i1[i.saturating_sub(2)]
                - b * i1[i.saturating_sub(4)]
                - a * i1[i.saturating_sub(6)])
                * adj;
            jq[i] = (a * q1[i] + b * q1[i.saturating_sub(2)]
                - b * q1[i.saturating_sub(4)]
                - a * q1[i.saturating_sub(6)])
                * adj;

            i2[i] = i1[i] - jq[i];
            q2[i] = q1[i] + ji[i];

            i2[i] = 0.2 * i2[i] + 0.8 * i2[i.saturating_sub(1)];
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i.saturating_sub(1)];

            re[i] = i2[i] * i2[i.saturating_sub(1)] + q2[i] * q2[i.saturating_sub(1)];
            im[i] = i2[i] * q2[i.saturating_sub(1)] - q2[i] * i2[i.saturating_sub(1)];

            re[i] = 0.2 * re[i] + 0.8 * re[i.saturating_sub(1)];
            im[i] = 0.2 * im[i] + 0.8 * im[i.saturating_sub(1)];

            if im[i] != 0.0 && re[i] != 0.0 {
                period[i] = 2.0 * std::f64::consts::PI / im[i].atan2(re[i]);
            }
            if period[i] > 1.5 * period[i.saturating_sub(1)] {
                period[i] = 1.5 * period[i.saturating_sub(1)];
            }
            if period[i] < 0.67 * period[i.saturating_sub(1)] {
                period[i] = 0.67 * period[i.saturating_sub(1)];
            }
            if period[i] < 6.0 {
                period[i] = 6.0;
            }
            if period[i] > 50.0 {
                period[i] = 50.0;
            }

            period[i] = 0.2 * period[i] + 0.8 * period[i.saturating_sub(1)];
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i.saturating_sub(1)];

            let dc_period = (smooth_period[i] + 0.5) as i32;
            it_period[i] = dc_period;
        }

        if i >= lookback {
            let dc_period = it_period[i].max(1) as usize;
            let lookback_count = dc_period.min(i);
            let mut sum = 0.0;
            for j in 0..lookback_count {
                sum += input[i - j];
            }
            output[i] = if lookback_count > 0 {
                sum / lookback_count as f64
            } else {
                input[i]
            };
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ht_trendline_basic() {
        let input: Vec<f64> = (0..200)
            .map(|i| 50.0 + 10.0 * (i as f64 * 0.2).sin())
            .collect();
        let result = ht_trendline(&input).unwrap();
        assert!(result[62].is_nan());
        assert!(!result[63].is_nan());
    }
}
