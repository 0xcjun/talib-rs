use crate::error::{TaError, TaResult};

/// MESA Adaptive Moving Average (MAMA)
///
/// 基于 Hilbert Transform 的自适应移动平均。
/// 返回 (mama, fama) 两条线。
/// fastlimit 默认 0.5, slowlimit 默认 0.05
/// lookback = 32 (Hilbert Transform 需要的初始周期)
pub fn mama(
    input: &[f64],
    fastlimit: f64,
    slowlimit: f64,
) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = input.len();
    let lookback = 32;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }
    if fastlimit <= 0.0 || fastlimit > 1.0 {
        return Err(TaError::InvalidParameter {
            name: "fastlimit",
            value: fastlimit.to_string(),
            reason: "must be in (0, 1]",
        });
    }
    if slowlimit <= 0.0 || slowlimit >= fastlimit {
        return Err(TaError::InvalidParameter {
            name: "slowlimit",
            value: slowlimit.to_string(),
            reason: "must be in (0, fastlimit)",
        });
    }

    let mut mama_out = vec![f64::NAN; len];
    let mut fama_out = vec![f64::NAN; len];

    // Hilbert Transform 状态变量
    let mut smooth = vec![0.0_f64; len];
    let mut detrender = vec![0.0_f64; len];
    let mut q1 = vec![0.0_f64; len];
    let mut i1 = vec![0.0_f64; len];
    let mut ji = vec![0.0_f64; len];
    let mut jq = vec![0.0_f64; len];
    let mut i2 = vec![0.0_f64; len];
    let mut q2 = vec![0.0_f64; len];
    let mut re = vec![0.0_f64; len];
    let mut im = vec![0.0_f64; len];
    let mut period = vec![0.0_f64; len];
    let mut smooth_period = vec![0.0_f64; len];
    let mut phase = vec![0.0_f64; len];

    let mut prev_mama = 0.0;
    let mut prev_fama = 0.0;

    let a = 0.0962;
    let b = 0.5769;

    for i in 0..len {
        if i >= 3 {
            smooth[i] = (4.0 * input[i] + 3.0 * input[i - 1]
                + 2.0 * input[i - 2] + input[i - 3]) / 10.0;
        }

        if i >= 6 + 3 {
            // Hilbert Transform
            detrender[i] = (a * smooth[i] + b * smooth[i.saturating_sub(2)]
                - b * smooth[i.saturating_sub(4)] - a * smooth[i.saturating_sub(6)])
                * (0.075 * period[i.saturating_sub(1)] + 0.54);

            // 计算 InPhase 和 Quadrature
            q1[i] = (a * detrender[i] + b * detrender[i.saturating_sub(2)]
                - b * detrender[i.saturating_sub(4)] - a * detrender[i.saturating_sub(6)])
                * (0.075 * period[i.saturating_sub(1)] + 0.54);
            i1[i] = detrender[i.saturating_sub(3)];

            // Advance phase by 90 degrees
            ji[i] = (a * i1[i] + b * i1[i.saturating_sub(2)]
                - b * i1[i.saturating_sub(4)] - a * i1[i.saturating_sub(6)])
                * (0.075 * period[i.saturating_sub(1)] + 0.54);
            jq[i] = (a * q1[i] + b * q1[i.saturating_sub(2)]
                - b * q1[i.saturating_sub(4)] - a * q1[i.saturating_sub(6)])
                * (0.075 * period[i.saturating_sub(1)] + 0.54);

            // Phasor addition
            i2[i] = i1[i] - jq[i];
            q2[i] = q1[i] + ji[i];

            // Smooth I2 and Q2
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i.saturating_sub(1)];
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i.saturating_sub(1)];

            // Homodyne discriminator
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

            // Phase
            if i1[i] != 0.0 {
                phase[i] = (q1[i] / i1[i]).atan().to_degrees();
            }

            let delta_phase = phase[i.saturating_sub(1)] - phase[i];
            let delta_phase = if delta_phase < 1.0 { 1.0 } else { delta_phase };

            let alpha = fastlimit / delta_phase;
            let alpha = if alpha < slowlimit { slowlimit } else { alpha };
            let alpha = if alpha > fastlimit { fastlimit } else { alpha };

            if i >= lookback {
                if prev_mama == 0.0 {
                    prev_mama = input[i];
                    prev_fama = input[i];
                }
                prev_mama = alpha * input[i] + (1.0 - alpha) * prev_mama;
                prev_fama = 0.5 * alpha * prev_mama + (1.0 - 0.5 * alpha) * prev_fama;

                mama_out[i] = prev_mama;
                fama_out[i] = prev_fama;
            }
        }
    }

    Ok((mama_out, fama_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mama_basic() {
        let input: Vec<f64> = (0..100)
            .map(|i| 50.0 + 10.0 * (i as f64 * 0.3).sin())
            .collect();
        let (mama, fama) = mama(&input, 0.5, 0.05).unwrap();
        // lookback = 32
        assert!(mama[31].is_nan());
        assert!(!mama[32].is_nan());
        assert!(!fama[32].is_nan());
    }
}
