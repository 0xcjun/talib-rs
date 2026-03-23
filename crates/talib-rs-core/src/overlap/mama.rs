use crate::error::{TaError, TaResult};

/// MESA Adaptive Moving Average (MAMA)
///
/// 基于 Hilbert Transform 的自适应移动平均。
/// 返回 (mama, fama) 两条线。
/// fastlimit 默认 0.5, slowlimit 默认 0.05
/// lookback = 32 (Hilbert Transform 需要的初始周期)
///
/// Optimized: uses fixed-size ring buffers instead of 13 heap-allocated Vecs.
pub fn mama(input: &[f64], fastlimit: f64, slowlimit: f64) -> TaResult<(Vec<f64>, Vec<f64>)> {
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

    let mut mama_out = vec![0.0_f64; len];
    mama_out[..lookback].fill(f64::NAN);
    let mut fama_out = vec![0.0_f64; len];
    fama_out[..lookback].fill(f64::NAN);

    // Ring buffers for Hilbert Transform state (need up to 7 history: [i], [i-1]...[i-6])
    // Index into ring: i % 7
    let mut smooth_buf = [0.0_f64; 7];
    let mut detrender_buf = [0.0_f64; 7];
    let mut q1_buf = [0.0_f64; 7];
    let mut i1_buf = [0.0_f64; 7];
    let mut ji_buf = [0.0_f64; 7];
    let mut jq_buf = [0.0_f64; 7];

    // Only need current and previous for these
    let mut i2_prev = 0.0_f64;
    let mut q2_prev = 0.0_f64;
    let mut re_prev = 0.0_f64;
    let mut im_prev = 0.0_f64;
    let mut period_prev = 0.0_f64;
    let mut smooth_period_prev = 0.0_f64;
    let mut phase_prev = 0.0_f64;

    let mut prev_mama = 0.0_f64;
    let mut prev_fama = 0.0_f64;

    let a = 0.0962;
    let b = 0.5769;

    for i in 0..len {
        let ri = i % 7;

        if i >= 3 {
            smooth_buf[ri] = (4.0 * input[i]
                + 3.0 * input[i - 1]
                + 2.0 * input[i - 2]
                + input[i - 3])
                / 10.0;
        }

        if i >= 9 {
            // indices into ring buffer for historical values
            let r0 = ri;                      // i
            let r2 = (i.wrapping_sub(2)) % 7; // i-2
            let r4 = (i.wrapping_sub(4)) % 7; // i-4
            let r6 = (i.wrapping_sub(6)) % 7; // i-6
            let r3 = (i.wrapping_sub(3)) % 7; // i-3

            let adj = 0.075 * period_prev + 0.54;

            // Detrender
            detrender_buf[r0] = (a * smooth_buf[r0] + b * smooth_buf[r2]
                - b * smooth_buf[r4]
                - a * smooth_buf[r6])
                * adj;

            // Q1
            q1_buf[r0] = (a * detrender_buf[r0] + b * detrender_buf[r2]
                - b * detrender_buf[r4]
                - a * detrender_buf[r6])
                * adj;

            // I1
            i1_buf[r0] = detrender_buf[r3];

            // JI
            ji_buf[r0] = (a * i1_buf[r0] + b * i1_buf[r2]
                - b * i1_buf[r4]
                - a * i1_buf[r6])
                * adj;

            // JQ
            jq_buf[r0] = (a * q1_buf[r0] + b * q1_buf[r2]
                - b * q1_buf[r4]
                - a * q1_buf[r6])
                * adj;

            // Phasor addition
            let mut i2_val = i1_buf[r0] - jq_buf[r0];
            let mut q2_val = q1_buf[r0] + ji_buf[r0];

            // Smooth I2 and Q2
            i2_val = 0.2 * i2_val + 0.8 * i2_prev;
            q2_val = 0.2 * q2_val + 0.8 * q2_prev;

            // Homodyne discriminator
            let mut re_val = i2_val * i2_prev + q2_val * q2_prev;
            let mut im_val = i2_val * q2_prev - q2_val * i2_prev;

            re_val = 0.2 * re_val + 0.8 * re_prev;
            im_val = 0.2 * im_val + 0.8 * im_prev;

            let mut period_val = if im_val != 0.0 && re_val != 0.0 {
                2.0 * std::f64::consts::PI / im_val.atan2(re_val)
            } else {
                0.0
            };

            if period_val > 1.5 * period_prev {
                period_val = 1.5 * period_prev;
            }
            if period_val < 0.67 * period_prev {
                period_val = 0.67 * period_prev;
            }
            if period_val < 6.0 {
                period_val = 6.0;
            }
            if period_val > 50.0 {
                period_val = 50.0;
            }

            period_val = 0.2 * period_val + 0.8 * period_prev;
            let smooth_period_val = 0.33 * period_val + 0.67 * smooth_period_prev;

            // Phase
            let phase_val = if i1_buf[r0] != 0.0 {
                (q1_buf[r0] / i1_buf[r0]).atan().to_degrees()
            } else {
                0.0
            };

            let delta_phase = phase_prev - phase_val;
            let delta_phase = if delta_phase < 1.0 { 1.0 } else { delta_phase };

            let alpha = fastlimit / delta_phase;
            let alpha = if alpha < slowlimit { slowlimit } else { alpha };
            let alpha = if alpha > fastlimit { fastlimit } else { alpha };

            if i >= lookback {
                if prev_mama == 0.0 {
                    prev_mama = input[i];
                    prev_fama = prev_mama;
                }
                let inp_i = input[i];
                prev_mama = alpha * inp_i + (1.0 - alpha) * prev_mama;
                prev_fama = 0.5 * alpha * prev_mama + (1.0 - 0.5 * alpha) * prev_fama;

                mama_out[i] = prev_mama;
                fama_out[i] = prev_fama;
            }

            // Update state for next iteration
            i2_prev = i2_val;
            q2_prev = q2_val;
            re_prev = re_val;
            im_prev = im_val;
            period_prev = period_val;
            smooth_period_prev = smooth_period_val;
            phase_prev = phase_val;
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
