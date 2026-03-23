use crate::cycle::{do_hilbert_even, do_hilbert_odd, HilbertVars, WmaState};
use crate::error::{TaError, TaResult};

const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;

/// MESA Adaptive Moving Average (MAMA)
///
/// Faithful port of C TA-Lib TA_MAMA using the same even/odd alternating
/// Hilbert Transform approach as the cycle indicators (HT_DCPERIOD etc.).
///
/// Returns (mama, fama).
/// fastlimit default 0.5, slowlimit default 0.05.
/// lookback = 32
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

    let start_idx = lookback;

    let mut mama_out = vec![0.0_f64; len];
    mama_out[..lookback].fill(f64::NAN);
    let mut fama_out = vec![0.0_f64; len];
    fama_out[..lookback].fill(f64::NAN);

    // --- WMA initialization (identical to HT_DCPERIOD) ---
    let trailing_wma_start = start_idx - lookback; // = 0
    let (mut wma, mut today) = WmaState::init(input, trailing_wma_start);

    // Warm up WMA: 9 iterations (matching C TA-Lib)
    for _ in 0..9 {
        let val = input[today];
        today += 1;
        let _ = wma.next(input, val);
    }

    // --- Hilbert Transform state ---
    let mut hilbert_idx: usize = 0;
    let mut detrender_vars = HilbertVars::new();
    let mut q1_vars = HilbertVars::new();
    let mut ji_vars = HilbertVars::new();
    let mut jq_vars = HilbertVars::new();

    let mut period: f64 = 0.0;
    let mut smooth_period: f64 = 0.0;

    let mut prev_i2: f64 = 0.0;
    let mut prev_q2: f64 = 0.0;
    let mut re: f64 = 0.0;
    let mut im: f64 = 0.0;

    let mut i1_for_odd_prev2: f64 = 0.0;
    let mut i1_for_odd_prev3: f64 = 0.0;
    let mut i1_for_even_prev2: f64 = 0.0;
    let mut i1_for_even_prev3: f64 = 0.0;

    // --- MAMA-specific state ---
    let mut prev_phase: f64 = 0.0;
    let mut prev_mama = 0.0_f64;
    let mut prev_fama = 0.0_f64;

    let mut out_idx = start_idx;

    while today < len {
        let adjusted_prev_period = 0.075 * period + 0.54;

        let today_value = input[today];
        let smoothed_value = wma.next(input, today_value);

        let (detrender, q1, i1, i2, q2);

        if today % 2 == 0 {
            // Even bar
            detrender = do_hilbert_even(
                &mut detrender_vars,
                smoothed_value,
                hilbert_idx,
                adjusted_prev_period,
            );
            q1 = do_hilbert_even(
                &mut q1_vars,
                detrender,
                hilbert_idx,
                adjusted_prev_period,
            );
            let _ji = do_hilbert_even(
                &mut ji_vars,
                i1_for_even_prev3,
                hilbert_idx,
                adjusted_prev_period,
            );
            let _jq = do_hilbert_even(
                &mut jq_vars,
                q1,
                hilbert_idx,
                adjusted_prev_period,
            );
            hilbert_idx += 1;
            if hilbert_idx == 3 {
                hilbert_idx = 0;
            }

            q2 = 0.2 * (q1 + _ji) + 0.8 * prev_q2;
            i2 = 0.2 * (i1_for_even_prev3 - _jq) + 0.8 * prev_i2;

            // I1 for MAMA phase = delayed detrender (i1_for_even_prev3)
            i1 = i1_for_even_prev3;

            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = detrender;
        } else {
            // Odd bar
            detrender = do_hilbert_odd(
                &mut detrender_vars,
                smoothed_value,
                hilbert_idx,
                adjusted_prev_period,
            );
            q1 = do_hilbert_odd(
                &mut q1_vars,
                detrender,
                hilbert_idx,
                adjusted_prev_period,
            );
            let _ji = do_hilbert_odd(
                &mut ji_vars,
                i1_for_odd_prev3,
                hilbert_idx,
                adjusted_prev_period,
            );
            let _jq = do_hilbert_odd(
                &mut jq_vars,
                q1,
                hilbert_idx,
                adjusted_prev_period,
            );

            q2 = 0.2 * (q1 + _ji) + 0.8 * prev_q2;
            i2 = 0.2 * (i1_for_odd_prev3 - _jq) + 0.8 * prev_i2;

            // I1 for MAMA phase = delayed detrender (i1_for_odd_prev3)
            i1 = i1_for_odd_prev3;

            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = detrender;
        }

        // --- Period estimation (identical to HT_DCPERIOD) ---
        re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re;
        im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im;
        prev_q2 = q2;
        prev_i2 = i2;

        let temp_real = period;
        if im != 0.0 && re != 0.0 {
            period = 360.0 / ((im / re).atan() * RAD2DEG);
        }
        let temp_real2 = 1.5 * temp_real;
        if period > temp_real2 {
            period = temp_real2;
        }
        let temp_real2 = 0.67 * temp_real;
        if period < temp_real2 {
            period = temp_real2;
        }
        if period < 6.0 {
            period = 6.0;
        } else if period > 50.0 {
            period = 50.0;
        }
        period = 0.2 * period + 0.8 * temp_real;

        smooth_period = 0.33 * period + 0.67 * smooth_period;

        // --- MAMA-specific: phase and adaptive alpha ---
        let phase = if i1 != 0.0 {
            (q1 / i1).atan() * RAD2DEG
        } else {
            0.0
        };

        let mut delta_phase = prev_phase - phase;
        if delta_phase < 1.0 {
            delta_phase = 1.0;
        }

        let mut alpha = fastlimit / delta_phase;
        if alpha < slowlimit {
            alpha = slowlimit;
        }
        if alpha > fastlimit {
            alpha = fastlimit;
        }

        prev_mama = alpha * today_value + (1.0 - alpha) * prev_mama;
        prev_fama = 0.5 * alpha * prev_mama + (1.0 - 0.5 * alpha) * prev_fama;

        if today >= start_idx {
            mama_out[out_idx] = prev_mama;
            fama_out[out_idx] = prev_fama;
            out_idx += 1;
        }

        prev_phase = phase;
        today += 1;
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

    #[test]
    fn test_mama_insufficient_data() {
        let input = vec![1.0; 32];
        assert!(mama(&input, 0.5, 0.05).is_err());
    }

    #[test]
    fn test_mama_invalid_params() {
        let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(mama(&input, 0.0, 0.05).is_err());
        assert!(mama(&input, 0.5, 0.5).is_err());
        assert!(mama(&input, 0.5, 0.6).is_err());
        assert!(mama(&input, 0.5, 0.0).is_err());
    }
}
