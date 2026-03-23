use crate::error::{TaError, TaResult};

// Faithful port of C TA-Lib HT_TRENDLINE.
// Uses the same even/odd alternating Hilbert Transform as cycle/mod.rs.
// The trendline = WMA(4) of SMA(price, dcPeriod), where dcPeriod is the
// instantaneous dominant cycle period from the Hilbert Transform.
// lookback = 63

const A: f64 = 0.0962;
const B: f64 = 0.5769;

struct HilbertVars {
    odd: [f64; 3],
    even: [f64; 3],
    prev_odd: f64,
    prev_even: f64,
    prev_input_odd: f64,
    prev_input_even: f64,
}

impl HilbertVars {
    fn new() -> Self {
        Self {
            odd: [0.0; 3],
            even: [0.0; 3],
            prev_odd: 0.0,
            prev_even: 0.0,
            prev_input_odd: 0.0,
            prev_input_even: 0.0,
        }
    }
}

#[inline(always)]
fn do_hilbert_even(
    vars: &mut HilbertVars,
    input: f64,
    hilbert_idx: usize,
    adjusted_prev_period: f64,
) -> f64 {
    let hilbert_temp_real = A * input;
    let mut result = -vars.even[hilbert_idx];
    vars.even[hilbert_idx] = hilbert_temp_real;
    result += hilbert_temp_real;
    result -= vars.prev_even;
    vars.prev_even = B * vars.prev_input_even;
    result += vars.prev_even;
    vars.prev_input_even = input;
    result *= adjusted_prev_period;
    result
}

#[inline(always)]
fn do_hilbert_odd(
    vars: &mut HilbertVars,
    input: f64,
    hilbert_idx: usize,
    adjusted_prev_period: f64,
) -> f64 {
    let hilbert_temp_real = A * input;
    let mut result = -vars.odd[hilbert_idx];
    vars.odd[hilbert_idx] = hilbert_temp_real;
    result += hilbert_temp_real;
    result -= vars.prev_odd;
    vars.prev_odd = B * vars.prev_input_odd;
    result += vars.prev_odd;
    vars.prev_input_odd = input;
    result *= adjusted_prev_period;
    result
}

struct WmaState {
    period_wma_sub: f64,
    period_wma_sum: f64,
    trailing_wma_value: f64,
    trailing_wma_idx: usize,
}

impl WmaState {
    fn init(input: &[f64], start: usize) -> (Self, usize) {
        let p0 = input[start];
        let p1 = input[start + 1];
        let p2 = input[start + 2];

        let period_wma_sub = p0 + p1 + p2;
        let period_wma_sum = p0 + p1 * 2.0 + p2 * 3.0;

        let state = WmaState {
            period_wma_sub,
            period_wma_sum,
            trailing_wma_value: 0.0,
            trailing_wma_idx: start,
        };
        (state, start + 3)
    }

    #[inline(always)]
    fn next(&mut self, input: &[f64], new_price: f64) -> f64 {
        self.period_wma_sub += new_price;
        self.period_wma_sub -= self.trailing_wma_value;
        self.period_wma_sum += new_price * 4.0;
        self.trailing_wma_value = input[self.trailing_wma_idx];
        self.trailing_wma_idx += 1;
        let smoothed = self.period_wma_sum * 0.1;
        self.period_wma_sum -= self.period_wma_sub;
        smoothed
    }
}

/// Hilbert Transform - Instantaneous Trendline
///
/// Faithful port of C TA-Lib ta_HT_TRENDLINE.c.
/// lookback = 63
pub fn ht_trendline(input: &[f64]) -> TaResult<Vec<f64>> {
    let len = input.len();
    let lookback: usize = 63;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let start_idx = lookback;

    let trailing_wma_start = start_idx - lookback; // = 0
    let (mut wma, mut today) = WmaState::init(input, trailing_wma_start);

    // Warm up WMA: 34 iterations (matching C TA-Lib for lookback 63 functions)
    for _ in 0..34 {
        let val = input[today];
        today += 1;
        let _ = wma.next(input, val);
    }

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

    // Trendline WMA(4) smoothing variables
    let mut i_trend1: f64 = 0.0;
    let mut i_trend2: f64 = 0.0;
    let mut i_trend3: f64 = 0.0;

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    while today < len {
        let adjusted_prev_period = 0.075 * period + 0.54;

        let today_value = input[today];
        let smoothed_value = wma.next(input, today_value);

        let (detrender, q1, i2, q2);

        if today % 2 == 0 {
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

            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = detrender;
        } else {
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

            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = detrender;
        }

        // Adjust period
        re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re;
        im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im;
        prev_q2 = q2;
        prev_i2 = i2;

        let temp_real = period;
        if im != 0.0 && re != 0.0 {
            period = 360.0 / ((im / re).atan() * (180.0 / std::f64::consts::PI));
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

        // Compute trendline: SMA of input prices over dcPeriod, then WMA(4) smooth
        let dc_period = smooth_period + 0.5;
        let dc_period_int = dc_period as i32;

        let mut temp = 0.0_f64;
        let mut price_idx = today;
        for _ in 0..dc_period_int {
            temp += input[price_idx];
            if price_idx == 0 {
                break;
            }
            price_idx -= 1;
        }

        if dc_period_int > 0 {
            temp /= dc_period_int as f64;
        }

        // WMA(4) smoothing of the SMA result
        let trendline = (4.0 * temp + 3.0 * i_trend1 + 2.0 * i_trend2 + i_trend3) / 10.0;
        i_trend3 = i_trend2;
        i_trend2 = i_trend1;
        i_trend1 = temp;

        if today >= start_idx {
            output[today] = trendline;
        }

        today += 1;
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
