use crate::error::TaResult;
use crate::ma_type::{compute_ma, MaType};

/// Percentage Price Oscillator (PPO)
///
/// PPO = ((MA(fast) - MA(slow)) / MA(slow)) * 100
pub fn ppo(
    input: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    matype: MaType,
) -> TaResult<Vec<f64>> {
    let (fp, sp) = if fastperiod < slowperiod {
        (fastperiod, slowperiod)
    } else {
        (slowperiod, fastperiod)
    };

    let fast = compute_ma(input, fp, matype)?;
    let slow = compute_ma(input, sp, matype)?;

    let len = input.len();
    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if !fast[i].is_nan() && !slow[i].is_nan() && slow[i] != 0.0 {
            output[i] = ((fast[i] - slow[i]) / slow[i]) * 100.0;
        }
    }

    Ok(output)
}
