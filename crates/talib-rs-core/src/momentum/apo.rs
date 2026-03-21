use crate::error::TaResult;
use crate::ma_type::{compute_ma, MaType};

/// Absolute Price Oscillator (APO)
///
/// APO = MA(fast) - MA(slow)
pub fn apo(
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
        if !fast[i].is_nan() && !slow[i].is_nan() {
            output[i] = fast[i] - slow[i];
        }
    }

    Ok(output)
}
