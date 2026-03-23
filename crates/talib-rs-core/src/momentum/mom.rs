use crate::error::{TaError, TaResult};

/// Momentum (MOM)
///
/// MOM = close - close[timeperiod ago]
/// lookback = timeperiod
pub fn mom(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    let len = input.len();
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let mut output = vec![0.0_f64; len]; // calloc — near-free
    output[..timeperiod].fill(f64::NAN);
    // iter_mut + zip: LLVM sees non-overlapping slices → auto-vectorizes
    for (out, (&cur, &prev)) in output[timeperiod..]
        .iter_mut()
        .zip(input[timeperiod..].iter().zip(input[..len - timeperiod].iter()))
    {
        *out = cur - prev;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mom_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mom(&input, 2).unwrap();
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 2.0).abs() < 1e-10);
    }
}
