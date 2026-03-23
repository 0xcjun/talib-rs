use crate::error::{TaError, TaResult};

/// Rate of Change (ROC)
/// ROC = ((close - close_n) / close_n) * 100
pub fn roc(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);
    for (out, (&cur, &prev)) in output[timeperiod..]
        .iter_mut()
        .zip(input[timeperiod..].iter().zip(input[..len - timeperiod].iter()))
    {
        *out = if prev != 0.0 {
            ((cur - prev) / prev) * 100.0
        } else {
            0.0
        };
    }
    Ok(output)
}

/// Rate of Change Percentage (ROCP)
/// ROCP = (close - close_n) / close_n
pub fn rocp(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);
    for (out, (&cur, &prev)) in output[timeperiod..]
        .iter_mut()
        .zip(input[timeperiod..].iter().zip(input[..len - timeperiod].iter()))
    {
        *out = if prev != 0.0 {
            (cur - prev) / prev
        } else {
            0.0
        };
    }
    Ok(output)
}

/// Rate of Change Ratio (ROCR)
/// ROCR = close / close_n
pub fn rocr(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);
    for (out, (&cur, &prev)) in output[timeperiod..]
        .iter_mut()
        .zip(input[timeperiod..].iter().zip(input[..len - timeperiod].iter()))
    {
        *out = if prev != 0.0 {
            cur / prev
        } else {
            0.0
        };
    }
    Ok(output)
}

/// Rate of Change Ratio 100 (ROCR100)
/// ROCR100 = (close / close_n) * 100
pub fn rocr100(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);
    for (out, (&cur, &prev)) in output[timeperiod..]
        .iter_mut()
        .zip(input[timeperiod..].iter().zip(input[..len - timeperiod].iter()))
    {
        *out = if prev != 0.0 {
            (cur / prev) * 100.0
        } else {
            0.0
        };
    }
    Ok(output)
}

fn validate_roc_params(input: &[f64], timeperiod: usize) -> TaResult<()> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    if input.len() <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: input.len(),
        });
    }
    Ok(())
}
