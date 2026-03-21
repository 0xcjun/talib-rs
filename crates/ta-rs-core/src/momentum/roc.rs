use crate::error::{TaError, TaResult};

/// Rate of Change (ROC)
/// ROC = ((close - close_n) / close_n) * 100
pub fn roc(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let mut output = vec![f64::NAN; input.len()];
    for i in timeperiod..input.len() {
        if input[i - timeperiod] != 0.0 {
            output[i] = ((input[i] - input[i - timeperiod]) / input[i - timeperiod]) * 100.0;
        } else {
            output[i] = 0.0;
        }
    }
    Ok(output)
}

/// Rate of Change Percentage (ROCP)
/// ROCP = (close - close_n) / close_n
pub fn rocp(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let mut output = vec![f64::NAN; input.len()];
    for i in timeperiod..input.len() {
        if input[i - timeperiod] != 0.0 {
            output[i] = (input[i] - input[i - timeperiod]) / input[i - timeperiod];
        } else {
            output[i] = 0.0;
        }
    }
    Ok(output)
}

/// Rate of Change Ratio (ROCR)
/// ROCR = close / close_n
pub fn rocr(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let mut output = vec![f64::NAN; input.len()];
    for i in timeperiod..input.len() {
        if input[i - timeperiod] != 0.0 {
            output[i] = input[i] / input[i - timeperiod];
        } else {
            output[i] = 0.0;
        }
    }
    Ok(output)
}

/// Rate of Change Ratio 100 (ROCR100)
/// ROCR100 = (close / close_n) * 100
pub fn rocr100(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let mut output = vec![f64::NAN; input.len()];
    for i in timeperiod..input.len() {
        if input[i - timeperiod] != 0.0 {
            output[i] = (input[i] / input[i - timeperiod]) * 100.0;
        } else {
            output[i] = 0.0;
        }
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
