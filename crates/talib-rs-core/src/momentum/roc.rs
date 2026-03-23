use crate::error::{TaError, TaResult};

/// Rate of Change (ROC)
/// ROC = ((close - close_n) / close_n) * 100
pub fn roc(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);
    for i in timeperiod..len {
        unsafe {
            let prev = *input.get_unchecked(i - timeperiod);
            *output.get_unchecked_mut(i) = if prev != 0.0 {
                ((*input.get_unchecked(i) - prev) / prev) * 100.0
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

/// Rate of Change Percentage (ROCP)
/// ROCP = (close - close_n) / close_n
pub fn rocp(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0; len];
    for i in 0..timeperiod {
        output[i] = f64::NAN;
    }
    roc_core_simd(input, &mut output, timeperiod, |cur, prev| {
        if prev != 0.0 { (cur - prev) / prev } else { 0.0 }
    }, |cur_v, prev_v| {
        #[cfg(feature = "simd")]
        { (cur_v - prev_v) / prev_v }
        #[cfg(not(feature = "simd"))]
        { let _ = (cur_v, prev_v); unreachable!() }
    });
    Ok(output)
}

/// Rate of Change Ratio (ROCR)
/// ROCR = close / close_n
pub fn rocr(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0; len];
    for i in 0..timeperiod {
        output[i] = f64::NAN;
    }
    roc_core_simd(input, &mut output, timeperiod, |cur, prev| {
        if prev != 0.0 { cur / prev } else { 0.0 }
    }, |cur_v, prev_v| {
        #[cfg(feature = "simd")]
        { cur_v / prev_v }
        #[cfg(not(feature = "simd"))]
        { let _ = (cur_v, prev_v); unreachable!() }
    });
    Ok(output)
}

/// Rate of Change Ratio 100 (ROCR100)
/// ROCR100 = (close / close_n) * 100
pub fn rocr100(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_roc_params(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);
    for i in timeperiod..len {
        unsafe {
            let prev = *input.get_unchecked(i - timeperiod);
            *output.get_unchecked_mut(i) = if prev != 0.0 {
                (*input.get_unchecked(i) / prev) * 100.0
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

/// Generic SIMD ROC core: processes input[timeperiod..] with SIMD chunks and scalar fixup for zero.
#[inline(always)]
fn roc_core_simd<F, G>(
    input: &[f64],
    output: &mut [f64],
    timeperiod: usize,
    scalar_fn: F,
    #[allow(unused_variables)]
    simd_fn: G,
) where
    F: Fn(f64, f64) -> f64,
    G: Fn(wide::f64x4, wide::f64x4) -> wide::f64x4,
{
    let len = input.len();

    #[cfg(feature = "simd")]
    {
        let count = len - timeperiod;
        let chunks = count / 4;

        for i in 0..chunks {
            let base = timeperiod + i * 4;
            let cur_v = wide::f64x4::new([input[base], input[base + 1], input[base + 2], input[base + 3]]);
            let prev_v = wide::f64x4::new([
                input[base - timeperiod],
                input[base - timeperiod + 1],
                input[base - timeperiod + 2],
                input[base - timeperiod + 3],
            ]);
            let result = simd_fn(cur_v, prev_v);
            let arr = result.to_array();
            let parr = prev_v.to_array();
            // Fixup zeros (rare in financial data)
            output[base] = if parr[0] != 0.0 { arr[0] } else { 0.0 };
            output[base + 1] = if parr[1] != 0.0 { arr[1] } else { 0.0 };
            output[base + 2] = if parr[2] != 0.0 { arr[2] } else { 0.0 };
            output[base + 3] = if parr[3] != 0.0 { arr[3] } else { 0.0 };
        }

        let tail = timeperiod + chunks * 4;
        for i in tail..len {
            output[i] = scalar_fn(input[i], input[i - timeperiod]);
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for i in timeperiod..len {
            output[i] = scalar_fn(input[i], input[i - timeperiod]);
        }
    }
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
