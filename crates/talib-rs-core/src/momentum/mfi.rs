use crate::error::{TaError, TaResult};

/// Money Flow Index (MFI) — 优化版，减少中间分配
///
/// MFI = 100 - (100 / (1 + Money Flow Ratio))
/// Money Flow Ratio = Positive Money Flow / Negative Money Flow
/// lookback = timeperiod
pub fn mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() || len != volume.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()).min(volume.len()),
        });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    // 合并 tp 和 mf 到单一数组: mf[i] = tp[i] * volume[i]
    // 我们需要 tp 值来比较方向，和 mf 值做滑动求和
    // 用两个并排数组: tp_arr 和 mf_arr，但在一次循环里同时计算
    let mut tp_arr = vec![0.0_f64; len];
    let mut mf_arr = vec![0.0_f64; len];
    for i in 0..len {
        unsafe {
            let tp = (*high.get_unchecked(i) + *low.get_unchecked(i) + *close.get_unchecked(i))
                / 3.0;
            *tp_arr.get_unchecked_mut(i) = tp;
            *mf_arr.get_unchecked_mut(i) = tp * *volume.get_unchecked(i);
        }
    }

    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    // 初始窗口
    let mut pos_mf = 0.0_f64;
    let mut neg_mf = 0.0_f64;
    for i in 1..=timeperiod {
        unsafe {
            let tp_cur = *tp_arr.get_unchecked(i);
            let tp_prev = *tp_arr.get_unchecked(i - 1);
            let mf = *mf_arr.get_unchecked(i);
            if tp_cur > tp_prev {
                pos_mf += mf;
            } else if tp_cur < tp_prev {
                neg_mf += mf;
            }
        }
    }

    output[timeperiod] = if neg_mf > 0.0 {
        100.0 - (100.0 / (1.0 + pos_mf / neg_mf))
    } else {
        100.0
    };

    // 滑动窗口
    for i in (timeperiod + 1)..len {
        unsafe {
            // 移除旧值
            let old_idx = i - timeperiod;
            let old_tp = *tp_arr.get_unchecked(old_idx);
            let old_tp_prev = *tp_arr.get_unchecked(old_idx - 1);
            let old_mf = *mf_arr.get_unchecked(old_idx);
            if old_tp > old_tp_prev {
                pos_mf -= old_mf;
            } else if old_tp < old_tp_prev {
                neg_mf -= old_mf;
            }
            // 添加新值
            let new_tp = *tp_arr.get_unchecked(i);
            let new_tp_prev = *tp_arr.get_unchecked(i - 1);
            let new_mf = *mf_arr.get_unchecked(i);
            if new_tp > new_tp_prev {
                pos_mf += new_mf;
            } else if new_tp < new_tp_prev {
                neg_mf += new_mf;
            }

            *output.get_unchecked_mut(i) = if neg_mf > 0.0 {
                100.0 - (100.0 / (1.0 + pos_mf / neg_mf))
            } else {
                100.0
            };
        }
    }

    Ok(output)
}
