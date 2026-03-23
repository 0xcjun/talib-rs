mod atr;

pub use atr::{atr, natr, trange};

/// 计算 True Range 数组（被多个模块复用）
pub fn true_range_array(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut tr = vec![0.0; len];
    if len > 0 {
        tr[0] = high[0] - low[0];
    }
    for i in 1..len {
        unsafe {
            let h = *high.get_unchecked(i);
            let l = *low.get_unchecked(i);
            let pc = *close.get_unchecked(i - 1);
            let hl = h - l;
            let hc = (h - pc).abs();
            let lc = (l - pc).abs();
            *tr.get_unchecked_mut(i) = hl.max(hc).max(lc);
        }
    }
    tr
}
