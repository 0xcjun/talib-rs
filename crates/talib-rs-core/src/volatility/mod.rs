mod atr;

pub use atr::{atr, natr, trange};

/// True Range array (used by multiple modules)
pub fn true_range_array(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = high.len();
    if len == 0 {
        return vec![];
    }
    let mut tr = Vec::with_capacity(len);
    tr.push(high[0] - low[0]);
    // zip for auto-vectorization
    for ((&h, &l), &pc) in high[1..].iter().zip(low[1..].iter()).zip(close[..len - 1].iter()) {
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        tr.push(hl.max(hc).max(lc));
    }
    tr
}
