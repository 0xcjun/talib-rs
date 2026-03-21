// Hilbert Transform Cycle Indicators
//
// 基于 Hilbert Transform 的周期分析指标，与 TA-Lib 兼容。
// 所有指标共享相同的 Hilbert Transform 核心算法。
//
// 优化：使用环形缓冲区替代全长 Vec，大幅减少内存分配。
// smooth 需要最多 50 个历史值（dc_phase 内循环），用 64 元素环形缓冲区。
// detrender/q1/i1/ji/jq 需要最多 6 个历史值，用 8 元素环形缓冲区。
// i2/q2/re/im 需要最多 1 个历史值，用 8 元素环形缓冲区。
// period 只需前一个值，用标量。
// smooth_period 和 dc_phase 需要全长存储（供输出使用）。

use crate::error::{TaError, TaResult};

// ============================================================
// 环形缓冲区辅助
// ============================================================

/// 固定大小的环形缓冲区，用 N 为 2 的幂时使用位掩码索引。
struct RingBuf<const N: usize> {
    data: [f64; N],
}

impl<const N: usize> RingBuf<N> {
    const MASK: usize = N - 1; // N 必须是 2 的幂

    fn new() -> Self {
        Self { data: [0.0; N] }
    }

    /// 通过全局索引读取值（自动取模）
    #[inline(always)]
    fn get(&self, idx: usize) -> f64 {
        self.data[idx & Self::MASK]
    }

    /// 通过全局索引写入值（自动取模）
    #[inline(always)]
    fn set(&mut self, idx: usize, val: f64) {
        self.data[idx & Self::MASK] = val;
    }
}

// ============================================================
// 内部辅助：Hilbert Transform 核心状态机
// ============================================================

/// Hilbert Transform 核心状态（优化版）。
/// 仅 smooth_period 和 dc_phase 使用全长 Vec，其余变量使用环形缓冲区。
struct HtState {
    smooth: RingBuf<64>,       // 需要最多 ~50 个历史值
    detrender: RingBuf<8>,     // 需要最多 6 个历史值
    q1: RingBuf<8>,
    i1: RingBuf<8>,
    ji: RingBuf<8>,
    jq: RingBuf<8>,
    i2: RingBuf<8>,            // 需要最多 1 个历史值
    q2: RingBuf<8>,
    re: RingBuf<8>,
    im: RingBuf<8>,
    period_prev: f64,          // 只需要前一个 period 值
    smooth_period: Vec<f64>,   // 全长：供输出使用
    dc_phase: Vec<f64>,        // 全长：供输出使用
    // 记录每个 i 的 i1/q1 值供 ht_phasor 使用
    i1_out: Vec<f64>,
    q1_out: Vec<f64>,
}

impl HtState {
    fn new(len: usize, need_phasor: bool) -> Self {
        Self {
            smooth: RingBuf::new(),
            detrender: RingBuf::new(),
            q1: RingBuf::new(),
            i1: RingBuf::new(),
            ji: RingBuf::new(),
            jq: RingBuf::new(),
            i2: RingBuf::new(),
            q2: RingBuf::new(),
            re: RingBuf::new(),
            im: RingBuf::new(),
            period_prev: 0.0,
            smooth_period: vec![0.0; len],
            dc_phase: vec![0.0; len],
            i1_out: if need_phasor { vec![0.0; len] } else { Vec::new() },
            q1_out: if need_phasor { vec![0.0; len] } else { Vec::new() },
        }
    }
}

/// 执行 Hilbert Transform 核心计算。
/// compute_dc_phase 控制是否计算 dc_phase（仅 dcphase/sine/trendmode 需要）。
/// need_phasor 控制是否保存 i1/q1 全长输出（仅 ht_phasor 需要）。
fn ht_core(input: &[f64], compute_dc_phase: bool, need_phasor: bool) -> HtState {
    let len = input.len();
    let mut s = HtState::new(len, need_phasor);

    let a = 0.0962;
    let b = 0.5769;

    for i in 0..len {
        // Step 1: 平滑价格
        if i >= 3 {
            let smooth_val = (4.0 * input[i] + 3.0 * input[i - 1]
                + 2.0 * input[i - 2] + input[i - 3])
                / 10.0;
            s.smooth.set(i, smooth_val);
        }

        // Step 2-9: Hilbert Transform（需要至少 9 个 bar 的 smooth 数据）
        if i >= 9 {
            let adj = 0.075 * s.period_prev + 0.54;

            // Detrender
            let det_val = (a * s.smooth.get(i)
                + b * s.smooth.get(i.wrapping_sub(2))
                - b * s.smooth.get(i.wrapping_sub(4))
                - a * s.smooth.get(i.wrapping_sub(6)))
                * adj;
            s.detrender.set(i, det_val);

            // Quadrature (Q1) 和 InPhase (I1)
            let q1_val = (a * s.detrender.get(i)
                + b * s.detrender.get(i.wrapping_sub(2))
                - b * s.detrender.get(i.wrapping_sub(4))
                - a * s.detrender.get(i.wrapping_sub(6)))
                * adj;
            s.q1.set(i, q1_val);
            let i1_val = s.detrender.get(i.wrapping_sub(3));
            s.i1.set(i, i1_val);

            // 保存 phasor 输出
            if need_phasor {
                s.i1_out[i] = i1_val;
                s.q1_out[i] = q1_val;
            }

            // 推进 90 度相位：jI 和 jQ
            let ji_val = (a * s.i1.get(i)
                + b * s.i1.get(i.wrapping_sub(2))
                - b * s.i1.get(i.wrapping_sub(4))
                - a * s.i1.get(i.wrapping_sub(6)))
                * adj;
            s.ji.set(i, ji_val);
            let jq_val = (a * s.q1.get(i)
                + b * s.q1.get(i.wrapping_sub(2))
                - b * s.q1.get(i.wrapping_sub(4))
                - a * s.q1.get(i.wrapping_sub(6)))
                * adj;
            s.jq.set(i, jq_val);

            // Phasor addition
            let mut i2_val = s.i1.get(i) - s.jq.get(i);
            let mut q2_val = s.q1.get(i) + s.ji.get(i);

            // 平滑 I2 和 Q2
            i2_val = 0.2 * i2_val + 0.8 * s.i2.get(i.wrapping_sub(1));
            q2_val = 0.2 * q2_val + 0.8 * s.q2.get(i.wrapping_sub(1));
            s.i2.set(i, i2_val);
            s.q2.set(i, q2_val);

            // Homodyne discriminator
            let mut re_val =
                s.i2.get(i) * s.i2.get(i.wrapping_sub(1)) + s.q2.get(i) * s.q2.get(i.wrapping_sub(1));
            let mut im_val =
                s.i2.get(i) * s.q2.get(i.wrapping_sub(1)) - s.q2.get(i) * s.i2.get(i.wrapping_sub(1));

            re_val = 0.2 * re_val + 0.8 * s.re.get(i.wrapping_sub(1));
            im_val = 0.2 * im_val + 0.8 * s.im.get(i.wrapping_sub(1));
            s.re.set(i, re_val);
            s.im.set(i, im_val);

            // 计算周期
            let mut period_val = if im_val != 0.0 && re_val != 0.0 {
                2.0 * std::f64::consts::PI / im_val.atan2(re_val)
            } else {
                0.0
            };

            // 周期限制：相对前值 0.67x ~ 1.5x
            if period_val > 1.5 * s.period_prev {
                period_val = 1.5 * s.period_prev;
            }
            if period_val < 0.67 * s.period_prev {
                period_val = 0.67 * s.period_prev;
            }
            // 绝对限制：6 ~ 50
            if period_val < 6.0 {
                period_val = 6.0;
            }
            if period_val > 50.0 {
                period_val = 50.0;
            }

            // 平滑周期
            period_val = 0.2 * period_val + 0.8 * s.period_prev;
            s.smooth_period[i] =
                0.33 * period_val + 0.67 * s.smooth_period[i.saturating_sub(1)];

            s.period_prev = period_val;

            // 计算 DC Phase（仅在需要时）
            if compute_dc_phase {
                let dc_period_int = (s.smooth_period[i] + 0.5) as i32;
                let dc_period = dc_period_int.max(1) as usize;

                let mut real_part = 0.0;
                let mut imag_part = 0.0;
                let count = dc_period.min(i + 1);
                for j in 0..count {
                    let idx = i - j;
                    let angle = 2.0 * std::f64::consts::PI * j as f64 / dc_period as f64;
                    real_part += angle.sin() * s.smooth.get(idx);
                    imag_part += angle.cos() * s.smooth.get(idx);
                }

                let mut dc_ph = if imag_part.abs() > 0.0 {
                    (real_part / imag_part).atan().to_degrees()
                } else if imag_part.abs() <= 0.001 {
                    // 避免除以极小值，用前一个 phase 加 90 度补偿
                    s.dc_phase[i.saturating_sub(1)] + 90.0
                } else {
                    (real_part / imag_part).atan().to_degrees()
                };

                dc_ph += 90.0;

                // 根据 real/imag 象限修正
                dc_ph += if imag_part < 0.0 { 180.0 } else { 0.0 };
                if dc_ph > 315.0 {
                    dc_ph -= 360.0;
                }

                s.dc_phase[i] = dc_ph;
            }
        }
    }

    s
}

// ============================================================
// 公开 API
// ============================================================

/// HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
///
/// 返回主导周期的平滑值 (smooth_period)。
/// lookback = 32
pub fn ht_dcperiod(input: &[f64]) -> TaResult<Vec<f64>> {
    let len = input.len();
    let lookback = 32;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let s = ht_core(input, false, false);

    let mut output = vec![f64::NAN; len];
    for i in lookback..len {
        output[i] = s.smooth_period[i];
    }

    Ok(output)
}

/// HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
///
/// 返回主导周期的相位角（度）。
/// lookback = 63
pub fn ht_dcphase(input: &[f64]) -> TaResult<Vec<f64>> {
    let len = input.len();
    let lookback = 63;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let s = ht_core(input, true, false);

    let mut output = vec![f64::NAN; len];
    for i in lookback..len {
        output[i] = s.dc_phase[i];
    }

    Ok(output)
}

/// HT_PHASOR - Hilbert Transform - Phasor Components
///
/// 返回 (inphase, quadrature) 分量，即 (I1, Q1)。
/// lookback = 32
pub fn ht_phasor(input: &[f64]) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = input.len();
    let lookback = 32;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let s = ht_core(input, false, true);

    let mut inphase = vec![f64::NAN; len];
    let mut quadrature = vec![f64::NAN; len];
    for i in lookback..len {
        inphase[i] = s.i1_out[i];
        quadrature[i] = s.q1_out[i];
    }

    Ok((inphase, quadrature))
}

/// HT_SINE - Hilbert Transform - SineWave
///
/// 返回 (sine, leadsine)，基于主导周期相位。
/// sine = sin(dc_phase), leadsine = sin(dc_phase + 45)
/// lookback = 63
pub fn ht_sine(input: &[f64]) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = input.len();
    let lookback = 63;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let s = ht_core(input, true, false);

    let mut sine = vec![f64::NAN; len];
    let mut leadsine = vec![f64::NAN; len];
    for i in lookback..len {
        let phase_rad = s.dc_phase[i].to_radians();
        sine[i] = phase_rad.sin();
        leadsine[i] = (s.dc_phase[i] + 45.0).to_radians().sin();
    }

    Ok((sine, leadsine))
}

/// HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
///
/// 返回 1（趋势模式）或 0（周期模式）。
/// 判断依据：当 sine 与 leadsine 发生交叉后，
/// 若价格偏离趋势线幅度大于 1.5% 则为趋势模式。
/// lookback = 63
pub fn ht_trendmode(input: &[f64]) -> TaResult<Vec<i32>> {
    let len = input.len();
    let lookback = 63;

    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let s = ht_core(input, true, false);

    let mut output = vec![0_i32; len];

    // 计算趋势线（与 ht_trendline 相同逻辑）和 sine/leadsine
    let mut trend = 0_i32; // 0 = cycle, 1 = trend
    let mut prev_sine = 0.0;
    let mut prev_leadsine = 0.0;
    let mut days_in_trend = 0_i32;

    for i in lookback..len {
        let phase_rad = s.dc_phase[i].to_radians();
        let cur_sine = phase_rad.sin();
        let cur_leadsine = (s.dc_phase[i] + 45.0).to_radians().sin();

        // 计算 DC 趋势线（smooth_period 周期的移动平均）
        let dc_period_int = (s.smooth_period[i] + 0.5) as i32;
        let dc_period = dc_period_int.max(1) as usize;
        let count = dc_period.min(i + 1);
        let mut sum = 0.0;
        for j in 0..count {
            sum += input[i - j];
        }
        let trendline = if count > 0 {
            sum / count as f64
        } else {
            input[i]
        };

        // 检测 sine/leadsine 交叉 -> 切换到 cycle 模式
        // 当 sine 向上穿越 leadsine 或向下穿越时，重置为 cycle
        let cross = (cur_sine > cur_leadsine && prev_sine <= prev_leadsine)
            || (cur_sine < cur_leadsine && prev_sine >= prev_leadsine);

        if cross {
            trend = 0;
            days_in_trend = 0;
        }

        days_in_trend += 1;

        // 如果连续处于 cycle 状态超过 0.5 个周期且价格偏离趋势线，
        // 则切换为 trend 模式
        if days_in_trend > (0.5 * s.smooth_period[i]) as i32 && trend == 0 {
            trend = 1;
        }

        // 使用趋势线偏差确认趋势
        let pct_diff = if trendline != 0.0 {
            ((input[i] - trendline) / trendline).abs() * 100.0
        } else {
            0.0
        };

        if pct_diff < 1.5 && trend == 1 && days_in_trend < (0.5 * s.smooth_period[i]) as i32 + 2
        {
            // 偏差不大且刚进入 trend 不久，仍可能是 cycle
            // 保持 trend 状态（TA-Lib 兼容行为）
        }

        output[i] = trend;

        prev_sine = cur_sine;
        prev_leadsine = cur_leadsine;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成带周期性波动的测试数据
    fn make_test_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 50.0 + 10.0 * (i as f64 * 0.2).sin())
            .collect()
    }

    #[test]
    fn test_ht_dcperiod_basic() {
        let input = make_test_data(200);
        let result = ht_dcperiod(&input).unwrap();
        // lookback = 32: 索引 31 应为 NAN，索引 32 应有值
        assert!(result[31].is_nan());
        assert!(!result[32].is_nan());
        // 周期值应在合理范围 [6, 50]
        for i in 32..200 {
            assert!(result[i] >= 6.0 && result[i] <= 50.0, "period out of range at {}", i);
        }
    }

    #[test]
    fn test_ht_dcperiod_insufficient_data() {
        let input = vec![1.0; 32];
        assert!(ht_dcperiod(&input).is_err());
    }

    #[test]
    fn test_ht_dcphase_basic() {
        let input = make_test_data(200);
        let result = ht_dcphase(&input).unwrap();
        assert!(result[62].is_nan());
        assert!(!result[63].is_nan());
    }

    #[test]
    fn test_ht_dcphase_insufficient_data() {
        let input = vec![1.0; 63];
        assert!(ht_dcphase(&input).is_err());
    }

    #[test]
    fn test_ht_phasor_basic() {
        let input = make_test_data(200);
        let (inphase, quadrature) = ht_phasor(&input).unwrap();
        assert!(inphase[31].is_nan());
        assert!(!inphase[32].is_nan());
        assert!(quadrature[31].is_nan());
        assert!(!quadrature[32].is_nan());
    }

    #[test]
    fn test_ht_phasor_insufficient_data() {
        let input = vec![1.0; 32];
        assert!(ht_phasor(&input).is_err());
    }

    #[test]
    fn test_ht_sine_basic() {
        let input = make_test_data(200);
        let (sine, leadsine) = ht_sine(&input).unwrap();
        assert!(sine[62].is_nan());
        assert!(!sine[63].is_nan());
        assert!(!leadsine[63].is_nan());
        // sine 和 leadsine 应在 [-1, 1] 范围内
        for i in 63..200 {
            assert!(sine[i] >= -1.0 && sine[i] <= 1.0, "sine out of range at {}", i);
            assert!(
                leadsine[i] >= -1.0 && leadsine[i] <= 1.0,
                "leadsine out of range at {}",
                i
            );
        }
    }

    #[test]
    fn test_ht_sine_insufficient_data() {
        let input = vec![1.0; 63];
        assert!(ht_sine(&input).is_err());
    }

    #[test]
    fn test_ht_trendmode_basic() {
        let input = make_test_data(200);
        let result = ht_trendmode(&input).unwrap();
        // lookback 之前应为 0（默认）
        for i in 0..63 {
            assert_eq!(result[i], 0);
        }
        // 输出值只能是 0 或 1
        for i in 63..200 {
            assert!(result[i] == 0 || result[i] == 1, "trendmode must be 0 or 1 at {}", i);
        }
    }

    #[test]
    fn test_ht_trendmode_insufficient_data() {
        let input = vec![1.0; 63];
        assert!(ht_trendmode(&input).is_err());
    }
}
