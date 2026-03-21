//! 单调队列 (Monotonic Deque) 实现的滑动窗口极值
//!
//! 用于 O(n) 计算固定窗口的最大/最小值及其索引。
//! 替代 O(n*period) 的朴素扫描，对 AROON/WILLR/MAX/MIN 等指标提速显著。

use std::collections::VecDeque;

/// 滑动窗口最大值 + 索引 (单调递减队列)
///
/// 返回每个窗口的 (max_value, max_index)。
/// 总复杂度 O(n)，每个元素最多入队出队各一次。
pub fn sliding_max(data: &[f64], window: usize) -> Vec<(f64, usize)> {
    let len = data.len();
    let mut result = Vec::with_capacity(len.saturating_sub(window - 1));
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(window);

    for i in 0..len {
        // 移除超出窗口的元素
        while let Some(&front) = deque.front() {
            if front + window <= i {
                deque.pop_front();
            } else {
                break;
            }
        }
        // 保持单调递减：移除所有小于当前值的队尾
        while let Some(&back) = deque.back() {
            if data[back] <= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);

        if i >= window - 1 {
            let max_idx = *deque.front().unwrap();
            result.push((data[max_idx], max_idx));
        }
    }

    result
}

/// 滑动窗口最小值 + 索引 (单调递增队列)
pub fn sliding_min(data: &[f64], window: usize) -> Vec<(f64, usize)> {
    let len = data.len();
    let mut result = Vec::with_capacity(len.saturating_sub(window - 1));
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(window);

    for i in 0..len {
        while let Some(&front) = deque.front() {
            if front + window <= i {
                deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = deque.back() {
            if data[back] >= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);

        if i >= window - 1 {
            let min_idx = *deque.front().unwrap();
            result.push((data[min_idx], min_idx));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_max() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0];
        let result = sliding_max(&data, 3);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], (3.0, 1)); // [1,3,2] -> max=3 at 1
        assert_eq!(result[1], (5.0, 3)); // [3,2,5] -> max=5 at 3
        assert_eq!(result[2], (5.0, 3)); // [2,5,4] -> max=5 at 3
        assert_eq!(result[3], (5.0, 3)); // [5,4,1] -> max=5 at 3
    }

    #[test]
    fn test_sliding_min() {
        let data = vec![5.0, 3.0, 4.0, 1.0, 2.0, 6.0];
        let result = sliding_min(&data, 3);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], (3.0, 1)); // [5,3,4] -> min=3 at 1
        assert_eq!(result[1], (1.0, 3)); // [3,4,1] -> min=1 at 3
        assert_eq!(result[2], (1.0, 3)); // [4,1,2] -> min=1 at 3
        assert_eq!(result[3], (1.0, 3)); // [1,2,6] -> min=1 at 3
    }
}
