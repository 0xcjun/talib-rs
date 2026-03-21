use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// 将 Rust Vec<f64> 转换为 Python NumPy 数组 (零拷贝)
pub fn to_py_array(py: Python<'_>, data: Vec<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, data).into()
}

/// 将 Rust Vec<i32> 转换为 Python NumPy 数组 (用于 K 线形态)
pub fn to_py_array_i32(py: Python<'_>, data: Vec<i32>) -> Py<PyArray1<i32>> {
    PyArray1::from_vec(py, data).into()
}
