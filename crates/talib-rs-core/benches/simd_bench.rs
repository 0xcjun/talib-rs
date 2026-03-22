use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use talib_rs::simd;

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f64");
    for size in [100, 1000, 10000, 100000] {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| simd::sum_f64(black_box(d)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| {
                let sum: f64 = black_box(d).iter().sum();
                sum
            });
        });
    }
    group.finish();
}

fn bench_sum_sq_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_sq_diff");
    for size in [100, 1000, 10000] {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
        let mean = simd::sum_f64(&data) / size as f64;
        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| simd::sum_sq_diff(black_box(d), mean));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| {
                let m = mean;
                let sum: f64 = black_box(d)
                    .iter()
                    .map(|&x| {
                        let diff = x - m;
                        diff * diff
                    })
                    .sum();
                sum
            });
        });
    }
    group.finish();
}

fn bench_add_arrays(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_arrays");
    for size in [1000, 10000, 100000] {
        let a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..size).map(|i| i as f64 * 2.0).collect();
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| simd::add_arrays(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                let r: Vec<f64> = black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(x, y)| x + y)
                    .collect();
                r
            });
        });
    }
    group.finish();
}

fn bench_sqrt_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt_array");
    for size in [1000, 10000, 100000] {
        let data: Vec<f64> = (1..=size).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| simd::sqrt_array(black_box(d)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| {
                let r: Vec<f64> = black_box(d).iter().map(|&v| v.sqrt()).collect();
                r
            });
        });
    }
    group.finish();
}

fn bench_indicators(c: &mut Criterion) {
    let close: Vec<f64> = (0..10000)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 10.0)
        .collect();

    let mut group = c.benchmark_group("indicators_10k");
    group.bench_function("SMA_20", |b| {
        b.iter(|| talib_rs::overlap::sma(black_box(&close), 20));
    });
    group.bench_function("EMA_20", |b| {
        b.iter(|| talib_rs::overlap::ema(black_box(&close), 20));
    });
    group.bench_function("RSI_14", |b| {
        b.iter(|| talib_rs::momentum::rsi(black_box(&close), 14));
    });
    group.bench_function("BBANDS_20", |b| {
        b.iter(|| {
            talib_rs::overlap::bbands(
                black_box(&close),
                20,
                2.0,
                2.0,
                talib_rs::MaType::Sma,
            )
        });
    });
    group.bench_function("STDDEV_20", |b| {
        b.iter(|| talib_rs::statistic::stddev(black_box(&close), 20, 1.0));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_sum,
    bench_sum_sq_diff,
    bench_add_arrays,
    bench_sqrt_array,
    bench_indicators
);
criterion_main!(benches);
