use rustcombination::*;
use crate::{ConditionerHelper, RecombineInterface};

use rand;
use rand::prelude::*;
use rand_distr;

use std::error::Error;
use std::time::{Duration, SystemTime, Instant};

struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Matrix {
        let rng = rand::thread_rng();
        let dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();

        Matrix {
            data: dist.sample_iter(rng).take(rows * cols).collect(),
            rows,
            cols,
        }
    }

    fn trace(&self) -> f64 {
        let mut result = 0.0f64;
        for i in 0..self.cols {
            result += self.data[i * (self.cols + 1)];
        }
        result
    }

    fn mean(&self, no_points: usize) -> Vec<f64> {
        let mut result = vec![0.0f64; self.cols];
        for i in 0..no_points {
            for j in 0..self.cols {
                result[j] += self.data[i * self.cols + j];
            }
        }
        result
    }

    fn weighted_indexed_mean(
        &self,
        _no_points: usize,
        indices: &[usize],
        weights: &[f64],
    ) -> Vec<f64> {
        let mut result = vec![0.0f64; self.cols];
        for (p, w) in indices.iter().zip(weights.iter()) {
            for j in 0..self.cols {
                result[j] += w * self.data[p * self.cols + j];
            }
        }
        result
    }
}

fn compare(mean1: &[f64], mean2: &[f64]) -> f64 {
    let t1: f64 = mean1.iter().map(|v| f64::abs(*v)).sum();
    let t2: f64 = mean2.iter().map(|v| f64::abs(*v)).sum();
    let diff: f64 = mean1
        .iter()
        .zip(mean2.iter())
        .map(|(u, v)| f64::abs(u - v))
        .sum();
    diff / (t1 + t2)
}

struct Output {
    num_kept_points: usize,
    locs: Vec<usize>,
    weights: Vec<f64>,
}

struct TestRecombineInterface {
    data: Matrix,
    output: Output,
    degree: usize,
    weights: Vec<f64>,
}

impl TestRecombineInterface {
    fn new(data: Matrix, degree: usize, weights: Vec<f64>) -> Self {
        let cols = data.cols;
        let num_kept_points = monomials::num_monomials(cols, degree);
        Self {
            data,
            degree,
            weights,
            output: Output {
                num_kept_points,
                locs: vec![0usize; num_kept_points],
                weights: vec![0.0f64; num_kept_points],
            },
        }
    }
}

impl RecombineInterface for TestRecombineInterface {
    fn points_in_cloud(&self) -> usize {
        self.data.rows
    }

    fn expand_points(
        &self,
        output: &mut [f64],
        _helper: &ConditionerHelper,
    ) -> Result<(), Box<dyn Error>> {
        // In practice, derive this otherwise
        let depth_of_vector = self.output.num_kept_points;
        let mut max_buf = vec![0.0f64; self.data.cols];
        let mut min_buf = vec![0.0f64; self.data.cols];

        for row in self.data.data.chunks_exact(self.data.cols) {
            for i in 0..self.data.cols {
                max_buf[i] = f64::max(row[i], max_buf[i]);
                min_buf[i] = f64::min(row[i], min_buf[i]);
            }
        }
        for (j, row) in self.data.data.chunks_exact(self.data.cols).enumerate() {
            if self.degree == 1 {
                output[j * depth_of_vector] = 1.0;
                for i in 0..self.data.cols {
                    output[j * depth_of_vector + i + 1] = if max_buf[i] == min_buf[i] {
                        0.0f64
                    } else {
                        (2.0 * row[i] - (min_buf[i] + max_buf[i])) / (max_buf[i] - min_buf[i])
                    };
                    debug_assert!(output[j * depth_of_vector + i + 1] >= (-1.0 - f64::EPSILON) &&
                        output[j * depth_of_vector + i + 1] <= (1.0 + f64::EPSILON));
                }
            } else {
                todo!();
            }
        }

        Ok(())
    }

    fn degree(&self) -> usize {
        self.output.num_kept_points
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn set_output(&mut self, locs: &[usize], weights: &[f64]) {
        debug_assert!(weights.len() == self.output.weights.len());
        debug_assert!(locs.len() == self.output.locs.len());
        self.output.locs.copy_from_slice(locs);
        self.output.weights.copy_from_slice(weights);
    }
}

struct SimpleTimer<'a> {
    start: Instant,
    out: &'a mut Duration,
}

impl<'a> SimpleTimer<'a> {
    fn new(out: &'a mut Duration) -> Self {
        SimpleTimer {
            start: Instant::now(),
            out,
        }
    }
}

impl<'a> Drop for SimpleTimer<'a> {
    fn drop(&mut self) {
        *self.out = Instant::now().duration_since(self.start);
    }
}

fn main() {
    const NUM_POINTS: usize = 10000;
    const DIMENSION: usize = 500;

    let m = Matrix::new(NUM_POINTS, DIMENSION);
    let indices: Vec<usize> = (0..NUM_POINTS).collect();

    let weights = vec![1.0f64 / (NUM_POINTS as f64); NUM_POINTS];

    let mean2 = m.weighted_indexed_mean(NUM_POINTS, &indices, &weights);

    let mut interface = TestRecombineInterface::new(m, 1, weights);

    let mut duration = Duration::new(0, 0);
    {
        SimpleTimer::new(&mut duration);
        recombine(&mut interface).unwrap();
    }

    println!("Test took {}s", duration.as_secs_f64());
    let mean3 = interface.data.weighted_indexed_mean(
        NUM_POINTS,
        &interface.output.locs,
        &interface.output.weights,
    );

    let comp = compare(&mean2, &mean3);
    dbg!(&comp);
    assert!(comp < 1.0e-12);
}
