#[allow(non_snake_case)]
mod Cinterface;
mod errors;
mod recombine_interface;
mod reduction_tool;
mod reweight;
mod tree_buffer_helper;

extern crate blas_src;
extern crate lapack_src;

use std::collections::BTreeMap;
use std::error::Error;

pub use crate::errors::RecombineError;
pub use crate::recombine_interface::{ConditionerHelper, RecombineInterface};
pub use crate::reduction_tool::{LinearAlgebraReductionTool, SVDReductionTool};
pub use crate::tree_buffer_helper::TreeBufferHelper;
pub use crate::Cinterface::*;

#[no_mangle]
pub fn recombine(interface: &mut dyn RecombineInterface) -> Result<(), Box<dyn Error>> {
    let mut helper = TreeBufferHelper::new(interface);

    if helper.num_points() <= 1 {
        let locs: Vec<usize> = (0..helper.num_points()).collect();
        interface.set_output(&locs, &helper.weights());
    } else {
        let mut reduction_tool = SVDReductionTool::new();
        let mut weights: Vec<f64> = Vec::from(&helper.weights[0..helper.num_trees()]);

        let degree = helper.degree();
        let mut points = vec![0.0f64; helper.num_trees() * degree];
        helper.update_points_buffer(&mut points);
        let mut num_lapack_calls = 0usize;

        let mut done = false;
        while !done {
            let mut max_set = reduction_tool.move_mass(&mut weights, &mut points, degree)?;

            dbg!(max_set.len());
            done = max_set.is_empty();

            for to_go_position in max_set.drain(..) {
                debug_assert!(helper.current_roots.contains_key(&to_go_position));
                debug_assert!(helper
                    .tree_position
                    .contains_key(&helper.current_roots[&to_go_position]));
                helper
                    .tree_position
                    .remove(&helper.current_roots.remove(&to_go_position).unwrap())
                    .unwrap();

                debug_assert!(!helper.tree_position.is_empty());
                let (to_split, to_split_pos) = helper
                    .tree_position
                    .iter()
                    .next_back()
                    .map(|(i, j)| (*i, *j))
                    .unwrap();
                // .ok_or(RecombineError::InvalidTreeIndex(
                //     "current roots map is empty".into(),
                // ))?;
                debug_assert!(helper.tree_position.contains_key(&to_split));
                debug_assert!(helper.current_roots.contains_key(&to_split_pos));

                if !helper.is_leaf(to_split) {
                    helper.tree_position.remove(&to_split).unwrap();
                    helper.current_roots.remove(&to_split_pos).unwrap();

                    // .pop_root_index(to_split.1)
                    // .ok_or(RecombineError::InvalidTreeIndex(
                    //     "last element invalid?".into(),
                    // ))?;

                    let split_left = helper.left_index(to_split);
                    let split_right = helper.right_index(to_split);
                    helper.current_roots.insert(to_go_position, split_left);
                    helper.tree_position.insert(split_left, to_go_position);
                    helper.current_roots.insert(to_split_pos, split_right);
                    helper.tree_position.insert(split_right, to_split_pos);

                    weights[to_go_position] =
                        weights[to_split_pos] * helper.weight(split_left) / helper.weight(to_split);
                    weights[to_split_pos] *= helper.weight(split_right) / helper.weight(to_split);

                    let (left_pts, right_pts) = if to_go_position < to_split_pos {
                        let (tmpl, tmpr) = points.split_at_mut(to_split_pos * degree);
                        (
                            &mut tmpl[to_go_position * degree..(to_go_position + 1) * degree],
                            &mut tmpr[0..degree],
                        )
                    } else if to_split_pos < to_go_position {
                        let (tmpl, tmpr) = points.split_at_mut(to_go_position * degree);
                        (
                            &mut tmpr[0..degree],
                            &mut tmpl[to_split_pos * degree..(to_split_pos + 1) * degree],
                        )
                    } else {
                        return Err(RecombineError::InvalidTreeIndex(
                            "indices cannot be equal".into(),
                        )
                        .into());
                    };

                    let in_left = &helper[split_left];
                    let in_right = &helper[split_right];

                    for i in 0..degree {
                        left_pts[i] = in_left[i];
                        right_pts[i] = in_right[i];
                    }
                }
            }

            helper.repack_buffer(&mut points, &mut weights);
            num_lapack_calls = reduction_tool.num_linalg_calls();
        }

        let mut locs: Vec<usize> = Vec::with_capacity(helper.tree_position.len());
        let mut out_weights: Vec<f64> = Vec::with_capacity(helper.tree_position.len());
        for (i, wi) in helper.tree_position {
            locs.push(i);
            out_weights.push(weights[wi]);
        }

        dbg!(locs.len(), weights.len());
        interface.set_output(&locs, &out_weights);
    }

    Ok(())
}

pub mod monomials {

    fn count_monomials(letters: usize, degree: usize) -> usize {
        if (letters == 0 && degree > 0) {
            0
        } else {
            let mut ans = 1;
            for j in 1..letters {
                ans *= (j + degree);
                debug_assert_eq!(ans % j, 0);
                ans /= j;
            }
            ans
        }
    }

    pub fn num_monomials(letters: usize, degree: usize) -> usize {
        count_monomials(letters + 1, degree)
    }
}

#[cfg(test)]
mod tests {
    use crate::{ConditionerHelper, RecombineInterface};
    use libc::labs;
    use rand;
    use rand::prelude::*;
    use rand_distr;
    use std::error::Error;
    use std::time::{Duration, SystemTime};

    struct Matrix {
        pub data: Vec<f64>,
        pub rows: usize,
        pub cols: usize,
    }

    impl Matrix {
        fn new(rows: usize, cols: usize) -> Matrix {
            let mut rng = rand::thread_rng();
            let mut dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();

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
            no_points: usize,
            indices: &[usize],
            weights: &[f64],
        ) -> Vec<f64> {
            let mut result = vec![0.0f64; self.cols];
            for (p, w) in indices.iter().zip(weights.iter()) {
                for j in 0..self.cols {
                    result[j] = self.data[p * self.cols + j];
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
        (diff / (t1 + t2))
    }

    struct Output {
        num_kept_points: usize,
        points: Matrix,
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
            let num_kept_points = super::monomials::num_monomials(cols, degree);
            dbg!(num_kept_points);
            Self {
                data,
                degree,
                weights,
                output: Output {
                    num_kept_points,
                    points: Matrix::new(num_kept_points, cols),
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
            helper: &ConditionerHelper,
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
                let out_row = &mut output[j * depth_of_vector..(j + 1) * depth_of_vector];
                if self.degree == 1 {
                    out_row[0] = 1.0;
                    for i in 0..self.data.cols {
                        out_row[i + 1] = if max_buf[i] == min_buf[i] {
                            0.0f64
                        } else {
                            (2.0 * row[i] - (min_buf[i] + max_buf[i])) / (max_buf[i] - min_buf[i])
                        };
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
            debug_assert!(weights.len() <= self.output.weights.len());
            debug_assert!(locs.len() <= self.output.locs.len());
            self.output.locs.copy_from_slice(locs);
            self.output.weights.copy_from_slice(weights);
        }
    }

    struct SimpleTimer<'a> {
        start: SystemTime,
        out: &'a mut Duration,
    }

    impl<'a> SimpleTimer<'a> {
        fn new(out: &'a mut Duration) -> Self {
            SimpleTimer {
                start: SystemTime::now(),
                out,
            }
        }
    }

    impl<'a> Drop for SimpleTimer<'a> {
        fn drop(&mut self) {
            *self.out = SystemTime::now().duration_since(self.start).unwrap();
        }
    }

    #[test]
    fn basic_test() {
        const NUM_POINTS: usize = 10000;
        const DIMENSION: usize = 100;

        let m = Matrix::new(NUM_POINTS, DIMENSION);
        let indices: Vec<usize> = (0..NUM_POINTS).collect();

        let weights = vec![1.0f64 / (NUM_POINTS as f64); NUM_POINTS];

        let mean2 = m.weighted_indexed_mean(NUM_POINTS, &indices, &weights);

        let mut interface = TestRecombineInterface::new(m, 1, weights);

        let mut duration = Duration::new(0, 0);
        {
            SimpleTimer::new(&mut duration);
            super::recombine(&mut interface).unwrap();
        }

        println!("Test took {}s", duration.as_secs_f64());
        let mean3 = interface.output.points.weighted_indexed_mean(
            NUM_POINTS,
            &interface.output.locs,
            &interface.output.weights,
        );

        let comp = compare(&mean2, &mean3);
        dbg!(&comp);
        assert!(comp < 1.0e-12);
    }
}
