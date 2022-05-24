#[allow(non_snake_case)]
mod Cinterface;
mod errors;
mod recombine_interface;
mod reduction_tool;
mod reweight;
mod tree_buffer_helper;

extern crate blas_src;
extern crate lapack_src;


use std::error::Error;
use rayon::ThreadPoolBuilder;

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
        ThreadPoolBuilder::new().num_threads(2).build_global()?;

        let reduction_tool = SVDReductionTool::new();
        let mut weights: Vec<f64> = vec![0.0f64; helper.num_trees()];

        let degree = helper.degree();
        let mut points = vec![0.0f64; helper.num_trees() * degree];
        helper.update_points_buffer(&mut points, &mut weights);

        debug_assert!(weights.iter().all(|v| *v >= 0.0 && *v <= 1.0));

        let mut num_lapack_calls = 0usize;

        let mut done = false;
        while !done {
            let mut max_set = reduction_tool.move_mass(&mut weights, &mut points, degree)?;

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
                let (&to_split, &to_split_pos) = helper
                    .tree_position
                    .iter()
                    .next_back()
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

                    // dbg!(to_split, split_left, split_right);
                    // dbg!(helper.weight(split_left), helper.weight(split_right), helper.weight(to_split));

                    weights[to_go_position] = weights[to_split_pos] * helper.weight(split_left) / helper.weight(to_split);
                    weights[to_split_pos] *= (helper.weight(split_right) / helper.weight(to_split));

                    debug_assert!(weights[to_go_position] <= 1.0 && weights[to_go_position] >= 0.0);
                    debug_assert!(weights[to_split_pos] <= 1.0 && weights[to_split_pos] >= 0.0);


                    for i in 0..degree {
                        points[to_go_position*degree + i] = helper.points()[split_left*degree + i];
                        points[to_split_pos*degree + i] = helper.points()[split_right*degree + i];
                    }
                }
            }

            debug_assert!(max_set.is_empty());
            helper.repack_buffer(&mut points, &mut weights);
            num_lapack_calls = reduction_tool.num_linalg_calls();
        }

        let mut locs: Vec<usize> = Vec::with_capacity(helper.tree_position.len());
        let mut out_weights: Vec<f64> = Vec::with_capacity(helper.tree_position.len());
        for (i, wi) in helper.tree_position {
            locs.push(i);
            out_weights.push(weights[wi]);
        }

        interface.set_output(&locs, &out_weights);
    }

    Ok(())
}

pub mod monomials {

    fn count_monomials(letters: usize, degree: usize) -> usize {
        if letters == 0 && degree > 0 {
            0
        } else {
            let mut ans = 1;
            for j in 1..letters {
                ans *= j + degree;
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
