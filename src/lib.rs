#[allow(non_snake_case)]
mod Cinterface;
mod errors;
mod recombine_interface;
mod reduction_tool;
mod reweight;
mod tree_buffer_helper;

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

    let out_buf = interface.out_buffer();
    if helper.num_points() <= 1 {
        let locs: Vec<usize> = (0..helper.num_points()).collect();
        interface.set_output(&locs, &helper.points(), &helper.weights());
    } else {
        let reduction_tool = SVDReductionTool;
        let mut weights: Vec<f64> = Vec::new();

        let mut points = helper.points().to_vec();
        let degree = helper.degree();
        let mut num_lapack_calls = 0usize;

        let mut max_set = Vec::<usize>::new();
        let mut done = false;
        while !done {
            reduction_tool.move_mass(helper.weights_mut(), &mut points, &mut max_set, degree)?;

            done = max_set.is_empty();

            while let Some(to_go_position) = max_set.pop() {
                helper.pop_index(to_go_position);
                let to_split = *helper
                    .current_roots()
                    .iter()
                    .next_back()
                    .ok_or(RecombineError::InvalidTreeIndex(
                        "current roots map is empty".into(),
                    ))?
                    .0;
                if !helper.is_leaf(to_split) {
                    let to_split_pos =
                        helper
                            .pop_root_index(to_split)
                            .ok_or(RecombineError::InvalidTreeIndex(
                                "last element invalid?".into(),
                            ))?;

                    let split_left = helper.left_index(to_split);
                    let split_right = helper.right_index(to_split);
                    helper.insert_root(to_go_position, split_left);
                    helper.insert_tree_pos(split_left, to_go_position);
                    helper.insert_root(to_split_pos, split_right);
                    helper.insert_tree_pos(split_right, to_split);

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

            num_lapack_calls = reduction_tool.num_linalg_calls();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
