#[allow(non_snake_case)]
mod Cinterface;
mod reduction_tool;
mod tree_buffer_helper;
mod reweight;

use std::error::Error;

pub use Cinterface::*;


#[repr(C)]
pub struct ConditionerHelper {
    pub no_points_to_be_processed: usize,
}

pub trait RecombineInterface {
    fn points_in_cloud(&self) -> usize;
    fn expand_points(
        &self,
        output: &mut [f64],
        helper: &ConditionerHelper,
    ) -> Result<(), Box<dyn Error>>;
    fn degree(&self) -> usize;
    fn weights(&self) -> &[f64];

    fn out_buffer(&mut self) -> &mut [f64];
    fn set_output(&mut self, locs: &[usize], weights: &[f64], points: &[f64]);
}

#[no_mangle]
pub fn recombine(interface: &mut impl RecombineInterface) -> Result<(), Box<dyn Error>> {
    let (num_points, mut points_buffer, mut weights_buffer) = insert_leaf_data(interface)?;
    let degree = points_buffer.len() / weights_buffer.len();
    debug_assert!(interface.degree() == degree);

    let out_buf = interface.out_buffer();
    if num_points <= 1 {
        let locs: Vec<usize> = (0..num_points).collect();
        interface.set_output(&locs, &points_buffer, &weights_buffer);
    } else {
        let max_points = 2 * degree;
    }

    Ok(())
}

fn insert_leaf_data(
    data: &impl RecombineInterface,
) -> Result<(usize, Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let n_points_in = data.points_in_cloud();

    let mut array_points_buffer = vec![f64::NAN; 2 * n_points_in * data.degree()];
    let weights_buffer = Vec::from(data.weights());
    let helper = ConditionerHelper {
        no_points_to_be_processed: n_points_in,
    };

    data.expand_points(&mut array_points_buffer, &helper)?;

    Ok((n_points_in, array_points_buffer, weights_buffer))
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
