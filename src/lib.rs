mod Cinterface;

use std::error::Error;
use std::ops::Index;
pub use Cinterface::*;


#[repr(C)]
pub struct ConditionerHelper {
    pub no_points_to_be_processed: usize
}

pub trait RecombineInterface
{
    fn points_in_cloud(&self) -> usize;
    fn expand_func(&self) -> Box<dyn FnMut(&dyn IntoIter<Item=&[f64]>, &mut [f64], &ConditionerHelper)>;
    fn expand_points(&self, output: &mut [f64], helper: &ConditionerHelper) -> Result<(), Box<dyn Error>>;
    fn degree(&self) -> usize;
    fn weights(&self) -> &[f64];

}

pub fn recombine(interface: &impl RecombineInterface)
{
    let (num_points, mut points_buffer, mut weights_buffer) = insert_leaf_data(interface);
    let degree = points_buffer.len() / weights_buffer.len();
    debug_assert!(interface.degree() == degree);


}


fn insert_leaf_data(data: &impl RecombineInterface) -> Result<(usize, Vec<f64>, Vec<f64>), Box<dyn Error>>
{
    let n_points_in = data.points_in_cloud();

    let mut array_points_buffer = vec![f64::NAN; 2*n_points_in*data.degree()];
    let weights_buffer = Vec::from(data.weights());
    let helper = ConditionerHelper { no_points_to_be_processed: n_points_in};

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
