use std::error::Error;

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

    fn set_output(&mut self, locs: &[usize], weights: &[f64]);
}
