pub struct TreeBufferHelper<'a> {
    data: &'a [f64],
    pub no_trees: usize,
    pub initial_no_leaves: usize,
}

impl<'a> TreeBufferHelper<'a> {
    pub fn new(data: &'a [f64], no_trees: usize, initial_no_leaves: usize) -> Self {
        Self {
            data,
            no_trees,
            initial_no_leaves,
        }
    }
}
