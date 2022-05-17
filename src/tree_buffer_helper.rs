use std::collections::BTreeMap;
use std::error::Error;
use std::ops::Index;
use std::ops::IndexMut;

use crate::recombine_interface::{ConditionerHelper, RecombineInterface};

pub struct TreeBufferHelper {
    data: Vec<f64>,
    weights: Vec<f64>,
    degree: usize,
    no_trees: usize,
    initial_no_leaves: usize,
    current_roots: BTreeMap<usize, usize>,
    tree_position: BTreeMap<usize, usize>,
}

fn insert_leaf_data(
    data: &dyn RecombineInterface,
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

impl TreeBufferHelper {
    pub fn new(
        interface: &dyn RecombineInterface, // data: &'a mut [f64],
                                            // weights: &mut [f64],
                                            // no_trees: usize,
                                            // initial_no_leaves: usize,
                                            // degree: usize
    ) -> Self {
        let no_points = interface.points_in_cloud();
        let degree = interface.degree();
        let mut data = vec![f64::NAN; 2 * no_points * degree];
        let mut weights = Vec::with_capacity(2 * no_points);

        let curr_weights = interface.weights();

        debug_assert!(degree == data.len() / weights.capacity());

        let initial_no_leaves = no_points;
        let no_trees = usize::min(2 * degree, no_points);
        let buffer_end = 2 * initial_no_leaves - no_trees;

        for i in initial_no_leaves..buffer_end {
            let left_parent = (i - initial_no_leaves) * 2;
            let right_parent = left_parent + 1;

            let left_weight = curr_weights[left_parent];
            let right_weight = curr_weights[right_parent];
            let sum = left_weight + right_weight;
            weights[i] = sum;

            let (lower, upper) = data.split_at_mut(initial_no_leaves);

            let lhs = &lower[left_parent * degree..(left_parent + 1) * degree];
            let rhs = &lower[right_parent * degree..(right_parent + 1) * degree];
            let out = &mut upper[(i * degree)..((i + 1) * degree)];

            if left_weight <= right_weight {
                let weight = left_weight / sum;
                for j in 0..degree {
                    out[i + j] = lhs[j] * weight + rhs[j] * (1.0f64 - weight);
                }
            } else {
                let weight = right_weight / sum;
                for j in 0..degree {
                    out[i + j] = lhs[j] * (1.0f64 - weight) + rhs[j] * weight;
                }
            }
        }

        let mut tree_position = BTreeMap::new();
        let mut current_roots = BTreeMap::new();
        for i in 0..no_trees {
            let root = i + buffer_end - no_trees;
            current_roots.insert(i, root);
            tree_position.insert(root, i);
        }

        Self {
            data,
            weights,
            degree,
            no_trees,
            initial_no_leaves,
            current_roots,
            tree_position,
        }
    }

    pub fn is_leaf(&self, idx: usize) -> bool {
        idx < self.initial_no_leaves
    }
    pub fn last_index(&self) -> usize {
        2 * self.initial_no_leaves - self.no_trees
    }
    pub fn is_node(&self, idx: usize) -> bool {
        idx < self.last_index()
    }
    pub fn is_root(&self, idx: usize) -> bool {
        self.parent(idx) == self.last_index()
    }
    pub fn parent(&self, idx: usize) -> usize {
        usize::min(self.initial_no_leaves + (idx / 2), self.last_index())
    }
    pub fn left_index(&self, idx: usize) -> usize {
        (idx - self.initial_no_leaves) * 2
    }
    pub fn right_index(&self, idx: usize) -> usize {
        let lft = self.left_index(idx);
        lft + 1
    }

    pub fn left(&self, idx: usize) -> &[f64] {
        let i = self.left_index(idx);
        &self[i]
    }

    pub fn right(&self, idx: usize) -> &[f64] {
        let i = self.right_index(idx);
        &self[i]
    }

    pub fn weight(&self, idx: usize) -> f64 {
        self.weights[idx]
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn points(&self) -> &[f64] {
        &self.data[..self.initial_no_leaves]
    }
    pub fn num_points(&self) -> usize {
        self.initial_no_leaves
    }
    pub fn num_trees(&self) -> usize {
        self.no_trees
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn clear_weights(&mut self) {
        self.weights.clear();
    }

    pub fn resize_weights(&mut self, new_size: usize) {
        self.weights.resize(new_size, 0.0f64)
    }

    pub fn weights_mut(&mut self) -> &mut [f64] {
        &mut self.weights
    }

    pub fn current_roots(&self) -> &BTreeMap<usize, usize> {
        &self.current_roots
    }
    pub fn current_roots_mut(&mut self) -> &BTreeMap<usize, usize> {
        &mut self.current_roots
    }

    pub fn pop_index(&mut self, idx: usize) -> Option<(usize, usize)> {
        let root = self.current_roots.remove(&idx);
        if let Some(i) = root {
            if let Some(v) = self.tree_position.remove(&i) {
                Some((i, v))
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn pop_root_index(&mut self, idx: usize) -> Option<usize> {
        self.tree_position
            .remove(&idx)
            .map(|k| match self.current_roots.remove(&k) {
                Some(_) => Some(k),
                None => None,
            })
            .flatten()
    }

    pub fn tree_position(&self, idx: usize) -> Option<usize> {
        self.tree_position.get(&idx).map(|&x| x)
    }

    pub fn insert_root(&mut self, key: usize, value: usize) {
        let _ = self.current_roots.insert(key, value);
    }
    pub fn insert_tree_pos(&mut self, key: usize, value: usize) {
        let _ = self.tree_position.insert(key, value);
    }
}

impl Index<usize> for TreeBufferHelper {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.degree..(index + 1) * self.degree]
    }
}

impl IndexMut<usize> for TreeBufferHelper {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.degree..(index + 1) * self.degree]
    }
}
