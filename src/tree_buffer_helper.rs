use std::collections::BTreeMap;
use std::error::Error;
use std::ops::Index;
use std::ops::IndexMut;
use std::thread::current;

use crate::recombine_interface::{ConditionerHelper, RecombineInterface};

pub struct TreeBufferHelper {
    pub data: Vec<f64>,
    pub weights: Vec<f64>,
    degree: usize,
    no_trees: usize,
    initial_no_leaves: usize,
    pub current_roots: BTreeMap<usize, usize>,
    pub tree_position: BTreeMap<usize, usize>,
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
        let mut weights = vec![0.0f64; 2 * no_points];

        dbg!(no_points, degree);

        let helper = ConditionerHelper {
            no_points_to_be_processed: no_points,
        };

        interface.expand_points(&mut data, &helper).unwrap();

        debug_assert!(degree == data.len() / weights.capacity());

        weights[0..no_points].copy_from_slice(interface.weights());

        let initial_no_leaves = no_points;
        let no_trees = usize::min(2 * degree, no_points);
        let buffer_end = initial_no_leaves - no_trees;
        dbg!(
            initial_no_leaves,
            no_trees,
            buffer_end,
            weights.len(),
            data.len()
        );

        for i in 0..buffer_end {
            let left_parent = i * 2;
            let right_parent = left_parent + 1;
            // dbg!(i, left_parent, right_parent);

            let left_weight = weights[left_parent];
            let right_weight = weights[right_parent];
            let sum = left_weight + right_weight;
            weights[i + initial_no_leaves] = sum;

            debug_assert!(right_parent < i + initial_no_leaves);
            let (lower, out) = data.split_at_mut((i + initial_no_leaves) * degree);

            let lhs = &lower[left_parent * degree..(left_parent + 1) * degree];
            let rhs = &lower[right_parent * degree..(right_parent + 1) * degree];

            if left_weight <= right_weight {
                let weight = left_weight / sum;
                for j in 0..degree {
                    out[j] = lhs[j] * weight + rhs[j] * (1.0f64 - weight);
                }
            } else {
                let weight = right_weight / sum;
                for j in 0..degree {
                    out[j] = lhs[j] * (1.0f64 - weight) + rhs[j] * weight;
                }
            }
        }

        let mut tree_position = BTreeMap::new();
        let mut current_roots = BTreeMap::new();
        for i in 0..no_trees {
            let root = i + 2 * buffer_end;
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

    pub fn repack_buffer(&mut self, points: &mut Vec<f64>, weights: &mut Vec<f64>) {
        let mut current_roots_new = BTreeMap::new();
        let mut tree_position_new = BTreeMap::new();
        let mut weights_new = vec![0.0f64; self.current_roots.len()];
        let mut points_new = vec![0.0f64; self.current_roots.len() * self.degree];

        for (i, (node, idx)) in self.current_roots.iter().enumerate() {
            tree_position_new.insert(*idx, i);
            current_roots_new.insert(i, *idx);
            weights_new[i] = weights[*node];
            points_new[i * self.degree..(i + 1) * self.degree]
                .copy_from_slice(&points[node * self.degree..(node + 1) * self.degree]);
        }

        let length = self.current_roots.len();

        self.current_roots = current_roots_new;
        self.tree_position = tree_position_new;
        *weights = weights_new;
        *points = points_new;
    }

    pub fn points_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn update_points_buffer(&mut self, buffer: &mut [f64]) {
        debug_assert!(buffer.len() >= self.no_trees * self.degree);
        for (root_i, buf_i) in self.tree_position.iter() {
            buffer[buf_i * self.degree..(buf_i + 1) * self.degree]
                .copy_from_slice(&self.data[root_i * self.degree..(root_i + 1) * self.degree]);
        }
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
