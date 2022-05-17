use std::error::Error;
use std::fmt;
use std::fmt::Formatter;

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum RecombineError {
    LinearAlgebraError(String),
    InvalidTreeIndex(String),
}

impl fmt::Display for RecombineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use RecombineError::*;
        match &self {
            LinearAlgebraError(ref message) => {
                write!(f, "linear algebra error: {}", message)
            }
            InvalidTreeIndex(ref message) => {
                write!(f, "invalid tree index error: {}", message)
            }
            _ => {
                write!(f, "unknown error")
            }
        }
    }
}

impl Error for RecombineError {}
