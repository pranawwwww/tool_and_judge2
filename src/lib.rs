use pyo3::prelude::*;

pub mod config;
pub mod tool_error_analysis;
pub mod tool_run_impl;
pub mod util;
pub mod category_cache;
pub mod models;

#[pymodule]
pub mod codebase_rs {
    use pyo3::{Bound, pyfunction, types::PyList};
    #[pyfunction] // Inline definition of a pyfunction, also made available to Python
    fn tool_run<'py>(configs: &Bound<'py, PyList>, num_gpus: usize) {
        println!("Running tool from Rust!\n");
        tool_run_impl::tool_run_impl(configs, num_gpus);
    }

    use crate::tool_run_impl;

    #[pymodule_export]
    use super::config::{
        AddNoiseMode, ApiModel, Language, LocalModel, Model, TranslateMode, TranslateOption, ToolConfig,
    };
}
