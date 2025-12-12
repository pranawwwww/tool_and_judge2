use pyo3::prelude::*;

pub mod category_cache;
pub mod config;
pub mod models;
pub mod tool_bfcl_decl;
pub mod tool_error_analysis;
pub mod tool_run;
pub mod util;

#[pymodule]
pub mod codebase_rs {
    use pyo3::{Bound, Py, pyfunction, types::PyList};
    #[pyfunction] // Inline definition of a pyfunction, also made available to Python
    async fn tool_run_async(configs: Py<PyList>, num_gpus: usize) {
        println!("Running tool from Rust!\n");
        tool_run::tool_run_async(configs, num_gpus).await;
    }

    use crate::tool_run;

    #[pymodule_export]
    use super::{
        config::{
            AddNoiseMode, ApiModel, Language, LocalModel, Model, ToolConfig, TranslateMode,
            TranslateOption,
        },
        models::backend::GenerationResult,
    };
}
