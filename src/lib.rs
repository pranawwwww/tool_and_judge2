use pyo3::prelude::*;

pub mod config;
pub mod judge;
pub mod models;
pub mod one_entry_map;
pub mod single_or_list;
pub mod tool;
pub mod utils;

#[pymodule]
pub mod codebase_rs {
    use pyo3::{Py, pyfunction, types::PyList};
    #[pyfunction] // Inline definition of a pyfunction, also made available to Python
    async fn tool_run_async(configs: Py<PyList>, num_gpus: usize) {
        println!("Running tool from Rust!\n");
        unimplemented!();
        // tool_run::tool_run_async(configs, num_gpus).await;
    }

    // use crate::tool_run;

    #[pymodule_export]
    use super::{
        config::{
            AddNoiseMode, ApiModel, Language, LocalModel, Model, ToolConfig, ToolExperiment, JudgeConfig, JudgeExperiment, 
            TranslateMode, TranslateOption,
        },
        models::backend::GenerationResult,
        judge::{concatenate_datasets::concatenate_perplexity_datasets, concatenate_datasets::concatenate_preference_datasets,},
        judge::{dispatch_results::dispatch_perplexity_results, dispatch_results::dispatch_preference_results,},
    };
}
