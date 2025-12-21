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
            AddNoiseMode, ApiModel, JudgeConfig, JudgeExperiment, Language, LocalModel, Model,
            ToolConfig, ToolExperiment, TranslateMode, TranslateOption,
        },
        judge::{
            concatenate_datasets::concatenate_perplexity_datasets,
            concatenate_datasets::concatenate_preference_datasets,
        },
        judge::{
            dispatch_results::dispatch_perplexity_results,
            dispatch_results::dispatch_preference_results,
        },
        models::backend::GenerationResult,
        tool::{
            passes::pass_pre_translate::{
                pass_pre_translation_aggregated_questions_input_file_path,
                pass_pre_translation_aggregated_questions_output_file_path,
                pass_pre_translation_prepare_aggregated_questions,
                pass_pre_translation_dispatch_results,
            },
        }
    };
}
