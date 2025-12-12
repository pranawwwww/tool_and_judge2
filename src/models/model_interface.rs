use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use serde::{Deserialize, Serialize};

use crate::{
    config::{ApiModel, Model},
    models::{
        backend::ModelBackend, function_name_mapper::FunctionNameMapper,
        gpt5_interface::Gpt5Interface,
    },
    tool_bfcl_decl::BfclFunctionDef, tool_error_analysis::EvaluationError, tool_file_models::ToolCallParsingResult,
};


#[async_trait::async_trait]
pub trait ModelInterface: Send + Sync {
    async fn generate_tool_call_async(
        &self,
        backend: Arc<dyn ModelBackend>,
        raw_functions: Vec<BfclFunctionDef>,
        user_question: String,
        prompt_passing_in_english: bool,
        name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    ) -> String;

    async fn translate_tool_question_async(
        &self,
        backend: Arc<dyn ModelBackend>,
        user_question: String,
    ) -> String;

    async fn translate_tool_answer_async(
        &self,
        backend: Arc<dyn ModelBackend>,
        parameter_value: String,
    ) -> String;

    fn postprocess_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    ) -> ToolCallParsingResult;
}

pub fn get_model_interface(model: Model) -> Arc<dyn ModelInterface> {
    match model {
        Model::Api(api_model) => match api_model {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => Arc::new(Gpt5Interface),
            _ => {
                unimplemented!("API model interfaces other than GPT-5 are not implemented yet.")
            }
        },
        Model::Local(_) => {
            unimplemented!("Local model interfaces are not implemented yet.")
        }
    }
}
