use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::{
    config::{ApiModel, LocalModel, Model},
    models::{
        backend::ModelBackend,
        deepseek_interface::DeepSeekInterface,
        function_name_mapper::{self, FunctionNameMapper},
        gpt5_interface::Gpt5Interface,
        llama3_1_interface::Llama3_1Interface,
    },
    tool::{
        bfcl_formats::{BfclFunctionDef, BfclOutputFunctionCall},
        error_analysis::EvaluationError,
    },
};

#[async_trait::async_trait]
pub trait ModelInterface: Send + Sync {
    // async fn generate_tool_call_async(
    //     &self,
    //     backend: Arc<ModelBackend>,
    //     raw_functions: Vec<BfclFunctionDef>,
    //     user_question: String,
    //     prompt_passing_in_english: bool,
    //     name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    // ) -> String;

    async fn translate_tool_question_async(
        &self,
        backend: Arc<ModelBackend>,
        user_question: String,
    ) -> String;

    async fn translate_tool_answer_async(
        &self,
        backend: Arc<ModelBackend>,
        parameter_value: String,
    ) -> String;

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError>;

    fn generate_tool_definitions(
        &self,
        bfcl_tool_definitions: &Vec<BfclFunctionDef>,
        function_name_mapper: &FunctionNameMapper,
    ) -> serde_json::Value;
}

pub fn get_model_interface(model: Model) -> Arc<dyn ModelInterface> {
    match model {
        Model::Api(api_model) => match api_model {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => Arc::new(Gpt5Interface),
            ApiModel::DeepSeek => Arc::new(DeepSeekInterface),
            _ => {
                unimplemented!(
                    "API model interfaces other than GPT-5 and DeepSeek are not implemented yet."
                )
            }
        },
        Model::Local(local_model) => match local_model {
            LocalModel::Llama3_1_8B | LocalModel::Llama3_1_70B => Arc::new(Llama3_1Interface),
            _ => {
                unimplemented!(
                    "Local model interfaces other than Llama 3.1 are not implemented yet."
                )
            }
        },
    }
}
