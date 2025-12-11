use crate::{
    config::{ApiModel, Model},
    models::{
        backend::ModelBackend, function_name_mapper::FunctionNameMapper,
        gpt5_interface::Gpt5Interface,
    },
    tool_bfcl_decl::BfclFunctionDef,
};

#[async_trait::async_trait]
pub trait ModelInterface: Send + Sync {
    async fn generate_tool_call_async(
        &self,
        backend: &dyn ModelBackend,
        raw_functions: &Vec<BfclFunctionDef>,
        user_question: &str,
        prompt_passing_in_english: bool,
        name_mapper: &mut FunctionNameMapper,
    ) -> String;

    async fn translate_tool_question_async(
        &self,
        backend: &dyn ModelBackend,
        user_question: &str,
    ) -> String;
}

pub fn get_model_interface(model: Model) -> Box<dyn ModelInterface> {
    match model {
        Model::Api(api_model) => match api_model {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => Box::new(Gpt5Interface),
            _ => {
                unimplemented!("API model interfaces other than GPT-5 are not implemented yet.")
            }
        },
        Model::Local(_) => {
            unimplemented!("Local model interfaces are not implemented yet.")
        }
    }
}
