use crate::{config::{ApiModel, Model}, models::gpt5_interface::Gpt5Interface};




pub trait ModelInterface {
    fn generate(&self, prompt: &str, max_new_tokens: usize) -> String {
        unimplemented!()
    }
}


pub fn get_interface(model: Model) -> Box<dyn ModelInterface> {
    match model {
        Model::Api(api_model) => {
            match api_model {
                ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => {
                    Box::new(Gpt5Interface::new(model))
                },
                _ => {
                    unimplemented!("API model interfaces other than GPT-5 are not implemented yet.")
            }
        },
        Model::Local(_) => {
            unimplemented!("Local model interfaces are not implemented yet.")
        },
    }
}