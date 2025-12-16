use pyo3::{Py, PyAny, Python, types::PyAnyMethods};

use crate::{
    config::{LocalModel, Model},
    models::backend::ModelBackend,
};

pub struct VllmBackend {
    pub model: LocalModel,
    pub engine: Py<PyAny>,
    pub tokenizer: Py<PyAny>,
}

impl VllmBackend {
    pub fn new(model: LocalModel, num_gpus: usize) -> VllmBackend {
        let (engine, tokenizer) = Python::attach(|py| {
            let vllm_backend_module = py
                .import("src_py.vllm_backend")
                .expect("Failed to import src_py.vllm_backend module");

            let create_fn = vllm_backend_module
                .getattr("create_vllm_backend")
                .expect("Failed to get create_vllm_backend function");

            let result = create_fn
                .call1((model.to_string(), num_gpus))
                .expect("Failed to call create_vllm_backend");

            // Result is a tuple (engine, tokenizer)
            let engine = result
                .get_item(0)
                .expect("Failed to get engine from tuple")
                .unbind();
            let tokenizer = result
                .get_item(1)
                .expect("Failed to get tokenizer from tuple")
                .unbind();

            (engine, tokenizer)
        });

        VllmBackend {
            model,
            engine,
            tokenizer,
        }
    }
}
