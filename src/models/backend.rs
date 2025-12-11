use std::collections::HashMap;

use lazy_static::lazy_static;
use pyo3::{Bound, pyclass};

use crate::{config::Model, models::api_backend::ApiBackend};


#[pyclass]
#[derive(Clone)]
pub struct GenerationResult {
    pub generated_text: String,
    pub generated_token_ids: Option<Vec<u32>>,
    pub logits: Option<Vec<HashMap<u32, f32>>>,
}

#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    async fn generate_async(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        return_logprobs: bool,
    ) -> GenerationResult;

    async fn forward_async(
        &self,
        prompt: &str,
        max_length: usize,
    ) -> (Vec<u32>, Vec<HashMap<u32, f32>>);
    async fn shutdown(&self);

    fn get_tokenizer<'py>(&self) -> Bound<'py, pyo3::PyAny>;

    fn get_model_info(&self) -> Model;
}

lazy_static! {
    pub static ref GLOBAL_MAIN_BACKEND: tokio::sync::RwLock<Option<Box<dyn ModelBackend>>> =
        tokio::sync::RwLock::new(None);
    pub static ref GLOBAL_ASSIST_BACKEND: tokio::sync::RwLock<Option<Box<dyn ModelBackend>>> =
        tokio::sync::RwLock::new(None);
}
pub enum WhichBackend {
    Main,
    Assist,
}


// def create_backend(
//     backend_type: str,
//     model_name: str,
//     api_key: Optional[str] = None,
//     base_url: Optional[str] = None,
//     device: str = "cuda",
//     num_gpus: int = 1,
//     use_cache: bool = True,
//     instance_name: str = "default",
//     **kwargs
// ) -> ModelBackend:

pub async fn get_or_create_backend(
    model: Model,
    which: WhichBackend,
    num_gpus: usize,
) -> tokio::sync::RwLockReadGuard<'static, Option<Box<dyn ModelBackend>>> {
    let backend_reference = match which {
        WhichBackend::Main => &*GLOBAL_MAIN_BACKEND,
        WhichBackend::Assist => &*GLOBAL_ASSIST_BACKEND,
    };
    let mut backend_guard = Some(backend_reference.read().await);
    // keep assigning to backend_guard until the backend is available while holding the read lock
    loop {
        let mut needs_to_create = true;
        if let Some(backend) = backend_guard.as_ref().unwrap().as_ref() {
            let existing_model = backend.get_model_info();
            if existing_model == model {
                needs_to_create = false;
            }
        }
        if !needs_to_create {
            break;
        }
        println!("Global backend not available, or model mismatch. Creating new backend for model {:?}...", model);
        // drop the read guard before acquiring the write lock
        backend_guard.take();
        // We assume that no other task is trying to create the backend at the same time.
        // Even if they are, the worst that can happen is that we create the backend multiple times.
        let mut write_backend_guard = backend_reference.write().await;
        // create the new backend and assign it to the global variable
        let new_backend: Box<dyn ModelBackend> = match &model {
            Model::Api(api_model) => Box::new(ApiBackend::new(*api_model)),
            Model::Local(_local_model) => {
                // Implement local model backend creation here
                unimplemented!()
            }
        };
        *write_backend_guard = Some(new_backend);
        // drop the write guard before acquiring the read guard
        drop(write_backend_guard);
        // re-acquire the read guard
        backend_guard = Some(backend_reference.read().await);
    }
    backend_guard.expect("Backend should end up having a value in both branches")
}

// # =============================================================================
// # Model Backend Abstract Base Class
// # =============================================================================

// class ModelBackend(ABC):
//     """
//     Abstract base class for model backends that handle inference.

//     Different backends (API, HuggingFace, vLLM) implement this interface to provide
//     low-level inference primitives. The backend's job is ONLY to execute inference,
//     not to format prompts or parse outputs (that's the interface's job).

//     Backends should automatically batch concurrent async requests for efficiency.
//     """

//     @abstractmethod
//     def generate(
//         self,
//         prompt: str,
//         max_new_tokens: int = 100,
//         temperature: float = 0.0,
//         return_logprobs: bool = False,
//         **kwargs
//     ) -> GenerationResult:
//         """
//         Synchronously generate text from a prompt.

//         Args:
//             prompt: The input prompt text (already formatted by interface)
//             max_new_tokens: Maximum number of new tokens to generate
//             temperature: Sampling temperature (0.0 for greedy decoding, >0 for sampling)
//             return_logprobs: If True, return log probabilities for generated tokens.
//                            Backend MUST provide logprobs when this is True, or raise error.
//             **kwargs: Additional backend-specific parameters

//         Returns:
//             GenerationResult containing generated text and token IDs.
//             If return_logprobs=True, GenerationResult.logits will contain
//             List[Dict[int, float]] with log probabilities.

//         Raises:
//             RuntimeError: If return_logprobs=True but backend fails to provide logprobs
//         """
//         pass

//     @abstractmethod
//     async def generate_async(
//         self,
//         prompt: str,
//         max_new_tokens: int = 100,
//         temperature: float = 0.0,
//         return_logprobs: bool = False,
//         **kwargs
//     ) -> GenerationResult:
//         """
//         Asynchronously generate text from a prompt.

//         The backend may batch multiple concurrent requests internally for efficiency.

//         Args:
//             prompt: The input prompt text (already formatted by interface)
//             max_new_tokens: Maximum number of new tokens to generate
//             temperature: Sampling temperature (0.0 for greedy decoding, >0 for sampling)
//             return_logprobs: If True, return log probabilities for generated tokens.
//                            Backend MUST provide logprobs when this is True, or raise error.
//             **kwargs: Additional backend-specific parameters

//         Returns:
//             GenerationResult containing generated text and token IDs.
//             If return_logprobs=True, GenerationResult.logits will contain
//             List[Dict[int, float]] with log probabilities.

//         Raises:
//             RuntimeError: If return_logprobs=True but backend fails to provide logprobs
//         """
//         pass

//     @abstractmethod
//     async def forward_async(
//         self,
//         prompt: str,
//         max_length: int = 2048,
//         **kwargs
//     ) -> ForwardResult:
//         """
//         Asynchronously run forward pass on a prompt to get logits.

//         This method is used for perplexity calculation and other tasks that
//         require access to the model's output logits.

//         Args:
//             prompt: The input prompt text (already formatted by interface)
//             max_length: Maximum sequence length for tokenization
//             **kwargs: Additional backend-specific parameters

//         Returns:
//             ForwardResult containing logits and input_ids

//         Raises:
//             NotImplementedError: If backend doesn't support forward pass (e.g., API backends)
//         """
//         pass

//     @abstractmethod
//     async def shutdown(self):
//         """
//         Cleanup resources and shutdown the backend.

//         Should be called when inference is complete to properly release resources.
//         """
//         pass

//     def get_tokenizer(self) -> Any:
//         """
//         Get the tokenizer associated with this backend.

//         Returns:
//             Tokenizer object (e.g., HuggingFace tokenizer)

//         Raises:
//             NotImplementedError: If backend doesn't support direct tokenizer access
//         """
//         raise NotImplementedError(
//             f"{self.__class__.__name__} does not provide direct tokenizer access"
//         )
