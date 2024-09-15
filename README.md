
# Preference-Aware LLM Fine-Tuning

## Overview

With the rise of large language models (LLMs), ensuring that AI-generated responses align with human preferences has become a key challenge. This project focuses on building a machine learning model to predict user preferences in head-to-head interactions between two anonymous LLMs. The project leverages cutting-edge techniques such as reinforcement learning from human feedback (RLHF), and parameter-efficient fine-tuning (PEFT), with a particular focus on overcoming common biases.

The goal is to predict which model's response a user will prefer, allowing for the development of more personalized and user-friendly AI-driven chat systems.

## Key Features

- **Multi-Model Inference**: This project employs two distinct models—Gemma-2 and Llama—to predict user preferences. Each model processes the inputs separately, and the final predictions are aggregated to enhance accuracy.
- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA (Low-Rank Adaptation) is used for fine-tuning the models to save memory while maintaining model performance.
- **Bias Mitigation**: The project tackles several known biases in human preference prediction, such as verbosity bias, position bias, and self-enhancement bias.

## Project Workflow

1. **Data Loading and Processing**: The input data, consisting of user prompts and model responses, is pre-processed and tokenized using specialized tokenizers for both the Gemma and Llama models.
2. **Model Inference**: The data is then passed through two fine-tuned models for inference. Both models predict the probability of each response being preferred by the user, and these results are averaged to obtain the final prediction.
3. **Parallel Processing**: The inference is optimized through multi-threading, enabling both models to run in parallel across multiple GPUs, significantly reducing the computational time.
4. **Post-Inference Aggregation**: The results from both models are combined to form the final predictions, ensuring more robust outcomes.

## Model Architecture

- **Gemma-2 Model**: Utilizes GemmaTokenizerFast for tokenization and is fine-tuned using LoRA.
- **Llama Model**: Tokenized using AutoTokenizer and also fine-tuned with LoRA, but utilizing 8-bit precision for memory efficiency.

## Tokenization Strategy

- Tokenization is handled separately for both models, with the option to spread the maximum sequence length among the user prompt and responses.
- Different tokenization formats are applied for Gemma and Llama models to fit their architectures.

## Performance Optimization

- **ThreadPoolExecutor**: Runs both models in parallel, drastically reducing inference time.
- **Mixed Precision Inference**: Inference is performed using `torch.cuda.amp.autocast()` for faster computations without compromising on precision.
- **Efficient Memory Usage**: Models are loaded with memory-efficient settings using BitsAndBytes, reducing the memory footprint while preserving computational power.

## Running the Project

# Unfortunately you can't directly run the code in your end, because the source data is too big to upload to GitHub :(

1. **Data Loading**: Load the input data from a CSV file.
2. **Data Preprocessing**: Process the user prompt and model responses by cleaning and tokenizing the text.
3. **Model Inference**: Run inference using both the Gemma and Llama models in parallel.
4. **Result Aggregation**: Combine the results from both models to generate the final prediction for user preferences.

### Code Snippet Example

```python
@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, tokenizer, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    # Inference logic for processing batches of data
    # Returns the probability scores for model A, model B, and ties.
```


## Conclusion

This project demonstrates how modern techniques like RLHF, PEFT, and multi-threaded inference can be effectively used to predict human preferences in AI-generated conversations. The work done here forms a foundation for developing more intuitive, human-aligned AI systems capable of understanding and catering to user preferences in real-time.

