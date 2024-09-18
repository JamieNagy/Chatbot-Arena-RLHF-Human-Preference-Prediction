
# Preference-Aware LLM Fine-Tuning with Gemma-2 and LoRA

This project demonstrates fine-tuning the **Gemma-2 9b Instruct** model with LoRA adapters, using a 4-bit quantized setup. The focus is on fine-tuning for **sequence classification** tasks with preference-aware learning techniques. The implementation leverages **PEFT**, **LoRA**, and **transformers** for efficient model fine-tuning and inference.

## Features
- **4-bit Quantization**: Optimize large models for faster training and inference.
- **LoRA Adapters**: Efficiently fine-tune the attention layers while reducing the number of trainable parameters.
- **Preference-Aware Learning**: Fine-tune models to handle scenarios where different responses have varying levels of preference.
- **Multi-GPU Support**: Use multiple GPUs during inference for faster results.


### Training

The training script `training.ipynb` fine-tunes the model on a custom dataset. Key configurations are handled through the `Config` class, which includes batch sizes, learning rates, and LoRA parameters. Below is a snippet of the key configuration:

```python
@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "unsloth/gemma-2-9b-it-bnb-4bit"
    max_length: int = 3120
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 6
    lr: float = 1e-4
    n_epochs: int = 1
    lora_r: int = 128
    lora_alpha: float = lora_r * 1
    freeze_layers: int = 16
```

#### LoRA Configuration

Fine-tune the self-attention layers using **LoRA** with the following configuration:

```python
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj","o_proj","gate_proj"],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
```

#### Model Training

The model is prepared for 4-bit training using **PEFT** and **BitsAndBytes** for optimized memory usage.

```python
trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
```

### Inference

The `inference.ipynb` notebook loads the fine-tuned model and performs inference on test data. It supports **Test-Time Augmentation (TTA)** to improve accuracy by swapping model responses.

Example configuration for inference:

```python
@dataclass
class Config:
    gemma_dir = '/path/to/gemma-2'
    lora_dir = '/path/to/lora-checkpoint'
    max_length = 2048
    batch_size = 4
```

Perform inference with multi-GPU support:

```python
@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size, max_length):
    # Perform inference on the dataset
```

## Results

- **Evaluation Set Log Loss**: 0.9371
- **Public LB Log Loss**: 0.941
- **Inference Time**: ~4 hours for max_length=2048

## Model

Utilize **Gemma-2 9b** model fine-tuned with **LoRA** adapters. The model is loaded and split across GPUs for efficient inference. 

```python
model_0 = Gemma2ForSequenceClassification.from_pretrained(cfg.gemma_dir, device_map=device_0)
model_1 = Gemma2ForSequenceClassification.from_pretrained(cfg.gemma_dir, device_map=device_1)
model_0 = PeftModel.from_pretrained(model_0, cfg.lora_dir)
model_1 = PeftModel.from_pretrained(model_1, cfg.lora_dir)
```
