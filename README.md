# Turn your LLM into a Mafioso Code Explanation Companion
## A slightly more interesting fine-tuning objective.

Train a Language Model capable of explaining Python code and providing explanations with a fun twist inspired by Hollywood mafia classics such as “The Godfather”, “Casino”, and “Goodfellas.

### Fine-Tuning Llama:

The process of fine-tuning LLMs has undergone significant advancements in recent years. With the introduction of the SFTTrainer class in the TRL library by Hugging Face, the developer experience has seen great improvements. Specifically, in the context of tuning LLMs through Quantized Low-Rank Adaptation (QLoRA), a methodical approach is employed to significatly reduce the hardware requirements of these VRAM hungry models. However, the training procedure involves a set of clearly defined and structured steps:


- **Dataset Creation**: Create a suitable dataset organized into Hugging Face Dataset objects.

- **Model Loading and Quantization**: Load the model onto GPU in 4-bit precision using the "bitsandbytes" library.

- **LoRA Configuration**: Define the LoRA (Low-Rank Adaptation) configuration, which depends on the specific problem you're addressing.

- **Training Hyperparameters**: Ultimately, as with every model, the success of fine-tuning depends on selecting appropriate training parameters.

- **SFTTrainer Integration**: Finally, the defined training parameters are used with the SFTTrainer class to initiate the fine-tuning process.


### Dataset Creation:

The dataset is structured as a straightforward JSON file. Each individual sample within the dataset is accessible through its respective key. These samples are organized as dictionaries containing three distinct keys: "prompt," "code," and "mafia_explanation." For our specific objectives, our focus lies solely on the "code" and "mafia_explanation" keys, which contain the essential information we require for our analysis and purposes.

### Model Loading and Quantization:
The subsequent step in the fine-tuning process involves loading the specific model you intend to utilize. As previously indicated, our choice is the "llama2-7b," which constitutes the 7 billion parameter base model. However, it's worth noting that the Hugging Face model hub offers a diverse selection of models for various natural language processing tasks. Feel free to explore and experiment with different models available at https://huggingface.co/models to tailor your fine-tuning process to the specific requirements of your project. 

To ensure that our model undergoes quantization as intended, we'll use the BitsAndBytesConfig, serving as a wrapper class encapsulating various attributes and features available for manipulation when working with a loaded model through bitsandbytes. 

Presently, the BitsAndBytesConfig supports quantization methods such as LLM.int8(), FP4, and NF4. Should additional methods be incorporated into bitsandbytes in the future, corresponding arguments will be introduced to this class to accommodate these extensions.

Notably, bitsandbytes operates as a lightweight wrapper encompassing custom CUDA functions, particularly optimized for 8-bit operations, matrix multiplication (LLM.int8()), and quantization functions. For a deeper understanding of bitsandbytes and its functionalities, further information is available at: https://github.com/TimDettmers/bitsandbytes.

```python
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             quantization_config=bnb_config, 
                                             device_map="auto")
```
### LoRA Configuration:

We don't need to go into too much detail about LoRA because there's already a lot of information out there. If you’re interested feel to read the paper https://arxiv.org/abs/2106.09685. 

Here's the basic idea: the rank of a matrix tells us the number of linearly independent rows or columns it contains. This rank is important because it shows us the smallest amount of space needed to fit all the rows or columns.

Matrices suffering from rank deficiency, characterized by linear dependencies, inherently possess redundancy. In simple terms, this means we can express the same information using fewer dimensions (see https://arxiv.org/abs/2012.13255).

In practical terms, this approach involves training two smaller matrices, A and B, instead of a large weight matrix. When you multiply these two smaller matrices together, you end up with a matrix of the same dimensions as the original weight matrix. The key point here is that your choice of rank for matrices A and B can significantly reduce the number of parameters in the model. However, there's a trade-off because this reduction in “effective” parameters can lead to loss of information. But, as mentioned earlier, the LoRA hypothesis is that this information loss may not be a major concern because a substantial portion of the parameters in the original weight matrix may not contribute significantly to the model's performance. Finally, we will prepare the model for training using the prepare_model_for_kbit_training() method. 

Here is what prepare_mode_for_kbit_training() does:

- It initiates the freezing of all model parameters by setting their gradient requirement to False, effectively preventing updates during the training process.

- For models that aren't quantized using the GPTQ method, it transforms parameters that are originally in formats such as 16-bit or bfloat16 into 32-bit floating-point format (fp32).

- If the model is initially loaded with lower bit precision (like 4-bit or 8-bit) or is quantized, and gradient checkpointing is enabled, it ensures that the inputs to the model necessitate gradients for training. This is achieved either by activating an existing function within the model or by registering a forward hook to the input embeddings to make sure their outputs require gradients.

- It conducts a compatibility check with the provided gradient checkpointing keyword arguments and provides a warning if the model doesn't support them.

- Ultimately, if all conditions align, the function enables gradient checkpointing with the appropriate parameters, thereby optimizing memory usage during training. This preparation proves especially valuable when training larger models or working with hardware that has limited memory resources.
```python
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
                  "q_proj",
                  "up_proj",
                  "o_proj",
                  "k_proj",
                  "down_proj",
                  "gate_proj",
                  "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Gradient checkpointing is a technique used to trade off memory usage for 
# computation time during backpropagation
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
```
### Training Hyperparameters

The following step is rather straightforward. Here, we simply configure the training settings. The values employed in this step are commonly used as standard starting points, but it's important to note that they may need adjustment for optimal performance based on your specific requirements.


```python

# set bf16, tf32 to True with an A10/A100
#                   False with T4 in colab
bf16=True,
tf32=True,
fp16 = False

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs= 3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,
    fp16=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    eval_steps=5,
    do_eval = True,
    disable_tqdm=False
)
```
### SFTTrainer Integration

Lastly, we can proceed with the setup of the Trainer. The Trainer class offers a comprehensive API for PyTorch training, suitable for most common scenarios. The SFTTrainer is an extension of the original transformers.Trainer class, designed to accommodate the direct initialization of the model for Parameter-Efficient Fine-Tuning (PEFT) by accepting the peft_config parameter.

You can make use of the formatting_func to structure your dataset samples in a way that fits your requirements. There are two options:  In case you’re using an already fine-tuned model you must stick with the existing prompt format. Alternatively, if you're working with the base model, you can define a prompt format that suits your specific needs. 

In or case the formatting_func is fairly simple. We add an instruction that should serve as a system prompt followed by the code and explanation blocks. Lastly, we can proceed with the setup of the Trainer. The Trainer class offers a comprehensive API for PyTorch training, suitable for most common scenarios. The SFTTrainer is an extension of the original transformers.Trainer class, designed to accommodate the direct initialization of the model for Parameter-Efficient Fine-Tuning (PEFT) by accepting the peft_config parameter.

```python

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset= dataset, 
    peft_config=config,
    formatting_func  = format_instruction,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
    max_seq_length = 2048   
    )

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
```

### Training
```python
trainer.train()
```
Training will take about 20 min on an A10 instance and cost around 0.4 $.During training, you should observe a relatively gradual and consistent decrease in the training loss. It's essential to note that for fine-tuning intended for production use, you would typically incorporate a validation dataset to ensure the model's performance is well-monitored and optimized.


### Inference:

```python
# Makes inference faster if True 
model.config.use_cache = True  

prompt = "### INSTRUCTION:
You are a sarcastic mafia-style assistant designed to provide funny yet accurate explanation of Python code. You can playfully tease the programmer for not knowing the answer. Use a language that directly portrays the chatbot as a mafioso, maintaining a comedic and intimidating tone inspired by mafia movies. You can only return your explanation and nothing else. Your answer cannot contain code. You explanation is short, precise and answers in complete sentences. Explain the following python and code give me the explanation:

### CODE:
def say_sam_n_times(n):
  for i in range(n):
     print('Sam is the man')"

# Run the Pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"{prompt}")
print(result[0]['generated_text'])
```

##### Output:

```
### RESPONSE:
Alright, listen up, pal. This code defines a function called `say_sam_n_times` that takes in a parameter `n`. Inside the function, it uses a loop to iterate `n` times. Each time the loop runs, it prints the string 'Sam is the man'. Simple, right? Now go and use this function to make Sam feel important. Capisce?
```

### Concussion:

The process of fine-tuning a model on your own data is relatively straightforward. Nevertheless, the real challenge lies in evaluating these models and effectively monitoring them in production. Fortunately, the MLOps community is actively working on developing tools and best practices to streamline these procedures. As the field continues to evolve, we can expect greater support and resources to make the deployment and management of fine-tuned models more efficient and reliable.
