from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Train a PEFT adapter
eli5 = load_dataset("eli5_category", split="train[:100]")

eli5 = eli5.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

eli5 = eli5.flatten()

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

training_args = TrainingArguments(
    output_dir="peft",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model.add_adapter(peft_config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_eli5["train"],
    eval_dataset=tokenized_eli5["test"],
    data_collator=data_collator,
)
trainer.train()