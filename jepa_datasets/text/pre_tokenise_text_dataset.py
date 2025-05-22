from datasets import load_dataset
from transformers import BertTokenizerFast

from configs import get_text_dataset_config, get_text_dataset_tokenization_config

if __name__ == "__main__":
    dataset_config = get_text_dataset_config()
    tokenization_config = get_text_dataset_tokenization_config()

    # 1. Load HuggingFace dataset and tokenizer
    dataset = load_dataset(
        dataset_config["UNTOKENIZED_DATASET_NAME"],
        dataset_config["UNTOKENIZED_DATASET_SPLIT"],
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        dataset_config["TOKENIZER_MODEL_NAME"]
    )
    print("✅ Dataset and tokenizer loaded")

    # 2. Apply pre-tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=tokenization_config["TRUNCATION"],
            padding=tokenization_config["PADDING"],
            max_length=tokenization_config["MAX_LENGTH"],  # match model's max_length
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=tokenization_config["BATCHED"],
        remove_columns=["text"],  # remove raw text
        num_proc=tokenization_config["NUM_PROC"],  # multi-processing
    )
    print("✅ Dataset pre-tokenized")

    # 3. Set format to PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("✅ Dataset set to PyTorch format")

    # 4. Save to disk
    tokenized_dataset.save_to_disk(dataset_config["TOKENIZED_DATASET_NAME"])
    print("✅ Dataset saved to disk")

    print(tokenized_dataset)
