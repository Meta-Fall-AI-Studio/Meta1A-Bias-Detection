from huggingface_hub import HfApi
from transformers import pipeline, AutoTokenizer

ORG_NAME = "Meta1A"  

api = HfApi()

def list_org_models():
    try:
        models = api.list_models(author=ORG_NAME, use_auth_token=False)
        return models
    except Exception as e:
        raise Exception(f"Failed to list models from Hugging Face: {e}")

def load_model(model_id: str):
    print("model id", model_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = pipeline("text-classification", model=model_id, tokenizer=tokenizer)
        return model, tokenizer    
    except Exception as e:
        raise Exception(f"Either the model repo does not exist or is empty...: {e}")

def chunk_text(text: str, tokenizer, max_tokens: int, overlap_tokens: int = 50):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=False
    )
    input_ids = encoding["input_ids"][0].tolist()
    if len(input_ids) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    end = max_tokens
    while start < len(input_ids):
        chunk_ids = input_ids[start:end]
        # decode chunk back to text
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(input_ids):
            break
        start = end - overlap_tokens
        end = start + max_tokens
    return chunks

def detect_bias(text: str, model, tokenizer):
    # Determine maximum
    try:
        max_len = model.model.config.max_position_embeddings
    except:
        max_len = tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 512

    # Split if needed
    chunks = chunk_text(text, tokenizer, max_tokens=max_len, overlap_tokens=50)

    labels = []
    scores = []
    key = {"LABEL_1": 1, "LABEL_0": 0}
    for chunk in chunks:
        result = model(chunk, truncation=True, max_length=max_len)[0]
        labels.append(key.get(result["label"], 0))
        scores.append(result["score"])

    import statistics
    avg_score = statistics.mean(scores)
    label_int = 1 if labels.count(1) > labels.count(0) else 0

    return label_int, avg_score

