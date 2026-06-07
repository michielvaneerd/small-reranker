import os
# Restrict ONNX to 2 threads, leaving 1 vCPU entirely free for your web server
os.environ["OMP_NUM_THREADS"] = "2"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

app = FastAPI()

MODEL_PATH = "/app/model"
ONNX_FILE_PATH = os.path.join(MODEL_PATH, "model_int8.onnx")

# 1. Download tokenizer & ONNX weights from an INT8 repository
MODEL_ID = "tss-deposium/bge-reranker-v2-m3-onnx-int8"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. Configure the ONNX Session with strict CPU limits
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Load the .onnx file (Hugging Face downloads this to your local cache automatically)
# For alternative repositories, ensure the file name matches (e.g., 'model_quantized.onnx')
# import huggingface_hub
# onnx_path = huggingface_hub.hf_hub_download(repo_id=MODEL_ID, filename="onnx/model_int8.onnx")
# session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
session = ort.InferenceSession(ONNX_FILE_PATH, sess_options, providers=["CPUExecutionProvider"])

class DocumentItem(BaseModel):
    id: int
    content: str

class RerankRequest(BaseModel):
    query: str
    documents: list[str | DocumentItem]

@app.get("/health")
async def health_check():
    # Controleer of het model correct in het geheugen zit
    if session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model of tokenizer niet geladen")
    
    # Optioneel: Voer een supersnelle dummy-berekening uit om te zien of de engine nog werkt
    try:
        dummy_pair = [["test", "test"]]
        encoded = tokenizer(dummy_pair, return_tensors="np")
        # Simpele status check zonder zware belasting
        if "input_ids" not in encoded:
            raise Exception("Tokenizer faalt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model is onbereikbaar: {str(e)}")

    return {"status": "healthy", "model": MODEL_ID}

@app.post("/rerank")
async def rerank(data: RerankRequest):
    # Normalize documents: support both plain strings and {"id": ..., "content": ...} objects
    doc_ids = []
    doc_texts = []
    for doc in data.documents:
        if isinstance(doc, DocumentItem):
            doc_ids.append(doc.id)
            doc_texts.append(doc.content)
        else:
            doc_ids.append(None)
            doc_texts.append(doc)

    # Construct sentence pairs: [[query, doc1], [query, doc2]...]
    pairs = [[data.query, text] for text in doc_texts]
    
    # Tokenize the pairs with padding and truncation
    # BGE-v2-M3 supports up to 8192 tokens, but limit to 512/1024 for speed on small CPUs
    encoded_inputs = tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="np"
    )
    
    # Format inputs specifically for the ONNX Runtime engine
    onnx_inputs = {
        "input_ids": encoded_inputs["input_ids"].astype(np.int64),
        "attention_mask": encoded_inputs["attention_mask"].astype(np.int64)
    }
    
    # If the specific ONNX export uses token_type_ids, include them
    if "token_type_ids" in encoded_inputs:
        onnx_inputs["token_type_ids"] = encoded_inputs["token_type_ids"].astype(np.int64)

    # Run execution directly through ONNX
    onnx_outputs = session.run(None, onnx_inputs)
    
    # Extract raw float values from the logits output layer
    scores = onnx_outputs[0].flatten().tolist()
    
    # Pair documents with scores and sort in descending order
    results = sorted(zip(doc_ids, doc_texts, scores), key=lambda x: x[2], reverse=True)

    def build_result(doc_id, text, score):
        if doc_id is not None:
            return {"id": doc_id, "content": text, "score": score}
        return {"document": text, "score": score}

    return {"results": [build_result(doc_id, text, score) for doc_id, text, score in results]}
