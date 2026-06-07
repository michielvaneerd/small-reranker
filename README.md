# Reranker with local model for webserver

Can run on webserver with 6GB RAM, and 3 CPU, doesn't need GPU.

**BAAI/bge-reranker-v2-m3**: The gold standard baseline for open-source RAG pipelines. At 0.6B parameters (~1.2GB), it supports over 100 languages, handles long context, and has fantastic accuracy. It runs comfortably on modest GPU hardware or a mainstream multi-core CPU.

Switching to an INT8 ONNX-quantized version of BGE-Reranker-v2-M3 is the ultimate optimization strategy for a 6GB RAM environment.

Quantization compresses the model's weights from floating-point (FP32) numbers down to 8-bit integers (INT8). Running this via onnxruntime yields significant structural improvements for your web server:

- Drastic RAM Drop: The standard model takes up about 2.2 GB of RAM. An INT8 ONNX version drops this footprint to ~570 MB. This safely preserves over 5 GB of RAM on your server for OS overhead and system requests
- CPU Speed Boost: Standard CPUs are much faster at calculating 8-bit integer math than complex floating-point calculations. You will experience reduced overall latency.
- Negligible Quality Loss: Compressing to INT8 preserves roughly 99% of the model's ranking accuracy while completely changing its operational footprint.

## Clone repo

```
git clone https://huggingface.co/tss-deposium/bge-reranker-v2-m3-onnx-int8
```

## Create local model directory

Create `reranker_model` directory.

## Copy model files into dir

Copy the onnx/model_int8.onnx and the .json files to the reranker_model directory:

```
reranker_model/model_int8.onnx
reranker_model/config.json
reranker_model/special_tokens_map.json
reranker_model/tokenizer_config.json
reranker_model/tokenizer.json
```

## Build and run

`docker compose up --build -d`

# Example `/rerank` request

## Request without document id

```curl
curl -X POST "http://localhost:8001/rerank" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Hoeveel RAM heeft mijn webserver nodig voor een lokaal LLM model?",
       "documents": [
         "Een standaard Nginx webserver draait prima op een machine met minder dan 512MB RAM.",
         "Voor het lokaal draaien van kleinere open-source LLM of reranker modellen is minimaal 2GB tot 8GB RAM vereist, afhankelijk van de modelgrootte en kwantisatie.",
         "De introductie van Docker containers helpt bij het isoleren van applicaties op Linux systemen.",
         "Kwantisatie comprimeert modelgewichten van FP32 naar INT8 om het geheugengebruik drastisch te verlagen."
       ]
     }'
```

### Response

```json
{
    "results": [
        {
            "document": "Voor het lokaal draaien van kleinere open-source LLM of reranker modellen is minimaal 2GB tot 8GB RAM vereist, afhankelijk van de modelgrootte en kwantisatie.",
            "score": 5.066494464874268
        },
        {
            "document": "Een standaard Nginx webserver draait prima op een machine met minder dan 512MB RAM.",
            "score": -1.8729965686798096
        },
        {
            "document": "Kwantisatie comprimeert modelgewichten van FP32 naar INT8 om het geheugengebruik drastisch te verlagen.",
            "score": -9.211983680725098
        },
        {
            "document": "De introductie van Docker containers helpt bij het isoleren van applicaties op Linux systemen.",
            "score": -10.98775863647461
        }
    ]
}
```

## Request with document id

```
curl -X POST "http://localhost:8001/rerank" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Hoeveel RAM heeft mijn webserver nodig voor een lokaal LLM model?",
       "documents": [
         {"id": 1, "content": "Een standaard Nginx webserver draait prima op een machine met minder dan 512MB RAM."},
         {"id": 2, "content": "Voor het lokaal draaien van kleinere open-source LLM of reranker modellen is minimaal 2GB tot 8GB RAM vereist, afhankelijk van de modelgrootte en kwantisatie."},
         {"id": 3, "content": "De introductie van Docker containers helpt bij het isoleren van applicaties op Linux systemen."},
         {"id": 4, "content": "Kwantisatie comprimeert modelgewichten van FP32 naar INT8 om het geheugengebruik drastisch te verlagen."}
       ]
     }'
```

### Response

```json
{
    "results": [
        {
            "id": 2,
            "content": "Voor het lokaal draaien van kleinere open-source LLM of reranker modellen is minimaal 2GB tot 8GB RAM vereist, afhankelijk van de modelgrootte en kwantisatie.",
            "score": 5.066494464874268
        },
        {
            "id": 1,
            "content": "Een standaard Nginx webserver draait prima op een machine met minder dan 512MB RAM.",
            "score": -1.8729965686798096
        },
        {
            "id": 4,
            "content": "Kwantisatie comprimeert modelgewichten van FP32 naar INT8 om het geheugengebruik drastisch te verlagen.",
            "score": -9.211983680725098
        },
        {
            "id": 3,
            "content": "De introductie van Docker containers helpt bij het isoleren van applicaties op Linux systemen.",
            "score": -10.98775863647461
        }
    ]
}
```