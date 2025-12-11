# Quantization Effects on RAG Performance - Evaluation Framework

Research-grade evaluation framework for measuring quantization effects on small language models in RAG scenarios. Optimized for T4 GPU (16GB VRAM).

## Features

- **Comprehensive Metrics**: Accuracy, faithfulness, efficiency, retrieval quality, perplexity
- **Multiple Evaluators**: Performance, perplexity, context length, RAG, robustness
- **LM-Eval Integration**: Automated lm-evaluation-harness wrapper
- **Quantization Support**: AWQ, GPTQ, BitsAndBytes, ExLlamaV2
- **Memory Efficient**: Designed for 16GB VRAM constraint
- **Research Grade**: Statistical analysis, degradation metrics, reproducible results

## Project Structure

```
.
├── configs/                 # Configuration files
│   └── evaluation.yaml
├── datasets/               # Dataset loaders
│   ├── __init__.py
│   └── loader.py
├── evaluators/             # Experiment orchestration
│   ├── __init__.py
│   ├── base.py
│   ├── context_eval.py
│   ├── performance_eval.py
│   ├── perplexity_eval.py
│   ├── rag_eval.py
│   └── robustness_eval.py
├── metrics/                # Atomic measurement functions
│   ├── __init__.py
│   ├── accuracy.py
│   ├── efficiency.py
│   ├── faithfulness.py
│   ├── perplexity.py
│   └── retrieval.py
├── models/                 # Model interfaces
│   ├── __init__.py
│   ├── model_interface.py
│   ├── awq_model.py
│   ├── bitsandbytes_model.py
│   ├── exllama_model.py
│   └── huggingface_model.py
├── quantization/           # Quantization methods
│   ├── __init__.py
│   ├── base.py
│   └── awq_quantizer.py
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── common.py
│   └── lm_eval_integration.py
├── run_evaluation.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Evaluation

```bash
python run_evaluation.py \
    --model_path "mistralai/Mistral-7B-v0.1" \
    --model_type hf \
    --output_dir results/baseline
```

### AWQ Quantized Model

```bash
python run_evaluation.py \
    --model_path "TheBloke/Mistral-7B-AWQ" \
    --model_type awq \
    --output_dir results/awq
```

### Custom Configuration

```bash
python run_evaluation.py \
    --model_path "your-model" \
    --model_type awq \
    --output_dir results/custom \
    --num_runs 20 \
    --ppl_samples 200 \
    --context_lengths 512 1024 2048 4096 \
    --context_samples 50
```

## Available Evaluators

### Performance Evaluator

Measures time and space metrics:

- Latency (ms/token)
- Throughput (tokens/s)
- Time to First Token (TTFT)
- Memory usage
- Model size

```python
from models import HuggingFaceModel
from evaluators import PerformanceEvaluator

model = HuggingFaceModel("model-path", device="cuda:0")
model.load()

evaluator = PerformanceEvaluator(model, num_runs=10)
results = evaluator.run()
```

### Perplexity Evaluator

Evaluates language modeling quality:

- Perplexity on WikiText/C4
- Cross-entropy loss
- Degradation metrics

```python
from evaluators import PerplexityEvaluator

evaluator = PerplexityEvaluator(
    model,
    dataset_name="wikitext",
    max_samples=100
)
results = evaluator.run()
```

### Context Length Evaluator

Tests performance degradation with context size:

- Multiple context lengths (512, 1024, 2048, 4096)
- Answer position sensitivity
- Degradation rate analysis

```python
from evaluators import ContextLengthEvaluator

evaluator = ContextLengthEvaluator(
    model,
    context_lengths=[512, 1024, 2048],
    samples_per_length=25
)
results = evaluator.run()
```

### RAG Evaluator

Comprehensive RAG evaluation:

- Retrieval quality metrics
- Answer quality (EM, F1, ROUGE)
- Faithfulness scores
- RAG vs no-RAG comparison

```python
from evaluators import RAGEvaluator

evaluator = RAGEvaluator(model, rag_pipeline)
results = evaluator.run(
    questions=questions,
    ground_truth_answers=answers,
    compare_no_rag=True
)
```

### Robustness Evaluator

Test model behavior under adversarial conditions:

- Noisy document injection
- Passage order sensitivity
- Retrieval count variation

```python
from evaluators import RobustnessEvaluator

evaluator = RobustnessEvaluator(
    model,
    rag_pipeline,
    noise_ratios=[0.0, 0.2, 0.5]
)
results = evaluator.run(questions, answers)
```

## Supported Model Types

- **hf/huggingface**: HuggingFace Transformers (FP16/BF16)
- **awq**: AutoAWQ 4-bit quantization
- **gptq/exllama**: GPTQ with ExLlamaV2 backend
- **bnb/bitsandbytes**: BitsAndBytes NF4/INT8

## Configuration

Edit `configs/evaluation.yaml` to customize:

```yaml
model:
  path: "your-model-path"
  type: "awq"
  device: "cuda:0"

performance:
  enabled: true
  num_runs: 10
  max_new_tokens: 128

perplexity:
  enabled: true
  max_samples: 100
  max_length: 512

context_length:
  enabled: true
  context_lengths: [512, 1024, 2048]
  samples_per_length: 25

rag_benchmarks:
  retrieval:
    enabled: false
    pipeline_config:
      chunking:
        chunk_size: 512
        chunk_overlap: 50
      embedding:
        model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

## Memory Optimization for T4 (16GB)

1. Use appropriate context lengths: Start with 512-1024 tokens
2. Reduce batch size: Always use `batch_size=1`
3. Limit samples: Use 25-50 samples per evaluation
4. Clear memory frequently: Call `clear_memory()` between runs
5. Skip attention evaluation: Not supported by quantized models

## Command Line Options

```bash
python run_evaluation.py \
    --model_path <path>              # Model path or HF ID
    --model_type <type>              # awq, hf, gptq, bnb
    --output_dir <dir>               # Output directory
    --device cuda:0                  # Device
    --num_runs 10                    # Performance runs
    --ppl_samples 100                # Perplexity samples
    --context_samples 25             # Samples per context length
    --context_lengths 512 1024 2048  # Context lengths to test
    --skip_performance               # Skip performance eval
    --skip_perplexity                # Skip perplexity eval
    --skip_context                   # Skip context eval
    --log_level INFO                 # Logging level
```

## Results Format

All results are saved as JSON:

```json
{
  "model_path": "model-id",
  "model_type": "awq",
  "model_info": {
    "num_layers": 32,
    "hidden_size": 4096,
    "dtype": "awq_4bit"
  },
  "performance": {
    "timing": {
      "latency_ms_per_token": 5.2,
      "throughput_tokens_per_sec": 192.3,
      "ttft_ms": 45.1
    },
    "memory": {
      "peak_mb": 4250.5,
      "allocated_mb": 4100.2
    },
    "model_size": {
      "size_gb": 3.8,
      "total_params": 7241728000
    }
  },
  "perplexity": {
    "perplexity": 15.2,
    "loss": 2.72,
    "num_samples": 100
  },
  "context": {
    "by_length": {
      "512": { "f1": 0.75, "accuracy": 0.68 },
      "1024": { "f1": 0.72, "accuracy": 0.65 },
      "2048": { "f1": 0.69, "accuracy": 0.62 }
    },
    "degradation": {
      "slope_per_1k_tokens": -0.003,
      "interpretation": "minimal"
    }
  }
}
```

## Using Metrics Directly

```python
from metrics import token_f1, exact_match, token_overlap

f1 = token_f1("predicted text", "reference text")
em = exact_match("predicted", "reference")
faithfulness = token_overlap("prediction", "context")
```

## Extending the Framework

### Add New Metrics

```python
# metrics/custom.py
def my_metric(prediction: str, reference: str) -> float:
    return score
```

### Add New Evaluator

```python
from evaluators.base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def run(self):
        return results
```

### Add New Model Type

```python
from models.model_interface import ModelInterface

class MyModel(ModelInterface):
    def load(self):
        pass

    def generate(self, prompt, config, return_attentions=False):
        pass
```

## LM-Eval Integration

Run standard benchmarks using lm-evaluation-harness:

```python
from utils import LMEvalRunner

runner = LMEvalRunner("model-path", model_type="awq")
results = runner.run_task_group("reasoning")
```

Or use the CLI directly:

```bash
lm_eval --model hf \
    --model_args pretrained=your-model \
    --tasks arc_easy,hellaswag \
    --device cuda:0 \
    --batch_size 1
```

## License

MIT License

## Contributing

Contributions welcome. Please follow existing code style and update documentation.
