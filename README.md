# Banking77 Ticket Classifier - MLOps/LLMOps Project

A production-ready machine learning pipeline for classifying customer banking tickets using DistilBERT, with integrated guardrails, model quantization, and MLflow experiment tracking.

##  Overview

This project implements an end-to-end MLOps pipeline for intent classification on the Banking77 dataset. It includes:

- **DistilBERT-based classifier** fine-tuned on 77 banking intents
- **Model quantization** for efficient deployment (int8 dynamic quantization)
- **MLflow integration** for experiment tracking and model management
- **Guardrails system** with rule-based and optional LLM-based checks for:
  - Urgency detection
  - Toxicity detection
  - Label validation
- **CLI tool** for easy inference
- **Hyperparameter tuning** with automated best model selection

##  Features

- **Efficient Model**: DistilBERT provides fast inference while maintaining high accuracy
- **Quantized Deployment**: Models are quantized to int8 for reduced size and faster inference
- **Experiment Tracking**: Full MLflow integration for tracking hyperparameters, metrics, and artifacts
- **Safety Guardrails**: Multi-layer validation system to catch urgent, toxic, or low-confidence predictions
- **LLM Integration**: Optional GPT-3.5-turbo integration for enhanced guardrail checks
- **Production Ready**: Clean CLI interface and modular codebase for easy deployment

##  Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- OpenAI API key (optional, for LLM-based guardrails)

##  Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **Linux/Mac**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up API key** (optional, for LLM guardrails):
   - Copy `secrets.json.example` to `secrets.json`
   - Add your OpenAI API key:
     ```json
     {
       "openai_api_key": "sk-your-actual-key-here"
     }
     ```
   - Alternatively, set the `OPENAI_API_KEY` environment variable

##  Project Structure

```
banking77-ticket-classifier-mlops-llmops/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── load_data.py   # Load Banking77 dataset
│   │   ├── label_mapping.py  # Label ID ↔ Intent name mappings
│   │   └── preprocess.py  # Tokenization and dataset preparation
│   ├── ml/                # Model training and inference
│   │   ├── model_def.py   # Model architecture definition
│   │   ├── train_basic.py # Basic training script
│   │   ├── train_mlflow.py # MLflow-integrated training
│   │   ├── hyperparameter_sweep.py  # Hyperparameter tuning
│   │   ├── find_and_quantize_best.py  # Find best model and quantize
│   │   ├── run_full_pipeline.py  # Complete pipeline orchestrator
│   │   ├── quantize_model.py  # Model quantization
│   │   ├── inference.py   # Model comparison script
│   │   └── infer.py       # Clean inference interface
│   ├── guardrails/        # Safety and validation checks
│   │   ├── config.py      # API key configuration helper
│   │   ├── urgency.py     # Urgency detection
│   │   ├── toxicity.py    # Toxicity detection
│   │   ├── validation.py  # Label validation
│   │   └── pipeline.py    # Guardrails orchestrator
│   └── cli/               # Command-line tools
│       └── classify.py    # CLI for ticket classification
├── notebooks/
│   └── exploration_banking77.ipynb  # EDA notebook
├── data/
│   └── processed/         # Processed datasets (gitignored)
├── models/                # Trained models (gitignored)
│   └── quantized/         # Quantized models
├── mlruns/                # MLflow tracking data (gitignored)
├── requirements.txt       # Python dependencies
├── secrets.json.example   # API key template
└── README.md             # This file
```

##  Usage Guide

### 1. Data Preparation

**Load and explore the dataset**:
```bash
python -m src.data.load_data
```

**Preprocess and tokenize**:
```bash
python -m src.data.preprocess
```

This creates tokenized datasets in `data/processed/banking77/`.

### 2. Training

#### Basic Training (without MLflow)
```bash
python -m src.ml.train_basic
```

#### Training with MLflow Tracking
```bash
python -m src.ml.train_mlflow
```

You can customize training parameters:
```python
from src.ml.train_mlflow import train_distilbert_mlflow

train_distilbert_mlflow(
    num_epochs=5,
    batch_size=32,
    learning_rate=3e-5,
    run_name="my-custom-run"
)
```

#### Hyperparameter Sweep
Run a grid search over multiple hyperparameter combinations:
```bash
python -m src.ml.hyperparameter_sweep
```

#### Full Pipeline (Sweep → Best Model → Quantize)
Automate the complete workflow:
```bash
python -m src.ml.run_full_pipeline
```

### 3. Model Quantization

Quantize a trained model for efficient deployment:
```bash
python -m src.ml.quantize_model
```

Or quantize a specific model:
```python
from src.ml.quantize_model import quantize_model

quantize_model(
    model_path="models/distilbert-banking77",
    output_path="models/quantized/my-model"
)
```

### 4. Inference

#### Using the CLI Tool
```bash
# Basic classification
python -m src.cli.classify "How do I activate my card?"

# With LLM guardrails
python -m src.cli.classify "My account was hacked!" --llm

# JSON output
python -m src.cli.classify "What is my balance?" --json
```

#### Using Python API
```python
from src.ml.infer import predict
from src.guardrails.pipeline import process_ticket

# Simple prediction
result = predict("How do I activate my card?")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.4f}")

# Full pipeline with guardrails
result = process_ticket(
    "My account was hacked and money is missing!",
    use_llm=True  # Enable LLM-based guardrails
)
print(f"Predicted: {result['predicted_label']}")
print(f"Priority: {result['priority']}")
print(f"Needs Review: {result['needs_manual_review']}")
```

### 5. MLflow UI

View experiment results:
```bash
# Start MLflow UI
python -m mlflow ui

# Or with custom port
python -m mlflow ui --port 5001

# Or with database backend
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` (or your custom port) in your browser.

## 🔧 Configuration

### API Key Setup

The project supports loading the OpenAI API key from:
1. `secrets.json` file in the project root (recommended)
2. `OPENAI_API_KEY` environment variable (fallback)

**Create `secrets.json`**:
```json
{
  "openai_api_key": "sk-your-key-here"
}
```

**Or set environment variable**:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"
```

### Model Paths

Default paths:
- **Trained models**: `models/`
- **Quantized models**: `models/quantized/`
- **Processed data**: `data/processed/banking77/`
- **MLflow runs**: `mlruns/`

You can override these in function calls or scripts.

##  Guardrails System

The guardrails system provides three layers of validation:

### 1. Urgency Detection
- **Rule-based**: Keyword matching (hacked, stolen, fraud, etc.)
- **LLM-based** (optional): GPT-3.5-turbo analysis
- **Output**: Priority level (high/normal) and urgency flag

### 2. Toxicity Detection
- **Rule-based**: Profanity keywords, excessive capitalization
- **LLM-based** (optional): Content moderation analysis
- **Output**: Toxicity flag and reason

### 3. Label Validation
- **Confidence-based**: Flags low-confidence predictions (< 0.5)
- **LLM-based** (optional): Validates label appropriateness
- **Output**: Manual review flag and reason

### Example Guardrail Output

```python
{
    "predicted_label": "compromised_card",
    "model_confidence": 0.92,
    "priority": "high",
    "is_toxic": False,
    "needs_manual_review": True,
    "urgency": {
        "urgent": True,
        "priority": "high",
        "reason": "Contains urgent keyword: 'hacked'"
    },
    "toxicity": {
        "toxic": False,
        "reason": "No toxicity detected"
    },
    "validation": {
        "needs_manual_review": True,
        "reason": "Urgent issue detected"
    }
}
```

##  MLflow Integration

### Experiment Tracking

All training runs are automatically logged to MLflow with:
- **Parameters**: epochs, batch size, learning rate, etc.
- **Metrics**: Training/validation loss, accuracy per epoch
- **Artifacts**: Model checkpoints, tokenizer, config
- **Final Metrics**: Validation and test accuracy

### Viewing Results

1. **Start MLflow UI**: `python -m mlflow ui`
2. **Browse experiments**: Select experiment from sidebar
3. **Compare runs**: Use the "Compare" feature
4. **View artifacts**: Click on a run to see logged files

### Finding Best Model

```python
from src.ml.find_and_quantize_best import find_and_quantize_best

result = find_and_quantize_best(
    experiment_name="banking77-distilbert-sweep",
    metric="final_val_accuracy"
)
```

This will:
1. Query MLflow for all runs in the experiment
2. Find the run with highest validation accuracy
3. Load the model
4. Quantize it
5. Save to `models/quantized/best-model`

##  Banking77 Dataset

The Banking77 dataset contains:
- **77 banking intents** (e.g., `activate_my_card`, `balance_not_updated_after_bank_transfer`)
- **~13,000 training samples**
- **~1,000 test samples**
- **Text**: Customer banking queries

### Example Intents

- `activate_my_card`
- `card_not_working`
- `compromised_card`
- `balance_not_updated_after_bank_transfer`
- `transfer_not_received_by_recipient`
- ... and 72 more

##  Model Architecture

- **Base Model**: `distilbert-base-uncased`
- **Task**: Sequence Classification
- **Output**: 77 classes (one per banking intent)
- **Input**: Tokenized text (max length: 64 tokens)
- **Quantization**: Dynamic int8 quantization for Linear layers

##  Performance

Typical performance metrics:
- **Validation Accuracy**: ~0.92-0.94 (depending on hyperparameters)
- **Test Accuracy**: ~0.91-0.93
- **Quantized Model Size**: ~130 MB (vs ~260 MB original)
- **Inference Speed**: ~2-3x faster on CPU (quantized vs original)

##  Development

### Running Tests

Test individual components:
```bash
# Test data loading
python -m src.data.load_data

# Test preprocessing
python -m src.data.preprocess

# Test inference
python -m src.ml.infer

# Test guardrails
python -m src.guardrails.pipeline

# Test CLI
python -m src.cli.classify "test query"
```

### Adding New Guardrails

1. Create a new file in `src/guardrails/` (e.g., `sentiment.py`)
2. Implement a `check_*()` function returning a dict
3. Add it to `src/guardrails/pipeline.py`
4. Update the CLI if needed

### Customizing Training

Modify `src/ml/train_mlflow.py` or create a new training script based on it.

##  Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**2. CUDA Out of Memory**
- Reduce `batch_size` in training arguments
- Use gradient accumulation

**3. MLflow UI Port Error**
- Try a different port: `python -m mlflow ui --port 5001`
- Check if port is in use: `netstat -ano | findstr :5000` (Windows)

**4. API Key Not Found**
- Ensure `secrets.json` exists in project root
- Or set `OPENAI_API_KEY` environment variable
- Check file permissions

**5. Quantized Model Loading Error**
- Ensure quantization was completed successfully
- Check that `quantized_model.pt` exists in model directory

##  License

This project is for educational and research purposes.

##  Acknowledgments

- **Banking77 Dataset**: [Hugging Face Datasets](https://huggingface.co/datasets/banking77)
- **DistilBERT**: [Hugging Face Transformers](https://huggingface.co/distilbert-base-uncased)
- **MLflow**: [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)

##  Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Banking77 Paper](https://arxiv.org/abs/2003.04807)


