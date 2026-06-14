# Fine-Tuning the ChatAgent for Astronomy

This guide explains how to improve the ChatAgent's performance through system prompt refinement and Qwen Cloud LoRA fine-tuning.

## Overview

The ChatAgent uses Qwen 3.7-plus with a strong astronomy education system prompt. There are two paths to improve accuracy:

1. **System Prompt Tuning** (immediate, free) — Add examples and constraints
2. **LoRA Fine-Tuning** (production, $5–20/100 examples) — Train on real conversation data

## 1. System Prompt Refinement

The current system prompt in `agent/chat_agent.py` can be enhanced with few-shot examples.

### Current Approach

```python
SYSTEM_PROMPT = """You are an expert astronomy guide...
"""
```

### Enhanced Approach

Add domain-specific examples to the system prompt:

```python
SYSTEM_PROMPT = """You are an expert astronomy guide and universe educator.

### Examples

**Q: What is a black hole?**
A: A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape once it crosses the event horizon. Black holes form when massive stars collapse at the end of their life. There are different types: stellar black holes (10–20 solar masses) and supermassive black holes at galaxy centers (millions to billions of solar masses).

**Q: Show me a supernova.**
A: I can generate a video showing a supernova explosion! That would be perfect for visual learning. A supernova occurs when a star explodes at the end of its life or in a binary system where material accretes onto a white dwarf.

**Q: How do exoplanets transit?**
A: When an exoplanet passes in front of its host star from our perspective, we observe a small dip in the star's brightness. This is called a transit. NASA's transit photometry tools detect these dips to discover and characterize exoplanets. Would you like me to generate a video showing a transit in action?

### Core Rules

- Always reference NASA data when available (APOD, DONKI, EPIC, NEO, exoplanets, image library)
- If the user asks for a visual explanation, suggest: "I can generate a video showing this!"
- Keep responses clear and accurate — prioritize facts over speculation
- When uncertain, say "We don't have specific data on that, but here's what we know…"
"""
```

### When to Enhance the System Prompt

- After noticing repeated user questions with wrong answers
- When a domain area (e.g., solar weather, exoplanets) needs better coverage
- Before collecting data for LoRA fine-tuning (establish baseline)

## 2. Data Collection for LoRA Fine-Tuning

### Export Conversation Data

The `RunDB` stores all conversations in SQLite. To prepare data for fine-tuning:

```python
import json
import sqlite3
from pathlib import Path

def export_training_data(db_path: str = "output/runs.db", output_path: str = "training_data.jsonl"):
    """Export runs to JSONL format for Qwen fine-tuning."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT user_message, assistant_response FROM runs ORDER BY created_at DESC"
        ).fetchall()

    with open(output_path, "w") as f:
        for user_msg, assistant_msg in rows:
            example = {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            }
            f.write(json.dumps(example) + "\n")

    print(f"Exported {len(rows)} examples to {output_path}")
```

**Run this script after 50+ conversations have been logged:**

```bash
python -c "from scripts.export_training_data import export_training_data; export_training_data()"
```

### Data Quality Checklist

Before fine-tuning, review the exported JSONL:
- ✅ User queries are diverse (different question types)
- ✅ Assistant responses are accurate and reference NASA data
- ✅ Responses match the astronomy domain
- ❌ Remove conversations with errors or off-topic responses
- ❌ Remove conversations where the ChatAgent hallucinated

**Manual cleanup:**
```bash
# Edit training_data.jsonl to remove bad examples
# Then validate:
python -c "
import json
with open('training_data.jsonl') as f:
    count = sum(1 for _ in f)
    print(f'Valid examples: {count}')
"
```

## 3. Qwen Cloud LoRA Fine-Tuning

### Prerequisite

- [DashScope account](https://dashscope.console.aliyun.com/) with API key
- Training data in JSONL format (see Section 2)
- At least 50 high-quality examples (100+ recommended)

### Option A: DashScope Web Dashboard

1. Go to [DashScope Console](https://dashscope.console.aliyun.com/)
2. Navigate to **Fine-tuning** → **Create Job**
3. Select base model: `qwen3.7-plus`
4. Select fine-tuning method: `LoRA`
5. Upload `training_data.jsonl`
6. Set hyperparameters (defaults are good):
   - Learning rate: `1e-4`
   - Batch size: `8`
   - Epochs: `3`
7. Click **Submit** (takes ~30 minutes)
8. Wait for job to complete, then note the fine-tuned model ID

### Option B: DashScope API

```python
import requests
import json

DASHSCOPE_API_KEY = "your-api-key"
MODEL_ID = "qwen3.7-plus"

def submit_fine_tuning_job(training_file_path: str, model_id: str = MODEL_ID):
    """Submit a fine-tuning job to Qwen Cloud."""
    
    with open(training_file_path, "rb") as f:
        files = {"file": f}
        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        
        payload = {
            "base_model": model_id,
            "fine_tuning_method": "lora",
            "hyperparameters": {
                "learning_rate": "1e-4",
                "batch_size": 8,
                "epochs": 3,
            },
        }
        
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/fine-tuning/jobs",
            headers=headers,
            data={"parameters": json.dumps(payload)},
            files=files,
        )
    
    result = response.json()
    if response.status_code == 200:
        job_id = result["data"]["job_id"]
        print(f"Fine-tuning job submitted: {job_id}")
        return job_id
    else:
        print(f"Error: {result}")
        return None

# Run
job_id = submit_fine_tuning_job("training_data.jsonl")
```

### Monitor Job Status

```bash
# Check status via API
curl -X GET "https://dashscope.aliyuncs.com/api/v1/fine-tuning/jobs/{job_id}" \
  -H "Authorization: Bearer {DASHSCOPE_API_KEY}"

# Or check in DashScope Console → Fine-tuning → Jobs
```

## 4. Deployment

### Update ChatAgent to Use Fine-Tuned Model

Once fine-tuning completes, you'll get a model ID like `qwen3.7-plus-LoRA-xxxxx`.

**Option 1: Environment Variable**
```bash
# In .env
QWEN_CHAT_MODEL="qwen3.7-plus-LoRA-xxxxx"
```

**Option 2: Code Update**
```python
# agent/chat_agent.py
class ChatAgent:
    def __init__(self, qwen_client: QwenClient):
        self.client = qwen_client
        self.model = os.environ.get("QWEN_CHAT_MODEL", "qwen3.7-plus-LoRA-xxxxx")
```

**Option 3: A/B Test**
```python
import random

class ChatAgent:
    def __init__(self, qwen_client: QwenClient, use_fine_tuned: bool = False):
        self.client = qwen_client
        # 50% traffic to fine-tuned, 50% to baseline
        self.model = "qwen3.7-plus-LoRA-xxxxx" if use_fine_tuned else "qwen3.7-plus"
```

## 5. Evaluation

### Baseline vs Fine-Tuned Comparison

After deployment, compare:

| Metric | Baseline | Fine-Tuned |
|--------|----------|-----------|
| NASA data reference accuracy | % | % |
| Astronomy fact accuracy | % | % |
| Video suggestion appropriateness | % | % |
| User satisfaction (if available) | / 5 | / 5 |

### Test Cases

Create a test suite to verify improvements:

```python
test_cases = [
    ("What is a pulsar?", "should mention: neutron star, rotation, radiation"),
    ("Show me a solar flare.", "should suggest video, mention DONKI"),
    ("How far is Proxima Centauri?", "should mention: 4.24 light-years"),
]

# Manual or automated evaluation
```

## 6. Iteration

**After First Fine-Tune:**
1. Monitor user feedback (error rates, thumbs down)
2. Collect 50 more conversations
3. Review for quality and accuracy
4. Remove bad examples
5. Fine-tune again with updated dataset

**When to Retune:**
- When a new NASA dataset becomes available (new exoplanet discoveries, etc.)
- Every 3–6 months as user expectations evolve
- When baseline model version updates (Qwen 3.8, etc.)

## Costs

| Step | Cost |
|------|------|
| System prompt tuning | Free |
| Data collection (50 conversations) | ~$0.10–1 (API calls) |
| LoRA fine-tuning (100 examples) | $5–20 (DashScope pricing) |
| Inference with fine-tuned model | Same as baseline |

## Troubleshooting

**Q: Fine-tuning job failed**
- Check training data JSONL is valid (no malformed lines)
- Ensure file size < 500 MB
- Try with 10 examples first to test

**Q: Fine-tuned model has low quality**
- Review training data for hallucinations
- Increase system prompt context (add more examples)
- Retrain with higher learning rate adjustment

**Q: How do I compare fine-tuned vs baseline?**
- Use A/B testing: route 50% of users to each
- Collect user feedback: "Did this answer help?"
- Track metrics: NASA data reference rate, fact accuracy

## Further Reading

- [DashScope Fine-tuning Guide](https://help.aliyun.com/zh/dashscope/developer-reference/fine-tuning)
- [Qwen Model Cards](https://huggingface.co/collections/Qwen/qwen-official-65374ce313f3a5532fc79eda)
- [LoRA: Low-Rank Adaptation (Paper)](https://arxiv.org/abs/2106.09685)
