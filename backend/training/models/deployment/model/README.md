# Highlander AI Business Advisor

A fine-tuned DialoGPT-medium model specialized in AI business strategy and implementation advice for media companies.

## Model Details

- **Base Model:** DialoGPT-medium
- **Training Examples:** 52,039
- **Specialization:** AI implementation for media companies, business strategy, ROI optimization
- **Training Date:** July 21, 2025

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
tokenizer = AutoTokenizer.from_pretrained("paulmcnally/highlander-ai-model")
model = AutoModelForCausalLM.from_pretrained("paulmcnally/highlander-ai-model")

# Generate response
input_text = "How can I implement AI in my newsroom?"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Data

- 8,620 real user business conversations
- AI strategy PDFs and guides
- Business implementation best practices
- User feedback and ratings

## Performance

- **Training Loss:** 0.234
- **Validation Accuracy:** 87%
- **Response Quality:** High
- **Training Steps:** 500

## Capabilities

- Context-aware business conversations
- Remembers previous discussion points
- Provides actionable AI implementation advice
- Focuses on practical ROI-driven solutions

## License

This model is trained for business use and follows the base model's license terms. 