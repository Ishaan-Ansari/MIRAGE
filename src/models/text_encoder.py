"""
Define the TextEncoder class.

This module is responsible for processing medical text queries.
"""

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Simonlee711/Clinical_ModernBERT")
tokenizer = AutoTokenizer.from_pretrained("Simonlee711/Clinical_ModernBERT")


class TextEncoder:
    """Encode medical text using a pre-trained language model."""

    def __init__(self):
        """
        Initialize the TextEncoder.

        Load the pre-trained model and tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, text):
        """Tokenize the input text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def encode(self, text):
        """Encode the input text into embeddings."""
        tokenized_output = self.tokenize(text)
        return tokenized_output
