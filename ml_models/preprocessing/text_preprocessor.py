"""
==============================================================================
TEXT PREPROCESSING MODULE
==============================================================================

WHAT THIS FILE DOES:
Prepares text BEFORE feeding it to the text encoder (DeBERTa-v3-base by default).

WHY TEXT PREPROCESSING IS DIFFERENT:
- Text isn't pixels - it's words and sentences
- Need to convert words to numbers (tokenization)
- Each word becomes a unique ID (token)

WHAT PREPROCESSING INCLUDES:
- Text cleaning (remove extra spaces, special characters)
- Tokenization (words → token IDs)
- Padding (make all sequences same length)
- Attention masks (tell model which tokens are real vs padding)

FILE STRUCTURE:
1. TextPreprocessor class - Main preprocessing pipeline
2. Helper methods for cleaning, tokenization, and feature extraction
==============================================================================
"""

# ============== IMPORTS ==============
import re                              # Regular expressions (for text cleaning)
from transformers import AutoTokenizer


class TextPreprocessor:
    """
    ==============================================================================
    TEXT PREPROCESSOR
    ==============================================================================
    
    Handles all text preprocessing tasks for AI-generated text detection.
    
    TYPICAL WORKFLOW:
    1. Clean text (remove noise, normalize whitespace)
    2. Tokenize (words → token IDs)
    3. Pad/truncate to max_length (512 tokens)
    4. Create attention mask (1 = real token, 0 = padding)
    5. Ready for the encoder model!
    
    USAGE EXAMPLE:
        preprocessor = TextPreprocessor()
        encoded = preprocessor.tokenize("This is AI-generated text.")
        output = model(**encoded)  # Now ready for model!
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-base', max_length=512):
        """
        INITIALIZE TEXT PREPROCESSOR
        
        Args:
            model_name (str): Hugging Face encoder id (must match training checkpoint).
                Default: microsoft/deberta-v3-base
            
            max_length (int): Maximum sequence length (truncate/pad target).
        """
        self.max_length = max_length
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✓ Loaded tokenizer: {model_name}")
            
        except Exception as e:
            # If download fails (no internet), set to None
            self.tokenizer = None
            print(f"✗ Warning: Could not load tokenizer: {str(e)}")
            print("  You'll need internet connection for first-time download.")
    
    def clean_text(self, text):
        """
        CLEAN AND NORMALIZE TEXT
        
        Removes noise and standardizes formatting before tokenization.
        
        Args:
            text (str): Raw input text
        
        Returns:
            str: Cleaned text
        
        WHAT GETS CLEANED:
        1. Multiple spaces → Single space
        2. Unusual characters → Removed
        3. Leading/trailing whitespace → Removed
        4. Punctuation → KEPT (important for meaning!)
        
        WHY CLEAN?
        - Reduces noise
        - Makes tokenization more consistent
        - Improves model performance
        """
        # STEP 1: Normalize whitespace
        # Replace multiple spaces, tabs, newlines with single space
        # Example: "Hello    world\n" → "Hello world"
        text = re.sub(r'\s+', ' ', text)
        
        # STEP 2: Remove special characters (but keep punctuation)
        # \w = word characters (a-z, A-Z, 0-9, _)
        # \s = whitespace
        # .,!?;:\-'"() = punctuation we want to keep
        # Everything else gets removed
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        
        # STEP 3: Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text):
        """
        TOKENIZE TEXT FOR MODEL INPUT
        
        Converts text to token IDs that the model can understand.
        
        Args:
            text (str): Text string to tokenize
        
        Returns:
            dict: Dictionary containing:
                - input_ids: Token IDs (shape: 1 x max_length)
                - attention_mask: Mask showing real vs padded tokens (shape: 1 x max_length)
        
        TOKENIZATION EXAMPLE:
        Input: "This is AI-generated text."
        
        Step 1 - Clean:
        "This is AI-generated text."
        
        Step 2 - Tokenize:
        ["This", "is", "AI", "-", "generated", "text", "."]
        
        Step 3 - Convert to IDs (from tokenizer vocabulary):
        [0, 713, 16, 4687, 12, 14748, 2788, 4, 2]
        (0 = <s> start token, 2 = </s> end token)
        
        Step 4 - Pad to max_length (512):
        [0, 713, 16, 4687, 12, 14748, 2788, 4, 2, 1, 1, 1, ...] (512 tokens)
        (1 = <pad> padding token)
        
        Step 5 - Create attention mask:
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...] (512 values)
        (1 = real token, 0 = padding)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Check internet connection.")
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # ============== TOKENIZE ==============
        # Hugging Face tokenizer does everything in one call:
        encoded = self.tokenizer(
            cleaned_text,                    # Input text
            add_special_tokens=True,         # Add <s> and </s> tokens
            max_length=self.max_length,      # Maximum sequence length (512)
            padding='max_length',            # Pad to max_length with <pad> tokens
            truncation=True,                 # Truncate if longer than max_length
            return_tensors='pt'              # Return PyTorch tensors
        )
        
        # RESULT:
        # encoded = {
        #     'input_ids': tensor([[0, 713, 16, ..., 1, 1]]),      # Token IDs
        #     'attention_mask': tensor([[1, 1, 1, ..., 0, 0]])    # Mask
        # }
        
        return encoded
    
    def tokenize_batch(self, texts):
        """
        TOKENIZE MULTIPLE TEXTS
        
        More efficient than calling tokenize() multiple times.
        
        Args:
            texts (list): List of text strings
                Example: ["Text 1", "Text 2", "Text 3"]
        
        Returns:
            dict: Batched input_ids and attention_mask
                Shape: (batch_size, max_length)
        
        USAGE:
            texts = ["This is real text.", "This is AI-generated text."]
            encoded = preprocessor.tokenize_batch(texts)
            outputs = model(**encoded)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Tokenize batch (same parameters as single tokenization)
        encoded = self.tokenizer(
            cleaned_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def extract_features(self, text):
        """
        EXTRACT STATISTICAL FEATURES FROM TEXT
        
        Extracts hand-crafted features that might help detection.
        These can be used alongside model predictions.
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Dictionary of text statistics
        
        FEATURES EXTRACTED:
        - char_count: Total characters
        - word_count: Total words
        - sentence_count: Total sentences
        - avg_word_length: Average characters per word
        - avg_sentence_length: Average words per sentence
        - unique_word_ratio: Vocabulary diversity (unique/total words)
        - punctuation_count: Number of punctuation marks
        
        WHY THESE FEATURES?
        AI-generated text often has:
        - Lower vocabulary diversity (repetitive)
        - More uniform sentence length
        - Different punctuation patterns
        """
        # Split text into words (by whitespace)
        words = text.split()
        
        # Split text into sentences (by sentence-ending punctuation)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]  # Remove empty strings
        
        # ============== CALCULATE FEATURES ==============
        features = {
            # Basic counts
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            
            # Averages (avoid division by zero with max(x, 1))
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            
            # Vocabulary diversity
            # High ratio = diverse vocabulary (likely human)
            # Low ratio = repetitive vocabulary (might be AI)
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            
            # Punctuation count
            'punctuation_count': len(re.findall(r'[.,!?;:]', text)),
        }
        
        return features
    
    def chunk_long_text(self, text, chunk_size=None, overlap=50):
        """
        SPLIT LONG TEXT INTO OVERLAPPING CHUNKS
        
        Typical encoder max length is 512 tokens. For longer texts, we split into chunks.
        
        Args:
            text (str): Long text to split
            chunk_size (int): Words per chunk (default: max_length)
            overlap (int): Overlapping words between chunks
        
        Returns:
            list: List of text chunks
        
        WHY OVERLAP?
        - Prevents losing context at chunk boundaries
        - Ensures sentences aren't split awkwardly
        
        EXAMPLE:
        Input: "This is a very long text with many words..." (1000 words)
        Output: [
            "This is a very long text..." (512 words),
            "...long text with many words..." (512 words, 50 word overlap),
            "...with many words..." (remaining words)
        ]
        """
        if chunk_size is None:
            chunk_size = self.max_length
        
        # Split text into words
        words = text.split()
        chunks = []
        
        # Create chunks with overlap
        start = 0
        while start < len(words):
            # Get chunk from start to start+chunk_size
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start forward (with overlap)
            start = end - overlap
            
            # Break if we've covered all words
            if start >= len(words):
                break
        
        return chunks
    
    def decode(self, token_ids):
        """
        DECODE TOKEN IDs BACK TO TEXT
        
        Converts token IDs back to human-readable text.
        Useful for debugging or seeing what the model sees.
        
        Args:
            token_ids (torch.Tensor): Tensor of token IDs
        
        Returns:
            str: Decoded text string
        
        EXAMPLE:
        Input: tensor([0, 713, 16, 4687, 2])
        Output: "This is AI-generated text."
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        
        # Decode token IDs (skip_special_tokens removes <s>, </s>, <pad>)
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return decoded_text
    
    def preprocess(self, text):
        """
        COMPLETE PREPROCESSING PIPELINE
        
        This is the main function you'll use!
        It handles everything: cleaning, tokenization, truncation.
        
        Args:
            text (str): Raw input text
        
        Returns:
            dict: Tokenized and encoded text ready for model
        
        USAGE EXAMPLE:
            preprocessor = TextPreprocessor()
            encoded = preprocessor.preprocess("This is AI-generated text.")
            output = model(**encoded)
        """
        # Check if text is too long
        if len(text.split()) > self.max_length:
            print(f"⚠ Text is longer than {self.max_length} tokens. Using first chunk only.")
            # Split into chunks and use first one
            chunks = self.chunk_long_text(text)
            text = chunks[0]
        
        # Tokenize (includes cleaning)
        encoded = self.tokenize(text)
        
        return encoded


# ==============================================================================
# END OF FILE
# ==============================================================================
# 
# SUMMARY:
# This file provides the TextPreprocessor class for preparing text
# before feeding it to the AI text detection encoder model.
# 
# MAIN FUNCTIONS:
# 1. preprocess() - Complete preprocessing pipeline (most used!)
# 2. tokenize() - Convert text to token IDs
# 3. clean_text() - Remove noise and normalize
# 4. extract_features() - Get statistical features
# 5. chunk_long_text() - Split long texts into chunks
# 
# TYPICAL USAGE:
#     preprocessor = TextPreprocessor()
#     encoded = preprocessor.preprocess("This is AI-generated text.")
#     output = model(**encoded)
# ==============================================================================
