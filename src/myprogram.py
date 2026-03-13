#!/usr/bin/env python
import os
import string
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import numpy as np
import urllib.request
import json
from collections import defaultdict, Counter


class CharDataset(Dataset):
    """Dataset for character-level language modeling"""
    def __init__(self, text, char_to_idx, seq_length=100):
        self.text = text
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_length]
        target = self.text[idx+self.seq_length]
        seq_tensor = torch.tensor([self.char_to_idx.get(c, 0) for c in seq], dtype=torch.long)
        target_tensor = torch.tensor(self.char_to_idx.get(target, 0), dtype=torch.long)
        return seq_tensor, target_tensor


class LSTMModel(nn.Module):
    """LSTM-based character-level language model"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        # Use the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class MyModel:
    """
    LSTM-based character prediction model trained on CulturaX dataset (end)
    Uses a dictionary-based fallback for fast test-time prediction.
    """

    def __init__(self, vocab_size=None, char_to_idx=None, idx_to_char=None, model=None, device=None, work_dir='work'):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.char_to_idx = char_to_idx if char_to_idx else {}
        self.idx_to_char = idx_to_char if idx_to_char else {}
        self.vocab_size = vocab_size if vocab_size else len(self.char_to_idx)
        self.model = model
        self.work_dir = work_dir
        self.next_char_map = None
        self.next_char_map_casefold = None
        self.global_next_chars = None
        self.trigram_bigram_map = None
        self.trigram_global_next_chars = None

    def _get_mapping_text(self):
        """Load text used to build next-character mapping."""
        cache_file = os.path.join(self.work_dir, 'training_data_cache.txt')
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                text = f.read()
            if text:
                return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        return ''

    def _build_next_char_mapping_from_text(self, text):
        """Build Unicode-safe mapping: previous char -> ranked next chars."""
        text = (text or '').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        next_by_prev = defaultdict(Counter)
        next_by_prev_casefold = defaultdict(Counter)
        global_next = Counter()

        if len(text) >= 2:
            for i in range(len(text) - 1):
                prev_char = text[i]
                next_char = text[i + 1]
                if next_char in ['\n', '\r', '\t']:
                    continue

                next_by_prev[prev_char][next_char] += 1
                next_by_prev_casefold[prev_char.casefold()][next_char] += 1
                global_next[next_char] += 1

        self.next_char_map = {
            prev_char: [ch for ch, _ in counter.most_common()]
            for prev_char, counter in next_by_prev.items()
        }
        self.next_char_map_casefold = {
            prev_char: [ch for ch, _ in counter.most_common()]
            for prev_char, counter in next_by_prev_casefold.items()
        }
        self.global_next_chars = [ch for ch, _ in global_next.most_common()]

    def _load_mapping_from_json(self):
        """Load precomputed mapping from JSON cache if available."""
        mapping_path = os.path.join(self.work_dir, 'char_next_map.json')
        if not os.path.exists(mapping_path):
            return False
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            self.next_char_map = payload.get('next_char_map', {})
            self.next_char_map_casefold = payload.get('next_char_map_casefold', {})
            self.global_next_chars = payload.get('global_next_chars', [])
            return True
        except Exception:
            return False

    def _save_mapping_to_json(self):
        """Save precomputed mapping to JSON cache for fast test-time loading."""
        if self.next_char_map is None:
            return
        mapping_path = os.path.join(self.work_dir, 'char_next_map.json')
        payload = {
            'next_char_map': self.next_char_map,
            'next_char_map_casefold': self.next_char_map_casefold,
            'global_next_chars': self.global_next_chars
        }
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)

    def _ensure_mapping_loaded(self):
        """Ensure mapping exists without expensive work during normal test-time runs."""
        if self.next_char_map is not None:
            return

        # Fast path: load precomputed mapping from JSON file.
        if self._load_mapping_from_json():
            return

        # Compatibility fallback for older checkpoints: build once from cached text.
        text = self._get_mapping_text()
        self._build_next_char_mapping_from_text(text)
        self._save_mapping_to_json()

    def _load_trigram_map_from_json(self):
        """Load precomputed trigram bigram->next map."""
        trigram_path = os.path.join(self.work_dir, 'trigram_bigram_map.json')
        if not os.path.exists(trigram_path):
            return False
        try:
            with open(trigram_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            self.trigram_bigram_map = payload.get('trigram_bigram_map', {})
            self.trigram_global_next_chars = payload.get('trigram_global_next_chars', [])
            return True
        except Exception:
            return False

    def _build_trigram_map_from_top_json(self, top_payload):
        """Build a fast bigram->next-char prior from wooorm/trigrams top.json."""
        # Approximate language-popularity weights; skew toward major languages.
        popular_codes = {
            'eng': 1.00,
            'cmn': 0.95,
            'zho': 0.95,
            'spa': 0.85,
            'hin': 0.82,
            'arb': 0.78,
            'ben': 0.72,
            'por': 0.68,
            'rus': 0.66,
            'urd': 0.64,
            'ind': 0.63,
            'deu_1996': 0.60,
            'deu_1901': 0.60,
            'jpn': 0.58,
            'kor': 0.56,
            'fra': 0.55,
            'ita': 0.50,
            'tur': 0.48,
            'vie': 0.46,
            'tam': 0.45,
            'tel': 0.44,
            'mar': 0.42,
            'swh': 0.40,
            'pol': 0.37,
            'ukr': 0.35,
            'nld': 0.33,
            'ron': 0.31,
            'ces': 0.29,
            'ell': 0.28,
            'tha': 0.27,
            'pes_1': 0.26,
            'pes_2': 0.26,
        }

        next_by_bigram = defaultdict(Counter)
        global_next = Counter()

        for code, weight in popular_codes.items():
            trigram_counts = top_payload.get(code)
            if not trigram_counts:
                continue

            for trigram, count in trigram_counts.items():
                if not trigram or len(trigram) < 3:
                    continue
                tri = trigram.casefold()
                bigram = tri[:2]
                next_char = tri[2]
                if next_char in ['\n', '\r', '\t']:
                    continue

                weighted = weight * float(count)
                next_by_bigram[bigram][next_char] += weighted
                global_next[next_char] += weighted

        self.trigram_bigram_map = {
            bg: [ch for ch, _ in counter.most_common()]
            for bg, counter in next_by_bigram.items()
        }
        self.trigram_global_next_chars = [ch for ch, _ in global_next.most_common()]

    def _ensure_trigram_loaded(self):
        """Ensure trigram priors are available, downloading once if needed."""
        if self.trigram_bigram_map is not None:
            return

        if self._load_trigram_map_from_json():
            return

        trigram_url = 'https://raw.githubusercontent.com/wooorm/trigrams/main/lib/top.json'
        trigram_json_path = os.path.join(self.work_dir, 'trigrams_top.json')
        trigram_map_path = os.path.join(self.work_dir, 'trigram_bigram_map.json')

        try:
            if not os.path.exists(trigram_json_path):
                with urllib.request.urlopen(trigram_url, timeout=20) as response:
                    payload = response.read().decode('utf-8')
                with open(trigram_json_path, 'w', encoding='utf-8') as f:
                    f.write(payload)

            with open(trigram_json_path, 'r', encoding='utf-8') as f:
                top_payload = json.load(f)

            self._build_trigram_map_from_top_json(top_payload)
            with open(trigram_map_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'trigram_bigram_map': self.trigram_bigram_map,
                        'trigram_global_next_chars': self.trigram_global_next_chars,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception:
            # If download or parsing fails, keep empty trigram priors.
            self.trigram_bigram_map = {}
            self.trigram_global_next_chars = []

    def _predict_from_context(self, context):
        """Predict top-3 next characters using trigram + dictionary backoff."""
        self._ensure_mapping_loaded()
        self._ensure_trigram_loaded()

        disallowed = {'\n', '\r', '\t'}
        fallback_chars = ['e', 't', 'a', ' ', 'o', 'i', 'n', 's', 'r', 'h']
        top_chars = []
        seen = set()

        def add_candidates(candidates):
            for char in candidates:
                if char in disallowed:
                    continue
                if char not in seen:
                    top_chars.append(char)
                    seen.add(char)
                if len(top_chars) >= 3:
                    return

        context = context or ''

        if len(context) >= 2:
            bigram = context[-2:].casefold()
            add_candidates(self.trigram_bigram_map.get(bigram, []))

        last_char = context[-1] if context else ''
        if last_char and len(top_chars) < 3:
            add_candidates(self.next_char_map.get(last_char, []))
            if len(top_chars) < 3:
                add_candidates(self.next_char_map_casefold.get(last_char.casefold(), []))

        if len(top_chars) < 3:
            add_candidates(self.trigram_global_next_chars if self.trigram_global_next_chars else [])

        if len(top_chars) < 3:
            add_candidates(self.global_next_chars if self.global_next_chars else [])

        if len(top_chars) < 3:
            add_candidates(fallback_chars)

        pred = ''.join(top_chars[:3])
        pred = pred.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        return pred.ljust(3, ' ')[:3]

    @classmethod
    def load_training_data(cls, work_dir='work', use_cache=True, max_examples=100000, hf_token=None):
        """
        Load training data. Currently uses Shakespeare text dataset.
        Can be easily swapped to CulturaX or other datasets later.
        
        Args:
            work_dir: Directory to save cache
            use_cache: Whether to use cached data if available
            max_examples: Not used for Shakespeare dataset (kept for compatibility)
            hf_token: Not used for Shakespeare dataset (kept for compatibility)
        """
        # Check for cached data first
        cache_file = os.path.join(work_dir, 'training_data_cache.txt')
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached training data from {cache_file}...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_text = f.read()
                if len(cached_text) >= 200_000:  # Use cache if it has reasonable amount of data
                    print(f"Using cached data: {len(cached_text):,} characters")
                    # Truncate to 300k if cache is larger
                    if len(cached_text) > 300_000:
                        print(f"Truncating cached data to 300k characters for fast training...")
                        cached_text = cached_text[:300_000]
                    return cached_text
                else:
                    print("Cached data too small, will re-download...")
            except Exception as e:
                print(f"Error reading cache: {e}, will re-download...")
        
        # Download Shakespeare dataset
        shakespeare_url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
        print(f"Downloading Shakespeare dataset from {shakespeare_url}...")
        
        try:
            # Download the file
            with urllib.request.urlopen(shakespeare_url) as response:
                text = response.read().decode('utf-8')
            
            print(f"Downloaded {len(text):,} characters from Shakespeare dataset")
            
            # Use only 300k characters for fast training
            target_chars = 300_000
            if len(text) > target_chars:
                print(f"Truncating to {target_chars:,} characters for fast training...")
                text = text[:target_chars]
            elif len(text) < target_chars:
                print(f"Repeating text to reach target of {target_chars:,} characters...")
                repeat_factor = (target_chars // len(text)) + 1
                text = text * repeat_factor
                text = text[:target_chars]  # Trim to exactly target_chars
            
            print(f"Total training text length: {len(text):,} characters")
            
            # Save to cache for future use
            os.makedirs(work_dir, exist_ok=True)
            print(f"Saving data to cache: {cache_file}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print("Cache saved. Future training runs will use this cached data (no re-download needed).")
            
            return text
            
        except Exception as e:
            print(f"Error downloading Shakespeare dataset: {e}")
            print("\nTroubleshooting tips:")
            print("1. Ensure you have internet connection")
            print("2. Check if the URL is accessible")
            print("3. Try downloading manually and placing in work/training_data_cache.txt")
            raise e

    @classmethod
    def load_test_data(cls, fname):
        """Load test data from file"""
        data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                inp = line.rstrip('\n\r')  # Remove trailing newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions to file - each line should have exactly 3 characters"""
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                # Ensure prediction is exactly 3 characters, no newlines
                pred_clean = str(p).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')[:3]
                # Pad to 3 if needed
                pred_clean = pred_clean.ljust(3, ' ')
                f.write('{}\n'.format(pred_clean))

    def build_vocab(self, text):
        """Build character vocabulary from text, excluding newlines"""
        # Get all unique characters, but exclude newlines and other control chars
        # We'll replace newlines with spaces during training
        text_no_newlines = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        unique_chars = sorted(set(text_no_newlines))
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        print(f"Vocabulary size: {self.vocab_size} (newlines excluded)")

    def run_train(self, data, work_dir):
        """Train the LSTM model"""
        if isinstance(data, str):
            text = data
        else:
            text = ''.join(data)
        
        # Build vocabulary
        print("Building vocabulary...")
        self.build_vocab(text)

        # Build and cache dictionary mapping during training so test-time is fast.
        self._build_next_char_mapping_from_text(text)
        
        # Convert text to indices (replace newlines with spaces)
        print("Converting text to indices...")
        text_clean = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text_indices = [self.char_to_idx.get(c, 0) for c in text_clean]
        
        # Create dataset
        print("Creating dataset...")
        seq_length = 100  # Use longer sequences for better learning
        dataset = CharDataset(text_indices, self.char_to_idx, seq_length)
        
        # Use reasonable batch size
        batch_size = 64
        # Use more training samples for better learning
        # Use at least 30k samples, or all available if less
        max_samples = min(len(dataset), 30000)  # Increased to 30k for better learning
        if len(dataset) > max_samples:
            print(f"Using subset of {max_samples:,} samples (out of {len(dataset):,}) for training")
            # Use different random samples each epoch by shuffling
            indices = torch.randperm(len(dataset))[:max_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            print(f"Using all {len(dataset):,} available samples for training")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Initialize model with reasonable size
        print("Initializing model...")
        self.model = LSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Training loop with early stopping (but not too aggressive)
        num_epochs = 5
        min_loss = float('inf')
        patience = 3  # More patience
        patience_counter = 0
        
        print(f"Training for up to {num_epochs} epochs on {self.device}...")
        print("(Early stopping enabled - will stop if loss doesn't improve for 3 epochs)")
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (seq, target) in enumerate(pbar):
                seq = seq.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                hidden = self.model.init_hidden(seq.size(0), self.device)
                output, _ = self.model(seq, hidden)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if batch_idx % 50 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
            # Early stopping (but only if loss really plateaus)
            if avg_loss < min_loss:
                min_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: loss hasn't improved for {patience} epochs")
                    break
            
            # Don't stop too early - ensure we train at least 3 epochs
            if epoch < 2 and avg_loss < 0.1:
                print(f"Loss is low but continuing training to ensure model learns properly...")
        
        print("Training completed!")

    def run_pred(self, data):
        """Generate predictions for test data"""
        # Fast path: trigram-informed backoff plus last-character dictionary map.
        preds = []
        for inp in data:
            inp_clean = inp.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            preds.append(self._predict_from_context(inp_clean))
        return preds

        # self.model.eval()
        # preds = []
        #
        # with torch.no_grad():
        #     for inp in tqdm(data, desc="Generating predictions"):
        #         # Clean input - replace newlines with spaces to match training
        #         inp_clean = inp.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        #
        #         # Convert input to indices
        #         seq = [self.char_to_idx.get(c, 0) for c in inp_clean]
        #         if len(seq) == 0:
        #             # Fallback if empty input - use common characters
        #             top_guesses = [' ', 'a', 'e']
        #             preds.append(''.join(top_guesses))
        #             continue
        #
        #         # Use sequence length similar to training (100 chars) for better context
        #         # But use the actual input length if shorter
        #         seq_length = min(len(seq), 100)
        #         if len(seq) > seq_length:
        #             # Use last seq_length characters (most recent context)
        #             seq = seq[-seq_length:]
        #         elif len(seq) < seq_length:
        #             # Pad with spaces (index 0) to match training sequence length
        #             seq = [0] * (seq_length - len(seq)) + seq
        #
        #         seq_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
        #
        #         # Get prediction
        #         hidden = self.model.init_hidden(1, self.device)
        #         output, _ = self.model(seq_tensor, hidden)
        #
        #         # Apply temperature to avoid collapse (make predictions less confident)
        #         temperature = 1.2
        #         logits = output[0] / temperature
        #         probs = torch.softmax(logits, dim=0)
        #
        #         # Get top 3 predictions - ensure they are distinct and exclude newlines
        #         top_chars = []
        #         seen_chars = set()
        #         top_probs, top_indices = torch.topk(probs, min(100, self.vocab_size))  # Get many candidates
        #
        #         # Filter out newline characters and other control characters
        #         for idx in top_indices:
        #             char = self.idx_to_char.get(idx.item(), ' ')
        #             # Skip newlines, carriage returns, and other control characters
        #             if char in ['\n', '\r', '\t']:
        #                 continue
        #             if char not in seen_chars:
        #                 top_chars.append(char)
        #                 seen_chars.add(char)
        #                 if len(top_chars) >= 3:
        #                     break
        #
        #         # Ensure we have exactly 3 characters (pad with space if needed)
        #         while len(top_chars) < 3:
        #             # Try to find a character not already used (excluding newlines)
        #             for char in [' ', 'a', 'e', 'i', 'o', 'u', 't', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']:
        #                 if char not in seen_chars and char not in ['\n', '\r', '\t']:
        #                     top_chars.append(char)
        #                     seen_chars.add(char)
        #                     break
        #             else:
        #                 # If all common chars are used, just use space
        #                 top_chars.append(' ')
        #
        #         # Take exactly 3 and join - ensure no newlines
        #         pred_str = ''.join(top_chars[:3]).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        #         # Ensure we have exactly 3 characters after replacement
        #         if len(pred_str) < 3:
        #             pred_str = pred_str.ljust(3, ' ')
        #         preds.append(pred_str[:3])  # Force exactly 3 chars
        #
        # return preds

    def save(self, work_dir):
        """Save model and vocabulary"""
        self.work_dir = work_dir
        self._save_mapping_to_json()

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'next_char_map': self.next_char_map,
            'next_char_map_casefold': self.next_char_map_casefold,
            'global_next_chars': self.global_next_chars,
            'model_config': {
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'dropout': 0.2
            }
        }
        
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save vocabulary as JSON for easier inspection
        vocab_path = os.path.join(work_dir, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {checkpoint_path}")

    @classmethod
    def load(cls, work_dir):
        """Load model and vocabulary"""
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct model
        config = checkpoint.get('model_config', {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.2
        })
        
        model = LSTMModel(
            vocab_size=checkpoint['vocab_size'],
            **config
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        instance = cls(
            vocab_size=checkpoint['vocab_size'],
            char_to_idx=checkpoint['char_to_idx'],
            idx_to_char=checkpoint['idx_to_char'],
            model=model,
            device=device,
            work_dir=work_dir
        )

        instance.next_char_map = checkpoint.get('next_char_map')
        instance.next_char_map_casefold = checkpoint.get('next_char_map_casefold')
        instance.global_next_chars = checkpoint.get('global_next_chars')

        if instance.next_char_map is None:
            instance._ensure_mapping_loaded()
        
        print(f"Model loaded from {checkpoint_path}")
        return instance

    @classmethod
    def load_fast(cls, work_dir):
        """Load only dictionary resources for fast test-time prediction."""
        instance = cls(work_dir=work_dir)
        instance._ensure_mapping_loaded()
        print(f"Fast dictionary predictor loaded from {work_dir}")
        return instance


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--max_examples', type=int, default=100000, 
                       help='Maximum number of examples (for future use with other datasets)')
    parser.add_argument('--no_cache', action='store_true', 
                       help='Disable using cached training data (force re-download)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face token (for future use with HF datasets)')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        print(f'Max examples to process: {args.max_examples:,}')
        train_data = MyModel.load_training_data(
            work_dir=args.work_dir, 
            use_cache=not args.no_cache,
            max_examples=args.max_examples,
            hf_token=args.hf_token
        )
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading fast dictionary predictor')
        model = MyModel.load_fast(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
