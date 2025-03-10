import os
import json
import pickle
import struct
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import re
from collections import defaultdict
import math
import heapq

class Posting:
    def __init__(self, doc_id, term_freq=1, importance_score=1.0):
        self.doc_id = doc_id
        self.term_freq = term_freq
        self.importance_score = importance_score

class TermEntry:
    """Helper class for multi-way merge to ensure proper term comparison"""
    def __init__(self, term: str, postings: list, file_idx: int):
        self.term = term
        self.postings = postings
        self.file_idx = file_idx
    
    def __lt__(self, other):
        return self.term < other.term

class InvertedIndex:
    def __init__(self, partial_index_size=50000):
        self.index = {}
        self.stemmer = PorterStemmer()
        self.partial_index_size = partial_index_size
        self.partial_index_count = 0
        self.important_tags = {
            'h1': 3.0, 'h2': 2.5, 'h3': 2.0,
            'title': 3.0, 'b': 1.5, 'strong': 1.5
        }
        
    def clean_and_tokenize(self, text):
        """Clean and tokenize text, returning stemmed tokens with their importance scores."""
        try:
            soup = BeautifulSoup(text, 'html.parser')
            
            # Initialize token importance tracking
            token_importance = defaultdict(float)
            
            # Process each important tag
            for tag_name, importance in self.important_tags.items():
                for tag in soup.find_all(tag_name):
                    words = re.findall(r'[a-z0-9]+', tag.get_text().lower())
                    stemmed_words = [self.stemmer.stem(word) for word in words]
                    for word in stemmed_words:
                        token_importance[word] = max(token_importance[word], importance)
            
            # Process all text
            text = soup.get_text()
        except:
            text = text  # If HTML parsing fails, use raw text
            
        # Find all words
        words = re.findall(r'[a-z0-9]+', text.lower())
        stemmed_words = [self.stemmer.stem(word) for word in words]
        
        # For words not in important tags, ensure they have at least importance 1.0
        result = [(word, token_importance.get(word, 1.0)) for word in stemmed_words]
        return result, len(words)
    
    def add_tokens(self, tokens_with_importance, doc_length, doc_id):
        """Add tokens from a document to the index with logarithmic scaling."""
        # Skip documents that are too short
        if doc_length < 50:
            return
            
        # Count frequency and track importance of each token in document
        term_freq = defaultdict(int)
        term_importance = defaultdict(float)
        
        for token, importance in tokens_with_importance:
            term_freq[token] += 1
            term_importance[token] = max(term_importance[token], importance)
        
        # Add to inverted index with logarithmic scaling
        for token, freq in term_freq.items():
            if token not in self.index:
                self.index[token] = []
            
            # Use logarithmic scaling for term frequency: 1 + log(tf)
            # This dampens the effect of high frequency terms without penalizing long documents too much
            normalized_freq = (1 + math.log10(freq)) / math.log10(doc_length)
            self.index[token].append(Posting(doc_id, normalized_freq, term_importance[token]))
        
        # Check if we need to write a partial index
        if len(self.index) >= self.partial_index_size:
            self._write_partial_index()
    
    def process_document(self, doc_id, content):
        """Process a document and add its tokens to the index."""
        tokens_with_importance, doc_length = self.clean_and_tokenize(content)
        self.add_tokens(tokens_with_importance, doc_length, doc_id)
    
    def _write_partial_index(self):
        """Write current index to disk as a partial index."""
        if not self.index:
            return
            
        # Create partial indexes directory if it doesn't exist
        os.makedirs("partial_indexes", exist_ok=True)
        
        # Write partial index to disk
        partial_path = f"partial_indexes/partial_{self.partial_index_count}.pkl"
        with open(partial_path, "wb") as f:
            pickle.dump(self.index, f)
        
        print(f"Wrote partial index {self.partial_index_count} with {len(self.index)} terms")
        self.partial_index_count += 1
        self.index.clear()
    
    def _merge_partial_indexes(self):
        """Merge all partial indexes using a multi-way merge with buffered I/O."""
        if not os.path.exists("partial_indexes"):
            return
            
        print("Merging partial indexes...")
        
        # Create output directories
        os.makedirs("index_files", exist_ok=True)
        
        # Initialize doc_id map
        doc_id_map = {}
        
        # Open all partial indexes simultaneously
        partial_files = []
        term_queues = []  # List[TermEntry]
        buffer_size = 8192  # 8KB buffer for each file
        
        for i in range(self.partial_index_count):
            f = open(f"partial_indexes/partial_{i}.pkl", "rb")
            partial_files.append(f)
            partial_index = pickle.load(f)
            if partial_index:
                first_term = min(partial_index.keys())
                # Create TermEntry for proper heap ordering
                term_queues.append(TermEntry(first_term, partial_index[first_term], i))
                # Store remaining terms for this file
                partial_index.pop(first_term)
                setattr(partial_files[i], 'remaining_terms', sorted(partial_index.keys()))
                setattr(partial_files[i], 'current_index', partial_index)
            f.seek(0)  # Reset file pointer for future reads
        
        # Create heapq for efficient minimum term selection
        heapq.heapify(term_queues)
        
        # Track term ranges for splitting into 4 files
        all_terms = set()
        for f in partial_files:
            if hasattr(f, 'remaining_terms'):
                all_terms.update(f.remaining_terms)
                if term_queues:  # Check if term_queues is not empty
                    all_terms.add(term_queues[0].term)  # Add the first term too
        
        all_terms = sorted(all_terms)
        num_ranges = 4
        terms_per_range = (len(all_terms) + num_ranges - 1) // num_ranges
        
        # Initialize variables for range processing
        current_range_idx = 0
        current_range_terms = []
        current_vocab_file = None
        current_postings_file = None
        current_range_name = None
        
        def write_range_files():
            nonlocal current_range_terms, current_vocab_file, current_postings_file, current_range_name
            if current_range_terms:
                start_term = current_range_terms[0]
                end_term = current_range_terms[-1]
                current_range_name = f"{start_term[:2]}-{end_term[:2]}"
                
                # Close previous files if open
                if current_vocab_file:
                    current_vocab_file.close()
                if current_postings_file:
                    current_postings_file.close()
                
                # Open new files
                current_vocab_file = open(f"index_files/vocab_{current_range_name}.txt", "w", encoding="utf-8", buffering=buffer_size)
                current_postings_file = open(f"index_files/postings_{current_range_name}.bin", "wb", buffering=buffer_size)
                print(f"Created range files for {current_range_name}")
                current_range_terms = []
        
        # Process terms in sorted order
        while term_queues:
            current_entry = heapq.heappop(term_queues)
            current_term = current_entry.term
            current_postings = current_entry.postings
            file_idx = current_entry.file_idx
            
            # Add term to current range
            current_range_terms.append(current_term)
            
            # Create first range files if none exist
            if current_vocab_file is None:
                write_range_files()
            # If we've filled a range, write it out
            elif len(current_range_terms) >= terms_per_range:
                write_range_files()
            
            # Merge all postings for current_term
            while term_queues and term_queues[0].term == current_term:
                next_entry = heapq.heappop(term_queues)
                current_postings.extend(next_entry.postings)
                
                # Load next term from this file
                if hasattr(partial_files[next_entry.file_idx], 'remaining_terms'):
                    remaining = partial_files[next_entry.file_idx].remaining_terms
                    if remaining:
                        next_term = remaining[0]
                        remaining.pop(0)
                        next_postings = partial_files[next_entry.file_idx].current_index[next_term]
                        del partial_files[next_entry.file_idx].current_index[next_term]
                        heapq.heappush(term_queues, TermEntry(next_term, next_postings, next_entry.file_idx))
            
            # Load next term from current file
            if hasattr(partial_files[file_idx], 'remaining_terms'):
                remaining = partial_files[file_idx].remaining_terms
                if remaining:
                    next_term = remaining[0]
                    remaining.pop(0)
                    next_postings = partial_files[file_idx].current_index[next_term]
                    del partial_files[file_idx].current_index[next_term]
                    heapq.heappush(term_queues, TermEntry(next_term, next_postings, file_idx))
            
            # Write current term's postings
            start_pos = current_postings_file.tell()
            
            # Write postings
            for posting in current_postings:
                doc_id_hash = hash(posting.doc_id) & 0xFFFFFFFF
                doc_id_map[doc_id_hash] = posting.doc_id
                current_postings_file.write(struct.pack("Iff", doc_id_hash, posting.term_freq, posting.importance_score))
            
            # Calculate postings length
            postings_length = current_postings_file.tell() - start_pos
            
            # Write vocabulary entry
            current_vocab_file.write(f"{current_term}\t{len(current_postings)}\t{start_pos}\t{postings_length}\n")
        
        # Write final range if needed
        write_range_files()
        
        # Close all files
        for f in partial_files:
            f.close()
        if current_vocab_file:
            current_vocab_file.close()
        if current_postings_file:
            current_postings_file.close()
        
        # Save document ID mapping
        with open("index_files/docid_map.pkl", "wb") as f:
            pickle.dump(doc_id_map, f)
        
        # Remove partial indexes
        for i in range(self.partial_index_count):
            os.remove(f"partial_indexes/partial_{i}.pkl")
        os.rmdir("partial_indexes")
        
        # Print statistics
        total_vocab_size = sum(os.path.getsize(f"index_files/vocab_{f.split('_')[1]}")
                             for f in os.listdir("index_files") if f.startswith("vocab_"))
        total_postings_size = sum(os.path.getsize(f"index_files/postings_{f.split('_')[1]}")
                                for f in os.listdir("index_files") if f.startswith("postings_"))
        
        print("\nIndex construction complete!")
        print(f"Number of terms: {len(all_terms)}")
        print(f"Total vocabulary size: {total_vocab_size / 1024:.2f} KB")
        print(f"Total postings size: {total_postings_size / 1024:.2f} KB")
        print(f"Document ID map size: {os.path.getsize('index_files/docid_map.pkl') / 1024:.2f} KB")
    
    def finalize(self):
        """Write final partial index and merge all indexes."""
        self._write_partial_index()  # Write any remaining terms
        self._merge_partial_indexes()

def main():
    # Create inverted index
    index = InvertedIndex(partial_index_size=50000)  # Write partial index every 50k terms
    
    # Process all documents in the DEV directory
    dev_directory = "DEV"
    for root, _, files in os.walk(dev_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Read and process document
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        index.process_document(data['url'], data['content'])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error processing {file_path}: {e}")
    
    # Finalize index
    index.finalize()

if __name__ == "__main__":
    main()
