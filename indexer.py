import os
import json
import pickle
import struct
import warnings
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import re
from collections import defaultdict
import math
import heapq
import concurrent.futures
import threading
import time
from functools import lru_cache

# Filter out BeautifulSoup warnings
warnings.filterwarnings("ignore")

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
    def __init__(self, partial_index_size=50000, num_workers=8, batch_size=100):
        self.index = {}
        self.stemmer = PorterStemmer()
        self.partial_index_size = partial_index_size
        self.partial_index_count = 0
        self.important_tags = {
            'h1': 3.0, 'h2': 2.5, 'h3': 2.0,
            'title': 3.0, 'b': 1.5, 'strong': 1.5
        }
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.index_lock = threading.Lock()
        self.count_lock = threading.Lock()
        self.processed_docs = 0
        self.total_docs = 0
        self.doc_vectors = {}  # Store document vectors: doc_id -> {term -> tfidf}
        
    @lru_cache(maxsize=10000)
    def stem_word(self, word):
        """Cached stemming for better performance"""
        return self.stemmer.stem(word)
        
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
                    stemmed_words = [self.stem_word(word) for word in words]
                    for word in stemmed_words:
                        token_importance[word] = max(token_importance[word], importance)
            
            # Process all text
            text = soup.get_text()
        except:
            text = text  # If HTML parsing fails, use raw text
            
        # Find all words
        words = re.findall(r'[a-z0-9]+', text.lower())
        stemmed_words = [self.stem_word(word) for word in words]
        
        # For words not in important tags, ensure they have at least importance 1.0
        result = [(word, token_importance.get(word, 1.0)) for word in stemmed_words]
        return result, len(words)
    
    def add_tokens(self, tokens_with_importance, doc_length, doc_id, local_index=None):
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
        
        # Use a local index if provided (for threaded processing)
        target_index = local_index if local_index is not None else self.index
        
        # Add to inverted index with logarithmic scaling
        for token, freq in term_freq.items():
            if token not in target_index:
                target_index[token] = []
            
            # Use logarithmic scaling for term frequency: 1 + log(tf)
            # This dampens the effect of high frequency terms without penalizing long documents too much
            normalized_freq = (1 + math.log10(freq)) / math.log10(doc_length)
            target_index[token].append(Posting(doc_id, normalized_freq, term_importance[token]))
    
    def process_document(self, doc_id, content, local_index=None):
        """Process a document and add its tokens to the index."""
        tokens_with_importance, doc_length = self.clean_and_tokenize(content)
        self.add_tokens(tokens_with_importance, doc_length, doc_id, local_index)
        
        # Update processed docs counter
        with self.count_lock:
            self.processed_docs += 1
            if self.processed_docs % 1000 == 0:
                print(f"Processed {self.processed_docs}/{self.total_docs} documents ({self.processed_docs/self.total_docs*100:.1f}%)")
    
    def process_batch(self, batch):
        """Process a batch of documents and return the local index"""
        local_index = {}
        for doc_id, content in batch:
            self.process_document(doc_id, content, local_index)
        return local_index
    
    def merge_local_index(self, local_index):
        """Merge a local index into the main index"""
        with self.index_lock:
            for term, postings in local_index.items():
                if term not in self.index:
                    self.index[term] = []
                self.index[term].extend(postings)
            
            # Check if we need to write a partial index
            if len(self.index) >= self.partial_index_size:
                self._write_partial_index()
    
    def _write_partial_index(self):
        """Write current index to disk as a partial index."""
        if not self.index:
            return
            
        # Create partial indices directory if it doesn't exist
        os.makedirs("partial_indices", exist_ok=True)
        
        # Write partial index to disk
        partial_path = f"partial_indices/partial_{self.partial_index_count}.pkl"
        with open(partial_path, "wb") as f:
            pickle.dump(self.index, f)
        
        print(f"Wrote partial index {self.partial_index_count} with {len(self.index)} terms")
        self.partial_index_count += 1
        self.index.clear()
    
    def _merge_partial_indices(self):
        """Merge all partial indices using a multi-way merge with buffered I/O."""
        if not os.path.exists("partial_indices"):
            return
            
        print("Merging partial indices...")
        
        # Create output directories
        os.makedirs("index_files", exist_ok=True)
        
        # Initialize doc_id map
        doc_id_map = {}
        
        # Open all partial indices simultaneously
        partial_files = []
        term_queues = []  # List[TermEntry]
        buffer_size = 16384  # 16KB buffer for each file (increased from 8KB)
        
        for i in range(self.partial_index_count):
            f = open(f"partial_indices/partial_{i}.pkl", "rb")
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
        
        # Pre-calculate the ranges based on term distribution
        range_boundaries = []
        if all_terms:
            terms_per_range = (len(all_terms) + num_ranges - 1) // num_ranges
            for i in range(num_ranges):
                start_idx = i * terms_per_range
                if start_idx < len(all_terms):
                    end_idx = min((i + 1) * terms_per_range, len(all_terms))
                    start_term = all_terms[start_idx]
                    end_term = all_terms[end_idx - 1]
                    range_boundaries.append((f"{start_term[:2]}-{end_term[:2]}", start_term, end_term))
        
        def write_range_files():
            nonlocal current_range_terms, current_vocab_file, current_postings_file, current_range_name, current_range_idx
            if current_range_terms and current_range_idx < len(range_boundaries):
                current_range_name = range_boundaries[current_range_idx][0]
                
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
                current_range_idx += 1
        
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
            
            # Write postings in batches for better performance
            postings_buffer = bytearray(len(current_postings) * 12)  # Pre-allocate buffer
            offset = 0
            
            for posting in current_postings:
                doc_id_hash = hash(posting.doc_id) & 0xFFFFFFFF
                doc_id_map[doc_id_hash] = posting.doc_id
                struct.pack_into("Iff", postings_buffer, offset, doc_id_hash, posting.term_freq, posting.importance_score)
                offset += 12
            
            current_postings_file.write(postings_buffer)
            
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
        
        # Remove partial indices
        for i in range(self.partial_index_count):
            os.remove(f"partial_indices/partial_{i}.pkl")
        os.rmdir("partial_indices")
        
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
    
    def process_documents_parallel(self, document_list):
        """Process documents in parallel using a thread pool."""
        print(f"Processing {len(document_list)} documents using {self.num_workers} workers...")
        self.total_docs = len(document_list)
        self.processed_docs = 0
        
        # Create batches
        batches = []
        current_batch = []
        
        for doc_id, content in document_list:
            current_batch.append((doc_id, content))
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
                
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
            
        print(f"Created {len(batches)} batches of size {self.batch_size}")
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                local_index = future.result()
                self.merge_local_index(local_index)
    
    def finalize(self):
        """Write final partial index and merge all indices."""
        self._write_partial_index()  # Write any remaining terms
        self._merge_partial_indices()
        self._create_document_vectors()
        
    def _create_document_vectors(self):
        """Create document vectors with TF-IDF weights and save to file."""
        print("Creating document vectors...")
        
        # We need to calculate document vectors after the index is finalized
        # so that we have access to all terms and can calculate IDF properly
        
        # First, load the vocabulary to get document frequencies
        vocab_terms = {}  # term -> df
        for filename in os.listdir("index_files"):
            if filename.startswith("vocab_"):
                with open(f"index_files/{filename}", "r", encoding="utf-8") as f:
                    for line in f:
                        term, df, _, _ = line.strip().split("\t")
                        vocab_terms[term] = int(df)
        
        # Count total documents
        with open("index_files/docid_map.pkl", "rb") as f:
            docid_map = pickle.load(f)
        total_docs = len(docid_map)
        
        # Create vectors for all documents
        doc_vectors = {}
        
        # Iterate through all vocabulary files
        for filename in os.listdir("index_files"):
            if not filename.startswith("vocab_"):
                continue
                
            range_name = filename.split("_")[1].split(".")[0]
            postings_path = f"index_files/postings_{range_name}.bin"
            
            with open(f"index_files/{filename}", "r", encoding="utf-8") as vocab_file:
                with open(postings_path, "rb") as postings_file:
                    for line in vocab_file:
                        term, df, offset, length = line.strip().split("\t")
                        df = int(df)
                        offset = int(offset)
                        length = int(length)
                        
                        # Calculate IDF for this term
                        idf = math.log10(total_docs / df) if df > 0 else 0
                        
                        # Read postings
                        postings_file.seek(offset)
                        postings_bytes = postings_file.read(length)
                        
                        # Process each posting
                        for i in range(length // 12):
                            start = i * 12
                            doc_id_hash, tf, importance = struct.unpack("Iff", postings_bytes[start:start + 12])
                            doc_id = docid_map[doc_id_hash]
                            
                            # Calculate TF-IDF score
                            tfidf = tf * idf * importance
                            
                            # Add to document vector
                            if doc_id not in doc_vectors:
                                doc_vectors[doc_id] = {}
                            doc_vectors[doc_id][term] = tfidf
        
        # Save document vectors
        print(f"Saving {len(doc_vectors)} document vectors...")
        with open("index_files/doc_vectors.pkl", "wb") as f:
            pickle.dump(doc_vectors, f)
        
        print(f"Document vectors saved successfully, size: {os.path.getsize('index_files/doc_vectors.pkl') / (1024*1024):.2f} MB")

def main():
    start_time = time.time()
    
    # Create inverted index with 8 worker threads and batch size of 100
    # Adjust the number of workers based on your CPU cores
    index = InvertedIndex(partial_index_size=50000, num_workers=8, batch_size=100)
    
    # Collect all documents before processing
    documents = []
    dev_directory = "DEV"
    
    print("Collecting documents...")
    for root, _, files in os.walk(dev_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Read document
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        documents.append((data['url'], data['content']))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Process all documents in parallel
    index.process_documents_parallel(documents)
    
    # Finalize index
    index.finalize()
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\nTotal indexing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
