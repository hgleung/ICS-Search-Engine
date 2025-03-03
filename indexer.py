import os
import json
import pickle
import struct
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import re
from collections import defaultdict

class Posting:
    def __init__(self, doc_id, term_freq=1, importance_score=1.0):
        self.doc_id = doc_id
        self.term_freq = term_freq
        self.importance_score = importance_score

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
        """Add tokens from a document to the index with normalized frequencies."""
        # Count frequency and track importance of each token in document
        term_freq = defaultdict(int)
        term_importance = defaultdict(float)
        
        for token, importance in tokens_with_importance:
            term_freq[token] += 1
            term_importance[token] = max(term_importance[token], importance)
        
        # Add to inverted index with normalized frequencies
        for token, freq in term_freq.items():
            if token not in self.index:
                self.index[token] = []
            
            # Normalize frequency by document length and multiply by importance
            normalized_freq = (freq / doc_length) * term_importance[token]
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
        """Merge all partial indexes and split into term ranges."""
        if not os.path.exists("partial_indexes"):
            return
            
        print("Merging partial indexes...")
        merged_index = defaultdict(list)
        
        # Load and merge all partial indexes
        for i in range(self.partial_index_count):
            partial_path = f"partial_indexes/partial_{i}.pkl"
            with open(partial_path, "rb") as f:
                partial_index = pickle.load(f)
                for term, postings in partial_index.items():
                    merged_index[term].extend(postings)
            
            # Remove partial index file
            os.remove(partial_path)
        
        # Create ranges for terms
        all_terms = sorted(merged_index.keys())
        range_size = len(all_terms) // 4  # Split into 4 ranges
        
        # Create output directories
        os.makedirs("index_files", exist_ok=True)
        
        # Process each range
        for i in range(0, len(all_terms), range_size):
            range_terms = all_terms[i:i+range_size]
            if not range_terms:
                continue
                
            # Determine range name
            start_term = range_terms[0]
            end_term = range_terms[-1]
            range_name = f"{start_term[:2]}-{end_term[:2]}"
            
            # Open files for this range
            vocab_path = f"index_files/vocab_{range_name}.txt"
            postings_path = f"index_files/postings_{range_name}.bin"
            
            with open(vocab_path, "w", encoding="utf-8") as vocab_file, \
                 open(postings_path, "wb") as postings_file:
                
                # Process each term in this range
                for term in range_terms:
                    postings = merged_index[term]
                    
                    # Record start position for postings
                    start_pos = postings_file.tell()
                    
                    # Write postings
                    for posting in postings:
                        # Hash document ID to 32-bit unsigned int
                        doc_id_hash = hash(posting.doc_id) & 0xFFFFFFFF
                        
                        # Write doc_id_hash, term_freq, and importance_score
                        postings_file.write(struct.pack("Iff", doc_id_hash, posting.term_freq, posting.importance_score))
                    
                    # Calculate postings length
                    postings_length = postings_file.tell() - start_pos
                    
                    # Write vocabulary entry
                    vocab_file.write(f"{term}\t{len(postings)}\t{start_pos}\t{postings_length}\n")
            
            print(f"Created index files for range {range_name}")
        
        # Create document ID mapping
        doc_id_map = {}
        for postings in merged_index.values():
            for posting in postings:
                doc_id_hash = hash(posting.doc_id) & 0xFFFFFFFF
                doc_id_map[doc_id_hash] = posting.doc_id
        
        # Save document ID mapping
        with open("index_files/docid_map.pkl", "wb") as f:
            pickle.dump(doc_id_map, f)
        
        # Remove partial indexes directory
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
