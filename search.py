import struct
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
import os
import time

class DiskIndex:
    def __init__(self):
        # Load document ID mapping
        print("Loading document ID mapping...")
        with open("index_files/docid_map.pkl", "rb") as f:
            self.doc_id_map = pickle.load(f)
        
        # Load vocabulary ranges
        print("Loading vocabulary ranges...")
        self.vocab_ranges = {}  # term -> (vocab_file, postings_file)
        self.postings_files = {}  # range_name -> file handle
        
        # Load all vocabulary files
        for filename in os.listdir("index_files"):
            if filename.startswith("vocab_"):
                range_name = filename.split("_")[1].split(".")[0]
                vocab_path = f"index_files/{filename}"
                postings_path = f"index_files/postings_{range_name}.bin"
                
                # Open postings file
                self.postings_files[range_name] = open(postings_path, "rb")
                
                # Load vocabulary
                with open(vocab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        term, df, offset, length = line.strip().split("\t")
                        self.vocab_ranges[term] = {
                            'range': range_name,
                            'df': int(df),
                            'offset': int(offset),
                            'length': int(length)
                        }
        
        self.stemmer = PorterStemmer()
    
    def get_postings(self, term):
        """Get postings list for a term from disk."""
        # Stem the term
        term = self.stemmer.stem(term.lower())
        
        if term not in self.vocab_ranges:
            return []
        
        # Get posting list location
        vocab_entry = self.vocab_ranges[term]
        range_name = vocab_entry['range']
        offset = vocab_entry['offset']
        length = vocab_entry['length']
        
        # Get correct postings file
        postings_file = self.postings_files[range_name]
        
        # Seek to position and read postings
        postings_file.seek(offset)
        postings_bytes = postings_file.read(length)
        
        # Each posting is 8 bytes (4 for doc_id, 4 for tf)
        num_postings = len(postings_bytes) // 8
        postings = []
        
        for i in range(num_postings):
            start = i * 8
            doc_id_hash, tf = struct.unpack("II", postings_bytes[start:start + 8])
            doc_id = self.doc_id_map[doc_id_hash]
            postings.append((doc_id, tf))
        
        return postings
    
    def search(self, query):
        """Search for documents matching the query."""
        start_time = time.time()
        
        # Tokenize and stem query
        query_terms = re.findall(r'[a-z0-9]+', query.lower())
        query_terms = [self.stemmer.stem(term) for term in query_terms]
        
        # Get postings for each query term
        results = defaultdict(int)
        for term in query_terms:
            if term in self.vocab_ranges:
                postings = self.get_postings(term)
                for doc_id, tf in postings:
                    # Simple tf scoring
                    results[doc_id] += tf
        
        # Sort by score
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        search_time = time.time() - start_time
        return sorted_results, search_time
    
    def close(self):
        """Close all postings files."""
        for file in self.postings_files.values():
            file.close()

def main():
    # Example usage
    index = DiskIndex()
    
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        results, search_time = index.search(query)
        print(f"\nSearch completed in {search_time:.4f} seconds")
        print(f"Found {len(results)} matching documents:")
        for doc_id, score in results[:10]:  # Show top 10 results
            print(f"Score: {score:5d} - Document: {doc_id}")
    
    index.close()

if __name__ == "__main__":
    main()
