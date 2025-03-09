import struct
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
import os
import time
import math
import sys

class DiskIndex:
    def __init__(self):
        # Load document ID mapping
        print("Loading document ID mapping...")
        with open("index_files/docid_map.pkl", "rb") as f:
            self.doc_id_map = pickle.load(f)
        
        # Calculate total number of documents for IDF
        self.total_docs = len(self.doc_id_map)
        
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
        
        # Each posting is 12 bytes (4 for doc_id, 4 for tf, 4 for importance)
        num_postings = len(postings_bytes) // 12
        postings = []
        
        for i in range(num_postings):
            start = i * 12
            doc_id_hash, tf, importance = struct.unpack("Iff", postings_bytes[start:start + 12])
            doc_id = self.doc_id_map[doc_id_hash]
            postings.append((doc_id, tf, importance))
        
        return postings
    
    def calculate_idf(self, term):
        """Calculate IDF for a term."""
        if term not in self.vocab_ranges:
            return 0
        df = self.vocab_ranges[term]['df']
        # Add 1 to denominator to prevent division by zero and smooth IDF
        return math.log10(self.total_docs / df) if df > 0 else 0
    
    def search(self, query):
        """Search for documents matching the query using TF-IDF and importance scoring."""
        start_time = time.time()
        
        # Tokenize and stem query
        query_terms = re.findall(r'[a-z0-9]+', query.lower())
        query_terms = [self.stemmer.stem(term) for term in query_terms]
        
        # Get postings and calculate scores for each query term
        results = defaultdict(float)
        for term in query_terms:
            if term in self.vocab_ranges:
                # Calculate IDF for term
                idf = self.calculate_idf(term)
                
                # Get postings
                postings = self.get_postings(term)
                for doc_id, tf, importance in postings:
                    # Score = TF * IDF * importance
                    # TF is already logarithmically scaled from indexer
                    score = tf * idf * importance
                    results[doc_id] += score
        
        # Sort by score
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        search_time = time.time() - start_time
        return sorted_results, search_time
    
    def close(self):
        """Close all postings files."""
        for file in self.postings_files.values():
            file.close()

    def run_query(self, query):
        results, search_time = self.search(query)
        print(f"\nSearch completed in {search_time*1000:.0f} milliseconds")
        print(f"Found {len(results)} matching documents:")
        for doc_id, score in results[:10]:  # Show top 10 results
            print(f"Score: {score:.4f} - Document: {doc_id}")

def main():
    index = DiskIndex()

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        for query in queries:
            print("\n" + query)
            index.run_query(query)
    else:
        while True:
            query = input("\nEnter search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            index.run_query(query)
    
    index.close()

if __name__ == "__main__":
    main()
