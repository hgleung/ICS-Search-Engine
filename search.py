import struct
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
import re
import os
import time
import math
import sys
from urllib.parse import urldefrag

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
        
        # Load document vectors for duplicate detection
        print("Loading document vectors...")
        if os.path.exists("index_files/doc_vectors.pkl"):
            with open("index_files/doc_vectors.pkl", "rb") as f:
                self.doc_vectors = pickle.load(f)
            print(f"Loaded {len(self.doc_vectors)} document vectors")
        else:
            print("Document vectors file not found, near-duplicate detection disabled")
            self.doc_vectors = {}
    
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
        seen_urls = set()  # Track seen defragged URLs
        
        for i in range(num_postings):
            start = i * 12
            doc_id_hash, tf, importance = struct.unpack("Iff", postings_bytes[start:start + 12])
            doc_id = self.doc_id_map[doc_id_hash]
            
            # Defragment URL
            defragged_url, _ = urldefrag(doc_id)
            
            # Skip if we've seen this URL before
            if defragged_url not in seen_urls:
                seen_urls.add(defragged_url)
                postings.append((defragged_url, tf, importance))
        
        return postings
    
    def calculate_idf(self, term):
        """Calculate IDF for a term."""
        if term not in self.vocab_ranges:
            return 0
        df = self.vocab_ranges[term]['df']
        # Add 1 to denominator to prevent division by zero and smooth IDF
        return math.log10(self.total_docs / df) if df > 0 else 0
    
    def calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two document vectors."""
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def _calculate_doc_vector_magnitude(self, vec):
        """Calculate magnitude of a document vector."""
        return math.sqrt(sum(val**2 for val in vec.values()))
    
    def _calculate_vector_similarity_optimized(self, vec1, vec2, mag1, mag2):
        """Optimized version of cosine similarity calculation."""
        # Quick size-based heuristic check
        size_ratio = len(vec1) / len(vec2) if len(vec2) > len(vec1) else len(vec2) / len(vec1)
        if size_ratio < 0.5:  # If one vector has less than half the terms of the other
            return 0.0
        
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        # Quick check on number of common terms
        if len(common_terms) < 3:
            return 0.0
            
        # Calculate dot product for common terms only
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def filter_near_duplicates(self, results, threshold=0.95):
        """Remove near-duplicate documents from results based on cosine similarity."""
        # If we don't have document vectors, return original results
        if not self.doc_vectors:
            return results
            
        print("Filtering near-duplicate documents...")
        start_time = time.time()
        
        # If there are very few results, just use the original method
        if len(results) <= 10:
            filtered_results = []
            seen_docs = set()
            
            for doc_id, score in results:
                if doc_id in seen_docs:
                    continue
                    
                if doc_id not in self.doc_vectors:
                    filtered_results.append((doc_id, score))
                    continue
                    
                doc_vector = self.doc_vectors[doc_id]
                is_duplicate = False
                
                for existing_doc_id, _ in filtered_results:
                    if existing_doc_id in self.doc_vectors:
                        existing_vector = self.doc_vectors[existing_doc_id]
                        similarity = self.calculate_cosine_similarity(doc_vector, existing_vector)
                        
                        if similarity >= threshold:
                            is_duplicate = True
                            seen_docs.add(doc_id)
                            break
                
                if not is_duplicate:
                    filtered_results.append((doc_id, score))
            
            filter_time = time.time() - start_time
            print(f"Filtered {len(results) - len(filtered_results)} near-duplicate documents in {filter_time:.4f} seconds")
            return filtered_results
        
        # For larger result sets, use optimized approach
        
        # Step 1: Precompute vector magnitudes to avoid redundant calculations
        doc_magnitudes = {}
        valid_docs = []
        
        # Filter out documents without vectors and precompute magnitudes
        for doc_id, score in results:
            if doc_id in self.doc_vectors:
                doc_vector = self.doc_vectors[doc_id]
                if doc_vector:  # Only process non-empty vectors
                    mag = self._calculate_doc_vector_magnitude(doc_vector)
                    if mag > 0:  # Skip zero-magnitude vectors
                        doc_magnitudes[doc_id] = mag
                        valid_docs.append((doc_id, score))
            else:
                # Keep documents without vectors (no dupe filtering for these)
                valid_docs.append((doc_id, score))
        
        # Step 2: Sort documents by score (highest first) to prioritize higher scoring docs
        valid_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Create filtered results using optimized similarity check
        filtered_results = []
        seen_docs = set()
        
        for doc_id, score in valid_docs:
            # Skip if already marked as duplicate
            if doc_id in seen_docs:
                continue
                
            # If no vector, keep the document
            if doc_id not in doc_magnitudes:
                filtered_results.append((doc_id, score))
                continue
                
            doc_vector = self.doc_vectors[doc_id]
            doc_mag = doc_magnitudes[doc_id]
            is_duplicate = False
            
            # Check against already accepted documents
            for existing_doc_id, _ in filtered_results:
                # Skip documents without vectors
                if existing_doc_id not in doc_magnitudes:
                    continue
                
                existing_vector = self.doc_vectors[existing_doc_id]
                existing_mag = doc_magnitudes[existing_doc_id]
                
                # Use optimized similarity check
                similarity = self._calculate_vector_similarity_optimized(
                    doc_vector, existing_vector, doc_mag, existing_mag
                )
                
                if similarity >= threshold:
                    is_duplicate = True
                    seen_docs.add(doc_id)
                    break
            
            # Add if not a duplicate
            if not is_duplicate:
                filtered_results.append((doc_id, score))
        
        filter_time = time.time() - start_time
        print(f"Filtered {len(results) - len(filtered_results)} near-duplicate documents in {filter_time:.4f} seconds")
        
        return filtered_results
    
    def search(self, query, k=1000):
        """Search for documents matching the query using TF-IDF and importance scoring."""
        start_time = time.time()
        
        # Tokenize and stem query
        query_terms = re.findall(r'[a-z0-9]+', query.lower())
        query_terms = [self.stemmer.stem(term) for term in query_terms]
        
        # Filter to terms in vocabulary and sort by ascending df
        query_terms_with_df = [(term, self.vocab_ranges[term]['df']) 
                             for term in query_terms if term in self.vocab_ranges]
        query_terms_with_df.sort(key=lambda x: x[1])  # Sort by df
        
        # Get postings and calculate scores for each query term
        results = defaultdict(float)
        max_possible_remaining_score = 0
        min_score_in_top_k = 0
        processed_terms = 0
        
        for term, df in query_terms_with_df:
            # Calculate maximum possible score contribution from remaining terms
            # Since terms are sorted by df, all remaining terms have df >= current df
            # Maximum TF is pre-normalized as (1 + log10(freq)) / log10(1 + doc_length)
            # Maximum importance score is 1.0
            idf_current = math.log10(self.total_docs / df) if df > 0 else 0
            max_possible_remaining_score = 0
            
            # Calculate max possible score from remaining terms including current
            remaining_terms = len(query_terms_with_df) - processed_terms
            if remaining_terms > 0:
                # Conservative estimate: assume remaining terms have same df as current
                # and max normalized TF (1.0) and max importance (1.0)
                max_possible_remaining_score = remaining_terms * (1.0 * idf_current * 1.0)
            
            # If we have at least k results and max possible remaining score
            # can't change top k, we can stop
            if len(results) >= k and max_possible_remaining_score < min_score_in_top_k:
                break
                
            # Calculate IDF for term
            idf = self.calculate_idf(term)
            
            # Get postings (already deduplicated)
            postings = self.get_postings(term)
            for doc_id, tf, importance in postings:
                # Score = TF * IDF * importance
                # TF is already logarithmically scaled from indexer
                score = tf * idf * importance
                results[doc_id] += score
            
            # Update minimum score in top k if we have enough results
            if len(results) >= k:
                min_score_in_top_k = sorted(results.values(), reverse=True)[k-1]
            
            processed_terms += 1

        search_time = time.time() - start_time
        
        # Sort by score and remove duplicates with identical scores
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out documents with identical scores (keeping first occurrence)
        unique_results = []
        seen_scores = set()
        for doc_id, score in sorted_results:
            # Round to handle floating point precision
            rounded_score = round(score, 6)
            if rounded_score not in seen_scores:
                unique_results.append((doc_id, score))
                seen_scores.add(rounded_score)
                if len(unique_results) >= k:
                    break
        
        # Filter near-duplicate documents
        filtered_results = self.filter_near_duplicates(unique_results, threshold=0.95)
        
        return filtered_results, search_time
    
    def run_query(self, query):
        results, search_time = self.search(query)
        print(f"\nSearch completed in {search_time:.4f} seconds")
        print(f"Found {len(results)} matching documents:")
        for doc_id, score in results[:10]:  # Show top 10 results
            print(f"Score: {score:.4f} - Document: {doc_id}")

    def close(self):
        """Close all postings files."""
        for file in self.postings_files.values():
            file.close()

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
