#!/usr/bin/env python3
"""
Preprocesses document vectors to remove near-duplicates.
This script computes cosine similarities between all document vectors and removes
those with similarity above a specified threshold (default 0.97).
"""

import pickle
import os
import math
import time
from tqdm import tqdm
import argparse

def calculate_magnitude(vector):
    """Calculate the magnitude of a document vector."""
    return math.sqrt(sum(val**2 for val in vector.values()))

def calculate_cosine_similarity(vec1, vec2, mag1=None, mag2=None):
    """Calculate cosine similarity between two document vectors."""
    if mag1 is None:
        mag1 = calculate_magnitude(vec1)
    if mag2 is None:
        mag2 = calculate_magnitude(vec2)
    
    # Find common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())
    
    if not common_terms:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    return dot_product / (mag1 * mag2)

def quick_similarity_check(vec1, vec2):
    """Apply quick heuristics to determine if full similarity calculation is needed."""
    # Size-based heuristic
    size_ratio = len(vec1) / len(vec2) if len(vec2) > len(vec1) else len(vec2) / len(vec1)
    if size_ratio < 0.5:  # If one vector has less than half the terms of the other
        return False
    
    # Check if they share enough common terms
    common_terms = len(set(vec1.keys()) & set(vec2.keys()))
    min_size = min(len(vec1), len(vec2))
    
    # If they share too few common terms relative to their sizes
    if common_terms < 3 or common_terms / min_size < 0.3:
        return False
    
    return True

def detect_and_remove_near_duplicates(doc_vectors, threshold=0.97, batch_size=1000):
    """
    Detect and remove near-duplicate documents based on cosine similarity.
    Uses batched processing to manage memory usage.
    
    Args:
        doc_vectors: Dictionary mapping doc_id -> {term -> tfidf}
        threshold: Similarity threshold above which docs are considered near-duplicates
        batch_size: Number of documents to process at once
        
    Returns:
        Filtered doc_vectors dictionary with near-duplicates removed
    """
    start_time = time.time()
    print(f"Starting near-duplicate detection with {len(doc_vectors)} documents...")
    
    # Precompute magnitudes to avoid redundant calculations
    print("Precomputing document vector magnitudes...")
    doc_magnitudes = {}
    for doc_id, vector in tqdm(doc_vectors.items()):
        mag = calculate_magnitude(vector)
        if mag > 0:  # Skip zero-magnitude vectors
            doc_magnitudes[doc_id] = mag
    
    # Filter out zero-magnitude vectors
    valid_docs = {doc_id: vector for doc_id, vector in doc_vectors.items() 
                  if doc_id in doc_magnitudes}
    
    print(f"Found {len(valid_docs)} valid documents with non-zero magnitudes")
    
    # Create a list of document IDs for easier batch processing
    doc_ids = list(valid_docs.keys())
    
    # Track documents to keep (start with all valid docs)
    docs_to_keep = set(doc_ids)
    duplicate_count = 0
    
    # Process in batches to limit memory usage
    for i in range(0, len(doc_ids), batch_size):
        batch_start = time.time()
        batch = doc_ids[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{len(doc_ids)//batch_size + 1} "
              f"({len(batch)} documents)")
        
        # Compare each document in the batch with all other documents
        for idx, doc_id in enumerate(tqdm(batch)):
            # Skip if already marked for removal
            if doc_id not in docs_to_keep:
                continue
                
            doc_vector = valid_docs[doc_id]
            doc_mag = doc_magnitudes[doc_id]
            
            # Compare with all other documents not yet processed in this batch
            for j in range(idx + 1, len(batch)):
                other_doc_id = batch[j]
                
                # Skip if already marked for removal
                if other_doc_id not in docs_to_keep:
                    continue
                
                other_vector = valid_docs[other_doc_id]
                other_mag = doc_magnitudes[other_doc_id]
                
                # Apply quick check before expensive similarity calculation
                if quick_similarity_check(doc_vector, other_vector):
                    similarity = calculate_cosine_similarity(
                        doc_vector, other_vector, doc_mag, other_mag
                    )
                    
                    if similarity >= threshold:
                        # Keep the document with more terms (richer content)
                        if len(doc_vector) >= len(other_vector):
                            docs_to_keep.discard(other_doc_id)
                        else:
                            docs_to_keep.discard(doc_id)
                            break  # No need to compare this document further
                        
                        duplicate_count += 1
        
        batch_time = time.time() - batch_start
        print(f"Batch processing completed in {batch_time:.2f} seconds")
    
    # Create filtered document vectors dictionary
    filtered_doc_vectors = {doc_id: doc_vectors[doc_id] for doc_id in docs_to_keep}
    
    total_time = time.time() - start_time
    print(f"Near-duplicate detection completed in {total_time:.2f} seconds")
    print(f"Removed {len(doc_vectors) - len(filtered_doc_vectors)} near-duplicate documents")
    print(f"Reduced document set from {len(doc_vectors)} to {len(filtered_doc_vectors)} documents")
    
    return filtered_doc_vectors

def main():
    parser = argparse.ArgumentParser(description='Preprocess document vectors to remove near-duplicates')
    parser.add_argument('--threshold', type=float, default=0.97,
                      help='Similarity threshold above which documents are considered duplicates (default: 0.97)')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Number of documents to process in each batch (default: 1000)')
    parser.add_argument('--input-path', type=str, default='index_files/doc_vectors.pkl',
                      help='Path to input document vectors file (default: index_files/doc_vectors.pkl)')
    parser.add_argument('--output-path', type=str, default=None,
                      help='Path to output filtered document vectors file (default: same as input)')
    
    args = parser.parse_args()
    
    # Set output path to input path if not specified
    if args.output_path is None:
        args.output_path = args.input_path
    
    # Load document vectors
    print(f"Loading document vectors from {args.input_path}...")
    try:
        with open(args.input_path, "rb") as f:
            doc_vectors = pickle.load(f)
        print(f"Loaded {len(doc_vectors)} document vectors")
    except FileNotFoundError:
        print(f"Error: Document vectors file not found at {args.input_path}")
        return
    except Exception as e:
        print(f"Error loading document vectors: {e}")
        return
    
    # Process document vectors to remove near-duplicates
    filtered_doc_vectors = detect_and_remove_near_duplicates(
        doc_vectors, args.threshold, args.batch_size
    )
    
    # Save filtered document vectors
    print(f"Saving filtered document vectors to {args.output_path}...")
    try:
        with open(args.output_path, "wb") as f:
            pickle.dump(filtered_doc_vectors, f)
        
        # Print file size
        file_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
        print(f"Filtered document vectors saved successfully")
        print(f"File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"Error saving filtered document vectors: {e}")
        
    print("Done!")

if __name__ == "__main__":
    main()
