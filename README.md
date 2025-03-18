# ICS Search Engine

A high-performance search engine implementation designed to efficiently index and search through web documents. The system uses TF-IDF scoring with importance weighting and near-duplicate detection to provide relevant search results.

## Project Structure

The project consists of three main Python scripts:

- `indexer.py`: Creates an inverted index from web documents
- `search.py`: Performs search queries on the indexed documents
- `filter_duplicates.py`: Removes near-duplicate documents to improve search quality

## Features

### 1. Efficient Indexing
- Multi-threaded document processing
- Partial index writing to manage memory usage
- Smart tokenization with HTML tag importance weighting
- URL normalization and deduplication
- Logarithmic term frequency scaling

### 2. Advanced Search
- TF-IDF scoring with importance weighting
- Fast disk-based index lookup
- Support for both interactive and batch queries
- Results grouped by relevance score

### 3. Near-Duplicate Detection
- Cosine similarity-based document comparison
- Efficient batched processing
- Quick similarity heuristics
- Configurable similarity threshold
- Memory-efficient implementation

## Generated Files

The indexing process creates several files in the `index_files` directory:

- `vocab_*.txt`: Vocabulary files containing term -> (df, offset, length) mappings
- `postings_*.bin`: Binary files containing the actual postings lists
- `docid_map.pkl`: Pickle file mapping hashed document IDs to original URLs
- `doc_vectors.pkl`: Document vector representations for duplicate detection

## Usage

### 1. Indexing Documents

```bash
python indexer.py [input_directory]
```

The indexer processes HTML documents and creates the necessary index files. It supports:
- Multi-threaded processing
- Partial index writing
- Duplicate URL detection
- HTML tag importance weighting

### 2. Searching

```bash
python search.py [optional: query_file]
```

Two modes available:
- Interactive mode: Enter queries directly
- Batch mode: Process queries from a file

### 3. Duplicate Detection

```bash
python filter_duplicates.py [options]
```

Options:
- `--threshold`: Similarity threshold (default: 0.99)
- `--batch-size`: Batch size for processing (default: 1000)
- `--input-path`: Input document vectors file
- `--output-path`: Output filtered vectors file

## Implementation Details

### Indexing Process
1. Documents are processed in parallel using multiple worker threads
2. Terms are stemmed using Porter Stemmer
3. HTML tags are weighted for importance (e.g., title, headers)
4. URLs are normalized to prevent duplicates
5. Term frequencies are logarithmically scaled
6. Partial indices are written to disk when memory threshold is reached
7. Final multi-way merge combines partial indices

### Search Process
1. Query terms are stemmed
2. Postings lists are retrieved from disk
3. TF-IDF scores are calculated with importance weighting
4. Results are grouped by score to handle duplicates
5. Top results are returned sorted by relevance

### Duplicate Detection
1. Document vectors are compared using cosine similarity
2. Quick heuristics filter out obvious non-duplicates
3. Batched processing manages memory usage
4. Documents with similarity above threshold are considered duplicates
5. The richer document (more terms) is kept when duplicates are found

## Performance Considerations

- Disk-based index for memory efficiency
- Binary postings format for compact storage
- Batched processing in duplicate detection
- Multi-threaded document processing
- Partial index writing to manage memory
- Quick similarity heuristics to reduce comparisons

## Requirements

- Python 3.x
- NLTK (for Porter Stemmer)
- BeautifulSoup4 (for HTML parsing)
- tqdm (for progress bars)

## Error Handling

The system includes robust error handling for:
- Invalid HTML documents
- Missing files
- Memory constraints
- Disk I/O errors
- Invalid queries