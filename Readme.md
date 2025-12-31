# RAG-Based Test Case Generator

A Retrieval-Augmented Generation (RAG) system that automatically generates comprehensive test cases from your documentation using Azure OpenAI and vector search.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Tesseract OCR (for image processing):**

   **Windows:**
   - Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to PATH, or set path in code
     ```bash
     winget install --id UB-Mannheim.TesseractOCR
     ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```



3. **Ingest your documents:**
```bash
python app.py ingest ./sample_data
```

4. **Generate test cases:**
```bash
python app.py query "flight booking search filters" --top-k 3 --output results.json
```

**Common options:**
- `--top-k N` - Number of relevant chunks to retrieve (default: 5)
- `--min-score X` - Minimum similarity threshold (default: 0.3)
- `--debug` - Show detailed retrieval information
- `--output FILE` - Save results to JSON file

## File Structure

```
devassure2/
├── app.py                          # Main application entry point
├── setup_azure_env.ps1             # Azure OpenAI environment setup
├── azure_test.py                   # Test Azure OpenAI connection
├── requirements.txt                # Python dependencies
├── results.json                    # Sample output (generated test cases)
├── src/
│   ├── ingestion/
│   │   ├── document_processor.py  # Handles PDF, DOCX, images (OCR)
│   │   └── chunking_strategy.py   # Smart document chunking
│   ├── retrieval/
│   │   └── retriever.py            # Hybrid search (vector + BM25 + HYDE)
│   ├── generation/
│   │   └── use_case_generator.py  # Azure OpenAI test case generation
│   ├── guards/
│   │   └── safety_checks.py       # Hallucination detection & safety
│   └── utils/
│       ├── config.py               # Configuration management
│       └── logger.py               # Logging setup
└── data/
    └── chroma_db/                  # Vector database storage
```

## Testing

1. **Test Azure OpenAI connection:**
   ```bash
   python azure_test.py
   # Output: ✓ Authentication successful!
   ```

2. **Test retrieval and check chunks:**
   ```bash
   # Use --debug to see retrieved chunks and their relevance scores
   python app.py query "your query" --debug --top-k 5
   ```
   This shows:
   - Which chunks were retrieved from your documents
   - Relevance scores for each chunk
   - Source files for each chunk
   - HYDE (hypothetical document) generation
   - Reranking results

3. **View results:**
   - **`results.json`** - Generated test cases in JSON format
   - Console output shows full details including assumptions and missing info

## How It Works

1. **Ingest**: Documents are processed, chunked, and embedded into a vector database
2. **Retrieve**: User query triggers hybrid search (semantic + keyword) with HYDE enhancement
3. **Generate**: Azure OpenAI creates structured test cases from retrieved context
4. **Validate**: Safety checks prevent hallucinations and ensure quality

## Example Output

```json
{
  "use_cases": [
    {
      "id": "UC001",
      "title": "Apply multiple flight filters",
      "goal": "Verify that applying multiple filters updates results dynamically",
      "preconditions": ["User is on search results page"],
      "test_data": {"price_range": "100-500", "stops": "Non-stop"},
      "steps": ["Step 1: Apply price filter...", "Step 2: Apply stops filter..."],
      "expected_results": ["Results update to show matching flights"],
      "test_type": "positive",
      "priority": "high",
      "tags": ["filters", "search"]
    }
  ],
  "assumptions": ["Filters are functional"],
  "missing_info": ["Specific UI element names"]
}
```

## Configuration


Set your API key via environment variable: `AZURE_OPENAI_API_KEY`
