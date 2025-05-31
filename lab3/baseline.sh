echo "DistilBERT with CLS method"
python lab3/main_extract.py --extract --extractor distilbert --method cls

echo "DistilBERT with mean pooling"
python lab3/main_extract.py --extract --extractor distilbert --method mean

echo "SBERT (all-MiniLM-L6-v2)"
python lab3/main_extract.py --extract --extractor sbert --method minilm

echo "SBERT (all-mpnet-base-v2)"
python lab3/main_extract.py --extract --extractor sbert --method mpnet