# F1 main 
echo "Running baseline model predictions"
python predict.py --interpro --pfam --cc --bp --mf --verbose --n_jobs=4 --n_iterations=60 --n_splits=3 --retrain --output_folder=baseline

echo "Running paper model predictions" 
python predict.py --interpro --pfam --cc --bp --mf --verbose --n_jobs=4 --n_iterations=60 --n_splits=3 --retrain --model=paper --output_folder=paper

# Individual
echo "Running isolated GO features experiment"
python predict.py --cc --bp --mf --n_jobs=4 --verbose --n_iterations=60 --n_splits=3 --retrain --model=paper --output_folder=paper_go

echo "Running isolated InterPro features experiment"
python predict.py --interpro --n_jobs=4 --verbose --n_iterations=60 --n_splits=3 --model=paper --retrain --output_folder=paper_interpro

echo "Running isolated Pfam features experiment"
python predict.py --pfam --n_jobs=4 --verbose --n_iterations=60 --n_splits=3 --model=paper --retrain --output_folder=paper_pfam
