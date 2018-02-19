# # F1 main 
# echo "Running final model predictions"
# python predict_ppis.py --interpro --pfam --cc --bp --mf --n_jobs=4 --n_iterations=30 --n_splits=3

# # Individual
# echo "Running isolated GO features experiment"
# python predict_ppis.py --cc --bp --mf --n_jobs=4 --n_iterations=30 --n_splits=3

echo "Running isolated InterPro features experiment"
python predict_ppis.py --interpro --n_jobs=4 --n_iterations=30 --n_splits=1

echo "Running isolated Pfam features experiment"
python predict_ppis.py --pfam --n_jobs=4 --n_iterations=30 --n_splits=3
