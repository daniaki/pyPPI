# F1 main 
echo "Running Baseline experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3

echo "Running Ternary encoding experiment"
python validation.py --interpro --pfam --cc --bp --mf --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3

echo "Running Inducer experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --induce --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3

# Individual
echo "Running isolated GO features experiment"
python validation.py --cc --bp --mf --binary --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3

echo "Running isolated induced GO features experiment"
python validation.py --cc --bp --mf --binary --induce --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3

echo "Running isolated InterPro features experiment"
python validation.py --interpro --binary --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3

echo "Running isolated Pfam features experiment"
python validation.py --pfam --binary --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3