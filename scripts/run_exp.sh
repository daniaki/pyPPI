# F1 main 
echo "Running Baseline experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --verbose --model=LogisticRegression --n_jobs=16 --h_iterations=30 --n_iterations=3

echo "Running Random Forest experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --verbose --model=RandomForestClassifier --n_jobs=16 --h_iterations=30 --n_iterations=3

echo "Running Ternary encoding experiment"
python validation.py --interpro --pfam --cc --bp --mf --verbose --model=LogisticRegression --n_jobs=16 --h_iterations=30 --n_iterations=3

echo "Running Inducer experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --verbose --induce --model=LogisticRegression --n_jobs=16 --h_iterations=30 --n_iterations=3

# Individual
echo "Running isolated GO features experiment"
python validation.py --cc --bp --mf --binary --verbose --model=LogisticRegression --n_jobs=16 --h_iterations=30 --n_iterations=3

echo "Running isolated induced GO features experiment"
python validation.py --cc --bp --mf --binary --induce --verbose --model=LogisticRegression --n_jobs=16 --h_iterations=30 --n_iterations=3

echo "Running isolated InterPro features experiment"
python validation.py --interpro --binary --model=LogisticRegression --verbose --n_jobs=16 --h_iterations=30 --n_iterations=3

echo "Running isolated Pfam features experiment"
python validation.py --pfam --binary --model=LogisticRegression --verbose --n_jobs=16 --h_iterations=30 --n_iterations=3