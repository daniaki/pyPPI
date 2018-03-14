# F1 main 
echo "Running Baseline experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --verbose --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=baseline

echo "Running Random Forest experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --verbose --model=RandomForestClassifier --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=randomforest

echo "Running Ternary encoding experiment"
python validation.py --interpro --pfam --cc --bp --mf --verbose --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=ternary

echo "Running Inducer experiment"
python validation.py --interpro --pfam --cc --bp --mf --binary --verbose --induce --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=inducer

# Individual
echo "Running isolated GO features experiment"
python validation.py --cc --bp --mf --binary --verbose --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=isolated_go

echo "Running isolated induced GO features experiment"
python validation.py --cc --bp --mf --binary --induce --verbose --model=LogisticRegression --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=isolated_inducer

echo "Running isolated InterPro features experiment"
python validation.py --interpro --binary --model=LogisticRegression --verbose --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=isolated_interpro

echo "Running isolated Pfam features experiment"
python validation.py --pfam --binary --model=LogisticRegression --verbose --n_jobs=4 --h_iterations=30 --n_iterations=3 --n_splits=5 --output_folder=isolated_pfam