echo isort
isort mast.py core/
echo black
black .
echo flake8
flake8 mast.py core/
echo mypy
mypy mast.py core/
