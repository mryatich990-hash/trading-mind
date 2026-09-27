web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 dashboard.app:app
worker: python main.py
