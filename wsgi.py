"""Entry point for a hosting server: gunicorn wsgi:app"""
from kept.web.app import create_app

app = create_app()
