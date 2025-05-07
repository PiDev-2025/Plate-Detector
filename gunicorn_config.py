# Gunicorn configuration file
import os
import multiprocessing

# Gunicorn config variables
loglevel = "info"
workers = 1  # Use just 1 worker for memory-constrained environments
timeout = 300  # Increased timeout to allow for model download
keepalive = 5
worker_class = "sync"  # Use sync workers for better memory management
worker_connections = 100
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
reload = False  # Disable auto-reload in production
