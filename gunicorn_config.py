"""
Optimized Gunicorn config for Render deployment
Focus on memory optimization and stability
"""
import os
import sys

# Get port from environment variable (critical for Render)
port = int(os.environ.get('PORT', '5000'))

# Explicitly print port information for debugging
print(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
print(f"Using port: {port}")

# Server settings
bind = f"0.0.0.0:{port}"  # Ensure we bind to the exact port Render expects
worker_class = "sync"  # Use sync workers for reliability
workers = 1  # Use just 1 worker for memory-constrained environments
threads = 2  # Use minimal threads for parallel processing

# Resource limits
timeout = 120  # Allow up to 120 seconds for a request
graceful_timeout = 30  # Wait up to 30s for workers
max_requests = 200  # Restart workers after this many requests
max_requests_jitter = 50  # Add randomness to prevent simultaneous restarts
keepalive = 5  # Keep connections alive for 5s

# Process naming
proc_name = 'platedetector'

# Logging
loglevel = "info"
accesslog = '-'
errorlog = '-'

# Lifecycle hooks
def on_starting(server):
    """Actions before the server starts."""
    print(f"Starting Gunicorn server at 0.0.0.0:{port}")
    print(f"Workers: {workers}, Threads: {threads}, Timeout: {timeout}s")
    
    # Turn on garbage collector
    import gc
    gc.enable()
    gc.collect()
    
    # Make sure model directory exists
    import os
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

def post_fork(server, worker):
    """Actions after forking a worker."""
    print(f"Worker {worker.pid} initialized and serving on 0.0.0.0:{port}")
    sys.stdout.flush()  # Ensure output is displayed in logs
