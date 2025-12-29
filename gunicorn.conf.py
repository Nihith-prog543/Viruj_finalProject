# Gunicorn configuration for memory-optimized deployment
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")
backlog = 2048

# Worker processes - use only 1 worker to minimize memory usage
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 5

# Memory optimization
max_requests = 1000  # Restart workers after this many requests to prevent memory leaks
max_requests_jitter = 50  # Add randomness to prevent all workers restarting at once
preload_app = False  # Don't preload app to reduce initial memory usage

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Process naming
proc_name = "synthesis_route_finder"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for Railway)
keyfile = None
certfile = None

