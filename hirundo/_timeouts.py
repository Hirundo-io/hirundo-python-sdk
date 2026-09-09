import httpx

READ_TIMEOUT = 30.0
MODIFY_TIMEOUT = 60.0
DOWNLOAD_READ_TIMEOUT = 600.0  # 10 minutes

# Run watchers may remain idle while waiting for a terminal event, so SSE reads
# must stay unbounded. Reuse the standard read timeout for connection setup.
SSE_TIMEOUT = httpx.Timeout(None, connect=READ_TIMEOUT)
