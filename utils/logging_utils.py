import sys
import logging

# Configure basic logging to stderr for the server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

for module_name in ["mcp.server.lowlevel.server", "mcp.server.streamable_http", "sse_starlette.sse"]:
    logging.getLogger(module_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)