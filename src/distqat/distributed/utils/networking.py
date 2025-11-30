import socket
from contextlib import closing
from typing import Optional

from hivemind.utils.logging import get_logger

Hostname, Port = str, int  # flavour types
Endpoint = str  # e.g. 1.2.3.4:1337 or [2a21:6с8:b192:2105]:8888, https://networkengineering.stackexchange.com/a/9435

logger = get_logger(__name__)

def get_port(endpoint: Endpoint) -> Optional[Port]:
    """get port or None if port is undefined"""
    try:
        return int(endpoint[endpoint.rindex(":") + 1 :], base=10)
    except ValueError:  # :* or not specified
        return None


def replace_port(endpoint: Endpoint, new_port: Port) -> Endpoint:
    assert endpoint.endswith(":*") or get_port(endpoint) is not None, endpoint
    return f"{endpoint[:endpoint.rindex(':')]}:{new_port}"


def strip_port(endpoint: Endpoint) -> Hostname:
    """Removes port from the end of endpoint. If port is not specified, does nothing"""
    maybe_port = endpoint[endpoint.rindex(":") + 1 :]
    return endpoint[: endpoint.rindex(":")] if maybe_port.isdigit() or maybe_port == "*" else endpoint


def get_free_port(params=(socket.AF_INET, socket.SOCK_STREAM), opt=(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)):
    """
    Finds a tcp port that can be occupied with a socket with *params and use *opt options.

    :note: Using this function is discouraged since it often leads to a race condition
           with the "Address is already in use" error if the code is run in parallel.
    """
    try:
        with closing(socket.socket(*params)) as sock:
            sock.bind(("", 0))
            sock.setsockopt(*opt)
            return sock.getsockname()[1]
    except Exception as e:
        raise e