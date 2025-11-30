from typing import Callable, Optional
from typing import Optional, Callable, Awaitable
import asyncio
import asyncio
import asyncssh
import base64
import logging

from .config import Server

logger = logging.getLogger(__name__)

class SSHSession:
    def __init__(self, server: Server, force_root: bool = False):
        self.server = server
        self.conn: Optional[asyncssh.SSHClientConnection] = None
        self.process: Optional[asyncssh.SSHClientProcess] = None
        self.force_root = force_root
        self._stop_event = asyncio.Event()

    async def connect(self):
        """Establish SSH connection."""
        if self.conn:
            # Check if connection is still active
            try:
                # A simple way to check is to send a dummy keepalive or just check the transport state
                # asyncssh doesn't have a simple is_connected property that covers all cases,
                # but checking if the transport is active is a good start.
                if not self.conn._transport.is_closing():
                    print(f"Connection already established to {self.server.hostname}...")
                    return
            except Exception:
                pass

            # If we get here, the connection is dead
            print(f"Previous connection to {self.server.hostname} appears dead, reconnecting...")
            self.conn = None

        username = "root" if self.force_root else self.server.username

        try:
            print(f"Connecting to {self.server.hostname}...")
            self.conn = await asyncssh.connect(
                self.server.hostname,
                port=self.server.ssh_port,
                username=username,
                client_keys=[self.server.key_path] if self.server.key_path else None,
                known_hosts=None,  # For simplicity in this prototype, ignore known_hosts
                keepalive_interval=30,  # Send keepalive every 30 seconds
                keepalive_count_max=3,  # Disconnect after 3 missed keepalives
                pkcs11_provider=None,   # disable PKCS#11 completely
            )
            print(f"Connected to {self.server.hostname}...")
        except Exception as e:
            logger.error(f"Failed to connect to {self.server.hostname}: {e}")
            raise

    async def run_command(
        self,
        command: str,
        log_callback: Callable[[str], None],
        status_callback: Optional[Callable[[bool], None]] = None
    ):
        """
        Run a command and stream output to log_callback.
        This function runs until the command finishes or stop() is called.
        """
        if not self.conn:
            await self.connect()

        try:
            print(f"Running command: {command}")
            log_callback(f"--- Connecting to {self.server.hostname}... ---\n")
            log_callback(f"--- Running: {command} ---\n")

            # Start the process
            self.process = await self.conn.create_process(
                command,
                term_type='xterm',  # To potentially get color output if we parsed it
            )
            print(f"Process created for {self.server.hostname}...")

            if status_callback:
                status_callback(True)

            # Read stdout and stderr concurrently
            async def read_stream(stream, prefix=""):
                async for line in stream:
                    log_callback(line)
                    # print(f"Line: {line}")

            print(f"Waiting for process to finish...")
            await asyncio.gather(
                read_stream(self.process.stdout),
                read_stream(self.process.stderr),
                self.process.wait()
            )

            return_code = self.process.returncode
            print(f"Process exited with code {return_code}")
            log_callback(f"\n--- Process exited with code {return_code} ---\n")

        except asyncio.CancelledError:
            log_callback("\n--- Task cancelled ---\n")
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
        except Exception as e:
            log_callback(f"\n--- Error: {str(e)} ---\n")
        finally:
            if status_callback:
                status_callback(False)
            self.process = None

    async def close(self):
        """Close the SSH connection."""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception:
                pass

        if self.conn:
            self.conn.close()
            await self.conn.wait_closed()
            self.conn = None

    async def test_connection(self) -> bool:
        try:
            await self.connect()
            return True
        except Exception:
            return False
        finally:
            await self.close()

async def run_bash_script_via_ssh(
    server: Server,
    script_text: str,
    log_callback: Callable[[str], None],
    status_callback: Optional[Callable[[bool], None]] = None,
    force_root: bool = False
) -> None:
    """
    Run the given bash script on the remote server via SSH.

    The script is transferred as base64 over stdin and piped to bash.
    All stdout/stderr is passed to log_callback.
    """
    if not script_text:
        raise ValueError("script_text must not be empty")

    session = SSHSession(server, force_root=force_root)
    try:
        # Encode script as base64 to avoid quoting issues.
        script_b64 = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
        command = f"echo '{script_b64}' | base64 -d | bash"
        await session.run_command(command, log_callback, status_callback)
    finally:
        # Optionally close connection when done.
        await session.close()
