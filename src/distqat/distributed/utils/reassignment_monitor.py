import threading
import json
from typing import Optional, Dict

from hivemind.utils.logging import get_logger
from hivemind.utils import get_dht_time
from hivemind.dht import DHT

logger = get_logger(__name__)


def get_reassignment_key(expert_uid: str) -> str:
    """
    Get the DHT key for a reassignment signal for a specific expert.
    
    Args:
        expert_uid: Expert UID (e.g., "stage0.0.1.0")
        
    Returns:
        DHT key string
    """
    return f"reassign:{expert_uid}"


def get_reassignment_signal(dht: DHT, expert_uid: str) -> Optional[Dict]:
    """
    Check if there's a reassignment signal for the given expert UID.
    
    Args:
        dht: DHT instance
        expert_uid: Expert UID to check for reassignment
        
    Returns:
        Dictionary with 'new_expert_index' and 'new_stage_index' if found, None otherwise
    """
    key = get_reassignment_key(expert_uid)
    
    try:
        result = dht.get(key, latest=True)
        if result is None:
            return None
        
        value, expiration_time = result
        if expiration_time < get_dht_time():
            # Signal has expired
            return None
        
        reassignment = json.loads(value)
        return reassignment
        
    except Exception as e:
        logger.error(f"Failed to get reassignment signal from DHT: {e}")
        return None


def clear_reassignment_signal(dht: DHT, expert_uid: str) -> bool:
    """
    Clear a reassignment signal from the DHT (by storing with past expiration).
    
    Args:
        dht: DHT instance
        expert_uid: Expert UID to clear reassignment for
        
    Returns:
        True if successfully cleared
    """
    key = get_reassignment_key(expert_uid)
    try:
        # Store with past expiration to effectively delete it
        dht.store(key, "", expiration_time=get_dht_time() - 1)
        logger.info(f"Cleared reassignment signal for {expert_uid}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear reassignment signal: {e}")
        return False


def store_reassignment_signal(
    dht: DHT,
    expert_uid: str,
    expiration: float = 300
) -> bool:
    """
    Store a reassignment signal in the DHT.
    
    Args:
        dht: DHT instance
        expert_uid: Current expert UID (e.g., "stage0.0.1.0")
        expiration: Expiration time in seconds (default: 5 minutes)
        
    Returns:
        True if successfully stored, False otherwise
    """
    key = get_reassignment_key(expert_uid)
    value = {
        'expert_uid': expert_uid
    }
    expiration_time = get_dht_time() + expiration
    
    try:
        store_ok = dht.store(key, json.dumps(value), expiration_time=expiration_time)
        logger.info(
            f"Stored reassignment signal in DHT: {expert_uid} -> "
        )
        return store_ok
    except Exception as e:
        logger.error(f"Failed to store reassignment signal in DHT: {e}")
        return False


class ReassignmentMonitorThread(threading.Thread):
    """
    A background thread that periodically checks for reassignment signals in the DHT.
    
    When a reassignment signal is detected, it triggers server shutdown so it can be restarted and get a new expert uid.
    
    :param expert_uid: Expert UID for this server (e.g., "stage0.0.1.0")
    :param dht: DHT instance to check for reassignment signals
    :param check_period: How often (in seconds) to check for reassignment signals
    :param shutdown_callback: Callback function to trigger graceful shutdown
    """
    
    def __init__(
        self,
        expert_uid: str,
        dht: DHT,
        check_period: int = 30,
        shutdown_callback: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.expert_uid = expert_uid
        self.dht = dht
        self.check_period = check_period
        self.shutdown_callback = shutdown_callback
        self.stop_event = threading.Event()
        self.daemon = True
        
    def run(self) -> None:
        """
        Main thread loop that periodically checks for reassignment signals.
        """
        logger.info(f"ReassignmentMonitorThread started for expert {self.expert_uid}")
        
        while not self.stop_event.wait(self.check_period):
            try:
                reassignment = get_reassignment_signal(self.dht, self.expert_uid)
                
                if reassignment is not None:
                    logger.warning(
                        f"Reassignment signal detected for expert {self.expert_uid}: "
                    )

                    clear_reassignment_signal(self.dht, self.expert_uid)
                    
                    # Log that server should be restarted
                    logger.warning(
                        f"Reassignment detected! Server with expert {self.expert_uid} will be restarted "
                    )
                    
                    # Trigger shutdown callback if available
                    if self.shutdown_callback:
                        logger.info("Triggering graceful shutdown for reassignment...")
                        try:
                            self.shutdown_callback()
                        except Exception as e:
                            logger.error(f"Error in shutdown callback: {e}")
            except Exception as e:
                logger.error(f"Error checking for reassignment signal: {e}")
                
        logger.info(f"ReassignmentMonitorThread stopped for expert {self.expert_uid}")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.stop_event.set()