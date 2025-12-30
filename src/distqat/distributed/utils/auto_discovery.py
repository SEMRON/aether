from typing import Tuple, Dict, List, Optional

from hivemind.utils.logging import get_logger
from hivemind.dht import DHT

from distqat.distributed.server.dht_handler import get_experts, get_expert_batch_size, get_expert_inner_steps
from distqat.config import Config

logger = get_logger(__name__)


def get_expected_stages(config: Config) -> List[str]:
    """
    Get the expected stages from the config
    """
    expected_stages = []
    for stage in config.model_pipeline.pipeline:
        _, stage_name = stage.model_name.split(".")
        expected_stages.append(stage_name)
    return expected_stages

def get_stage_name(config: Config, stage_index: int) -> str:
    """
    Get the stage name from the config
    """
    return config.model_pipeline.pipeline[stage_index].model_name.split(".")[1]


def discover_experts(dht: DHT, config: Config) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Discover experts using the dht_handler.get_experts function

    Args:
        dht: DHT instance
        config: Config instance

    Returns:
        complete_pipelines: Dictionary of complete pipelines
        incomplete_pipelines: Dictionary of incomplete pipelines
    """
    # First, discover all stage names from the config
    expected_stages = get_expected_stages(config)
    
    logger.debug(f"Discovering experts with stages: {expected_stages}")
    
    # Generate UIDs in the same format as your current method
    max_expert_index = config.max_expert_index
    all_uids = []
    
    for stage_name in expected_stages:
        for expert_index in range(max_expert_index):
            uid = f"{stage_name}.0.{expert_index}.0"
            all_uids.append(uid)
    
    logger.debug(f"Querying {len(all_uids)} UIDs")
    
    # Use the dht_handler.get_experts function
    try:
        remote_experts = get_experts(dht, all_uids)
        logger.debug(f"dht_handler.get_experts returned {len(remote_experts)} results")
        
        # Process results similar to your current method
        available_experts = {}  # expert_index -> {stage_name: expert_info}
        
        for i, remote_expert in enumerate(remote_experts):
            if remote_expert is not None:
                uid = all_uids[i]
                stage_name, _, expert_index, _ = uid.split(".")
                expert_index = int(expert_index)
                
                logger.debug(f"Found expert {uid} at {remote_expert.endpoint}")
                
                # Retrieve batch_size_per_step for this expert
                batch_size = get_expert_batch_size(dht, uid, latest=True)
                inner_steps = get_expert_inner_steps(dht, uid, latest=True)
                if expert_index not in available_experts:
                    available_experts[expert_index] = {}
                
                available_experts[expert_index][stage_name] = {
                    'uid': uid,
                    'endpoint': remote_expert.endpoint,
                    'batch_size_per_step': batch_size,
                    'inner_steps': inner_steps
                }
        
        # Separate complete and incomplete pipelines
        complete_pipelines = {}
        incomplete_pipelines = {}
        
        for expert_index, stages in available_experts.items():
            if all(stage in stages for stage in expected_stages):
                complete_pipelines[expert_index] = stages
                logger.debug(f"Complete pipeline found for expert_index {expert_index}: {list(stages.keys())}")
            else:
                missing_stages = [stage for stage in expected_stages if stage not in stages]
                incomplete_pipelines[expert_index] = {
                    'stages': stages,
                    'missing_stages': missing_stages
                }
                logger.debug(f"Incomplete pipeline for expert_index {expert_index}: missing {missing_stages}")
        
        logger.debug(f"Found {len(complete_pipelines)} complete and {len(incomplete_pipelines)} incomplete pipelines")
        
        return complete_pipelines, incomplete_pipelines
        
    except Exception as e:
        logger.error(f"Error using dht_handler.get_experts: {e}")
        import traceback
        traceback.print_exc()
        return ({}, {})


def discover_pipeline_gaps(dht: DHT, cfg: Config) -> Tuple[Optional[int], Optional[int]]:
    """
    Discover incomplete pipelines and return the next stage to create.
    
    Returns:
        (expert_index, stage_index) for the next stage to create, or (None, None) if no gaps found
    """
    complete_pipelines, incomplete_pipelines = discover_experts(dht, cfg)
    expected_stages = get_expected_stages(cfg)

    # Find the first incomplete pipeline and return the next stage to create
    if incomplete_pipelines:
        # Sort by expert_index to get consistent ordering
        for expert_index in sorted(incomplete_pipelines.keys()):
            missing_stages = incomplete_pipelines[expert_index]['missing_stages']
            # Find the first missing stage
            for stage_name in expected_stages:
                if stage_name in missing_stages:
                    stage_index = expected_stages.index(stage_name)
                    logger.debug(f"Will create stage {stage_index} ({stage_name}) for expert {expert_index}")
                    return expert_index, stage_index
    
    logger.debug("No incomplete pipelines found")
    return None, None


def find_next_expert_index(cfg: Config, dht: DHT) -> int:
    """
    Find the next available expert index for a new pipeline.
    """
    # Get all expected stages from config
    expected_stages = []
    for stage in cfg.model_pipeline.pipeline:
        _, stage_name = stage.model_name.split(".")
        expected_stages.append(stage_name)
    
    # Check existing experts to find the highest expert_index
    max_expert_index = -1
    max_check = 50  # Check up to expert index 50
    
    for expert_index in range(max_check):
        # Check if any stage exists for this expert_index
        has_any_stage = False
        for stage_name in expected_stages:
            uid = f"{stage_name}.0.{expert_index}.0"
            response = dht.get(uid, latest=True)
            logger.debug(f"Checking expert {expert_index} for stage {stage_name}: {response}")
            if response is not None:
                logger.debug(f"Expert {expert_index} for stage {stage_name} exists: {response}")
                has_any_stage = True
                break
        
        logger.debug(f"Has any stage: {has_any_stage}")
        if has_any_stage:
            logger.debug(f"Max expert index: {expert_index}")
            max_expert_index = expert_index
        else:
            # No stages found for this expert_index, we can use it
            break
    
 
    next_expert_index = max_expert_index + 1
    logger.debug(f"Next available expert index: {next_expert_index}")
    return next_expert_index


def get_random_expert_index(dht: DHT, cfg: Config, max_attempts: int = 100) -> int:
    """
    Generate a random expert index that does not yet exist in the DHT.
    Returns the first available random index, or raises RuntimeError if none found in max_attempts.
    """
    import random

    # Get all expected stages from config
    stage_names = get_expected_stages(cfg)

    attempted_indices = set()
    num_attempts = 0

    while num_attempts < max_attempts:
        candidate_index = random.randint(0, cfg.max_expert_index - 1)
        if candidate_index in attempted_indices:
            continue
        attempted_indices.add(candidate_index)
        num_attempts += 1

        # Check if any stage exists for this expert index
        exists = False
        for stage_name in stage_names:
            uid = f"{stage_name}.0.{candidate_index}.0"
            response = dht.get(uid, latest=True)
            if response is not None:
                exists = True
                break

        if not exists:
            return candidate_index

    raise RuntimeError(f"Failed to find a random non-existing expert_index in {max_attempts} attempts")


def generate_expert_and_stage_idx(dht: DHT, cfg: Config) -> Tuple[int, int]:
    """
    Generate an expert index and stage index for a new server to join the swarm.
    """
    expert_index, stage_index = discover_pipeline_gaps(dht, cfg)
    if expert_index is None or stage_index is None:
        expert_index = get_random_expert_index(dht, cfg, max_attempts=100)
        stage_index = 0
    logger.debug(f"Generated expert index: {expert_index} and stage index: {stage_index}")
    return expert_index, stage_index
    