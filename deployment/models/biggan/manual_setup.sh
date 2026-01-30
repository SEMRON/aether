# Cleanup commands
pkill -f distqat && clear

# Train only the head and body experts

# Trainer 0 on 2x A6000
    ## Server 1
    python start_servers.py \
    --config-path configs/gptneo_split_2.yaml \
    --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
    --wandb-project none \
    --expert-index 0 \
    --stage-index 0 \
    --diloco-min-matchmaking-time 1800.0 \
    --diloco-inner-steps 62 
    ## Server 2
    python start_servers.py \
    --config-path configs/gptneo_split_2.yaml \
    --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
    --wandb-project none \
    --expert-index 0 \
    --stage-index 1 \
    --diloco-min-matchmaking-time 1800.0 \
    --diloco-inner-steps 62 

# Trainer 1 on 2x A100
## Server 1
python start_servers.py \
   --config-path configs/gptneo_split_2.yaml \
   --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
   --wandb-project none \
   --expert-index 1 \
   --stage-index 0 \
   --diloco-min-matchmaking-time 1800.0 \
   --diloco-inner-steps 100 

## Server 2
python start_servers.py \
   --config-path configs/gptneo_split_2.yaml \
   --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
   --wandb-project none \
   --expert-index 1 \
   --stage-index 1 \
   --diloco-min-matchmaking-time 1800.0 \
   --diloco-inner-steps 100 


# Trainer 2 on H200
python start_servers.py \
   --config-path configs/gptneo_split_2.yaml \
   --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
   --wandb-project none \
   --num-servers 2 \
   --expert-index 2 \
   --expert-index 2 \
   --stage-index 0 \
   --stage-index 1 \
   --diloco-min-matchmaking-time 1800.0 \
   --diloco-inner-steps 140

# Trainer 3 on a A6000 and a RTX A6000
## Server 1
python start_servers.py \
   --config-path configs/gptneo_split_2.yaml \
   --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
   --wandb-project none \
   --expert-index 3 \
   --stage-index 0 \
   --diloco-min-matchmaking-time 1800.0 \
   --diloco-inner-steps 75 

## Server 2
python start_servers.py \
   --config-path configs/gptneo_split_2.yaml \
   --network-initial-peers "/ip4/95.217.225.248/tcp/50010/p2p/QmcisFomaTvqqGAZ7kdpyNWDsssT7jUKpM52JnRWNWKRed" \
   --wandb-project none \
   --expert-index 3 \
   --stage-index 1 \
   --diloco-min-matchmaking-time 1800.0 \
   --diloco-inner-steps 75 



