# Node Arrangement Diagram

```mermaid
graph TB
    subgraph "Machine 1 (Coordinator)"
        M1[Monitor<br/>:50000<br/>DHT Entry Point]
        C1[Client<br/>:51000<br/>Trainer Manager]
    end
    
    subgraph "Machine 2 (Worker A)"
        S1A[Server<br/>:50500<br/>head.0.0.0]
        S1B[Server<br/>:50501<br/>tail.0.0.0]
    end
    
    subgraph "Machine 3 (Worker B)"
        S2A[Server<br/>:50500<br/>head.0.1.0]
        S2B[Server<br/>:50501<br/>tail.0.1.0]
    end
    
    subgraph "Machine 4 (Worker C)"
        S3A[Server<br/>:50500<br/>head.0.2.0]
        S3B[Server<br/>:50501<br/>tail.0.2.0]
    end
    
    subgraph "Dynamic Trainers (spawned by Client)"
        T1[Trainer 1<br/>Pipeline: 0→1]
        T2[Trainer 2<br/>Pipeline: 0→1]
        T3[Trainer 3<br/>Pipeline: 0→1]
    end
    
    M1 -.DHT.-> C1
    M1 -.DHT.-> S1A
    M1 -.DHT.-> S1B
    M1 -.DHT.-> S2A
    M1 -.DHT.-> S2B
    M1 -.DHT.-> S3A
    M1 -.DHT.-> S3B
    
    C1 -->|spawns| T1
    C1 -->|spawns| T2
    C1 -->|spawns| T3
    
    T1 -->|uses| S1A
    T1 -->|uses| S1B
    T2 -->|uses| S2A
    T2 -->|uses| S2B
    T3 -->|uses| S3A
    T3 -->|uses| S3B
```

## Port Assignment Pattern

```
Monitor:  50000
Client:   51000
Servers:  50500 + <0 to num_servers>
          51500 + <0 to num_servers>

Examples:
- head.0.0.0 → 50500 and 51500 (stage 0, expert 0)
- tail.0.0.0  → 50501 and 51501 (stage 1, expert 0)
- head.0.1.0 → 50502 and 51502 (stage 0, expert 1)
- tail.0.1.0  → 50503 and 51503 (stage 1, expert 1)
# if they are started in the same process (num_servers = 4)
```

## Expert UID Naming Convention

```
{stage_name}.{version}.{expert_index}.{subversion}

Examples:
- head.0.0.0  (head stage, replica 0)
- head.0.1.0  (head stage, replica 1)
- tail.0.0.0   (tail stage, replica 0)
- tail.0.1.0   (tail stage, replica 1)
- head.0.0.0   (head stage for 3-stage models)
- body.0.0.0   (body stage for 3-stage models)
- tail.0.0.0   (tail stage for 3-stage models)
```

## Multi-Machine Deployment Example:

### Scenario: 5 Machines, 2-Stage Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Machine 1: 192.168.1.10 (Coordinator)                   │
│ ├─ Monitor (port 50000) - DHT entry point               │
│ └─ Client (port 51000) - spawns trainers                │
└─────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬──────────┬───────────────┐
        │                       │          │               │
┌───────▼─────────┐   ┌─────────▼─────┐  ┌─▼──────────┐  ┌─▼──────────┐
│ Machine 2:      │   │ Machine 3:    │  │ Machine 4: │  │ Machine 5: │
│ 192.168.1.11    │   │ 192.168.1.12  │  │ 10.0.0.5   │  │ 10.0.0.6   │
│ (Local Worker)  │   │ (Local Worker)│  │ (Remote)   │  │ (Remote)   │
├─────────────────┤   ├───────────────┤  ├────────────┤  ├────────────┤
│ head.0.0.0      │   │ head.0.1.0    │  │ tail.0.0.0 │  │ tail.0.1.0 │
│ port 50500      │   │ port 50500    │  │ port 50500 │  │ port 50500 │
│ GPU: cuda:0     │   │ GPU: cuda:0   │  │ GPU: cuda:0│  │ GPU: cuda:0│
└─────────────────┘   └───────────────┘  └────────────┘  └────────────┘
```

### Load Balancing Across Replicas:

```
Client Discovery Result:
├─ Stage 0 (head): [head.0.0.0, head.0.1.0]
└─ Stage 1 (tail):  [tail.0.0.0, tail.0.1.0]

Complete Pipelines (all possible paths):
1. head.0.0.0 → tail.0.0.0  (Trainer 1)
2. head.0.0.0 → tail.0.1.0  (Trainer 2)
3. head.0.1.0 → tail.0.0.0  (Trainer 3)
4. head.0.1.0 → tail.0.1.0  (Trainer 4)

Client spawns 4 trainers, one for each path
Each trainer uses a unique pipeline to maximize throughput
```

## Scaling Patterns:

### Horizontal Scaling (Add More Replicas):

```
Initial: 1 replica per stage
├─ head.0.0.0 → tail.0.0.0 (1 trainer)

Add Workers:
├─ head.0.0.0 → tail.0.0.0 (2 trainers)
└─ head.0.1.0 → tail.0.1.0 

Add Workers:
├─ head.0.0.0 → tail.0.0.0 (3 trainers)
├─ head.0.1.0 → tail.0.1.0
└─ head.0.2.0 → tail.0.2.0
```

### Vertical Scaling (Add More Stages):

```
2-Stage Pipeline:
Input → head.0.X.0 → tail.0.X.0 → Output

3-Stage Pipeline:
Input → head.0.X.0 → body.0.X.0 → tail.0.X.0 → Output

4-Stage Pipeline:
Input → head.0.X.0 → body1.0.X.0 → body2.0.X.0 → tail.0.X.0 → Output
```

