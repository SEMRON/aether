# Failover Mechanism Diagram

```mermaid
stateDiagram-v2
    [*] --> Healthy: Server starts
    Healthy --> No_Heartbeat: Server fails
    No_Heartbeat --> Incomplete_Pipeline: Client monitors pool
    Incomplete_Pipeline --> New_Server_Joins: 
    New_Server_Joins --> Fills_missing_stage: Auto-discovery
    Fills_missing_stage --> Registering: Load latest params
    Registering --> Healthy: Register in DHT
    
    note right of Healthy
        Sends Heartbeat to DHT
    end note
    
    note right of New_Server_Joins
        Scans DHT for missing
        stage indices
    end note
```

## Failure Timeline

```mermaid
gantt
    title Server Failover Timeline
    dateFormat ss
    axisFormat %S
    
    section Training
    Normal operation    :active, train1, 00, 20s
    Failure detected    :crit, fail1, 20, 5s
    Waiting for recovery :wait1, 25, 15s
    Resume training     :active, train2, 40, 20s
    
    section Server A (head.0.0.0)
    Running             :active, sa1, 00, 60s
    
    section Server B (tail.0.0.0)
    Running             :active, sb1, 00, 20s
    Failed              :crit, sb2, 20, 20s
    
    section Server C (tail.0.0.0)
    Joining DHT         :join, 25, 5s
    Loading parameters from peers  :load, 30, 5s
    Registering         :reg, 35, 5s
    Running             :active, sc1, 40, 20s
```

## Topology Changes

```mermaid
flowchart TB
    %% Timeline step labels
    label0([Before Failure ])
    label1([During Failure ])
    label2([After Recovery ])

    %% Before Failure (t=0)
    subgraph Before_Failure
        direction LR
        A1["Trainer 1"]
        B1["Server A<br/>head.0.0.0"]
        C1["Server B<br/>tail.0.0.0"]
        A1 --> B1
        B1 --> C1
    end

    %% During Failure (t=20)
    subgraph During_Failure
        direction LR
        A2["Trainer 1<br/>⚠️ Waiting"]
        B2["Server A<br/>head.0.0.0"]
        C2["Server B<br/>tail.0.0.0<br/>❌ FAILED"]
        A2 -.timeout.-> B2
        B2 -.failed.-> C2
    end

    %% After Recovery (t=40)
    subgraph After_Recovery
        direction LR
        A3["Trainer 1<br/>✓ Restarted"]
        B3["Server A<br/>head.0.0.0"]
        C3["Server C<br/>tail.0.0.0<br/>✓ NEW"]
        A3 --> B3
        B3 --> C3
    end

    %% Align the subgraphs vertically via invisible links and labels for strict ordering
    label0 -.-> Before_Failure
    Before_Failure -.-> label1
    label1 -.-> During_Failure
    During_Failure -.-> label2
    label2 -.-> After_Recovery

    %% Visual grouping by timeline
    classDef before fill:#ebf8ff,stroke:#3182ce,stroke-width:2px;
    classDef failure fill:#fff5f5,stroke:#e53e3e,stroke-width:2px,stroke-dasharray: 5 5;
    classDef recovery fill:#f0fff4,stroke:#38a169,stroke-width:2px;

    class Before_Failure before;
    class During_Failure failure;
    class After_Recovery recovery;

    %% Optional: clickable labels
    click label0 "#"
    click label1 "#"
    click label2 "#"
```

## Failover Process Steps

### 1. Failure Detection
```
Server fails and thus stops updating its heartbeat in the DHT
Client continuously monitors the DHT for running servers
Notices a server failed and a pipeline is now incomplete
```

### 2. Expert Removal
```
Client stops trainer for that pipeline
```

### 3. Auto-Discovery
```
New Server C starts
Connect to DHT via initial_peers
Query DHT for existing expert UIDs
Find gap: tail.0.0.0 missing
Assign self as tail.0.0.0
```

### 4. Checkpoint Loading
```
Query DHT for latest checkpoint metadata
Download model parameters for "tail" stage from peer, checkpoint or parameter mirror of the client
```

### 5. Registration & Resume
```
Register expert UID in DHT: tail.0.0.0
Client discovers new expert
Restart trainer 0 with new server
```


## Complex Failover Scenario: Multiple Server Failures with Pipeline Reorganization

This diagram demonstrates a more sophisticated failover scenario where multiple servers fail from different pipelines, causing all pipelines to break. The surviving servers then reorganize to form a new complete pipeline.

**Scenario:**
- Initial: 4 servers forming 2 complete pipelines with 2 trainers
- Failures: One head server and one tail server fail (from different pipelines)
- Recovery: The 2 surviving servers reorganize into 1 new pipeline

```mermaid
flowchart TB
    %% Timeline labels
    t0([Initial State: 2 Complete Pipelines])
    t1([After Failures: Both Pipelines Broken])
    t2([After Recovery: New Pipeline Formed])

    %% Initial State: 4 servers, 2 complete pipelines, 2 trainers
    subgraph Initial["Initial State (t=0)"]
        direction TB
        
        subgraph Pipeline1["Pipeline 1 ✓"]
            direction LR
            T1["Trainer 1"]
            S1["Server 1<br/>head.0.0.0<br/>✓ Running"]
            S2["Server 2<br/>tail.0.0.0<br/>✓ Running"]
            T1 --> S1
            S1 --> S2
        end
        
        subgraph Pipeline2["Pipeline 2 ✓"]
            direction LR
            T2["Trainer 2"]
            S3["Server 3<br/>head.0.1.0<br/>✓ Running"]
            S4["Server 4<br/>tail.0.1.0<br/>✓ Running"]
            T2 --> S3
            S3 --> S4
        end
    end

    %% During Failure: Server 1 (head.0.0.0) and Server 4 (tail.0.1.0) fail
    subgraph Failure["After Failures (t=20)"]
        direction TB
        
        subgraph Broken1["Pipeline 1 ✗ Broken"]
            direction LR
            T1F["Trainer 1<br/>⚠️ Stopped"]
            S1F["Server 1<br/>head.0.0.0<br/>❌ FAILED"]
            S2F["Server 2<br/>tail.0.0.0<br/>✓ Alive (orphaned)"]
            T1F -.X.-> S1F
            S1F -.X.-> S2F
        end
        
        subgraph Broken2["Pipeline 2 ✗ Broken"]
            direction LR
            T2F["Trainer 2<br/>⚠️ Stopped"]
            S3F["Server 3<br/>head.0.1.0<br/>✓ Alive (orphaned)"]
            S4F["Server 4<br/>tail.0.1.0<br/>❌ FAILED"]
            T2F --> S3F
            S3F -.X.-> S4F
        end
    end

    %% After Recovery: Server 3 (head) + Server 2 (tail) form new pipeline
    subgraph Recovery["After Recovery (t=40)"]
        direction TB
        
        subgraph NewPipeline["New Pipeline 3 ✓ Active"]
            direction LR
            T3["Trainer 3<br/>✓ New Trainer"]
            S3R["Server 3<br/>head.0.2.0<br/>✓ Alive"]
            S2R["Server 2<br/>tail.0.2.0<br/>✓ Alive"]
            T3 --> S3R
            S3R --> S2R
        end
        
        Note1["💡 Client detected:<br/>• head.0.1.0 available<br/>• tail.0.0.0 available<br/>→ Restarted those servers so that they get a new expert id and thus form a new complete pipeline<br/>→ Spawned Trainer 3"]
    end

    %% Timeline flow
    t0 -.-> Initial
    Initial -.-> t1
    t1 -.-> Failure
    Failure -.-> t2
    t2 -.-> Recovery

    %% Styling
    classDef initial fill:#ebf8ff,stroke:#3182ce,stroke-width:2px;
    classDef failure fill:#fff5f5,stroke:#e53e3e,stroke-width:2px,stroke-dasharray: 5 5;
    classDef recovery fill:#f0fff4,stroke:#38a169,stroke-width:2px;
    classDef note fill:#fefcbf,stroke:#d69e2e,stroke-width:2px;

    class Initial initial;
    class Failure failure;
    class Recovery recovery;
    class Note1 note;
```

### Detailed Walkthrough

#### Stage 1: Initial State (t=0)
**Available Experts:**
- `head.0.0.0` (Server 1)
- `tail.0.0.0` (Server 2)
- `head.0.1.0` (Server 3)
- `tail.0.1.0` (Server 4)

**Complete Pipelines Found:**
1. Pipeline 1: `head.0.0.0` → `tail.0.0.0` → Trainer 1 spawned
2. Pipeline 2: `head.0.1.0` → `tail.0.1.0` → Trainer 2 spawned

**Status:** 2 active trainers, both training successfully

#### Stage 2: Failures Occur (t=20)
**Failed Servers:**
- ❌ Server 1 (`head.0.0.0`) - network disconnection
- ❌ Server 4 (`tail.0.1.0`) - hardware failure

**Remaining Servers:**
- ✓ Server 2 (`tail.0.0.0`) - orphaned (no head stage)
- ✓ Server 3 (`head.0.1.0`) - orphaned (no tail stage)

**Client Actions:**
1. Detects incomplete pipeline for 0 (missing `head.0.0.0`)
2. Detects incomplete pipeline for 1 (missing `tail.0.1.0`)
3. Marks both experts as unavailable in DHT
4. Stops both trainers (no complete pipelines available)
5. Continues scanning DHT for new complete pipelines

**Status:** 0 active trainers, training paused, 2 idle servers

#### Stage 3: Automatic Recovery (t=40)
**Client Discovery Process:**
```
Scanning DHT for available experts...
Found: head.0.1.0 ✓
Found: tail.0.0.0 ✓
Missing: head.0.0.0 ✗
Missing: tail.0.1.0 ✗

Number of pipeline stages for full pipeline = 2
- head.0.1.0 + tail.0.0.0 = Can form a pipeline! ✓

Restart both servers
Through automatic discovery they form a new complete pipeline
Client detects new pipeline and starts a new trainer 3: head.0.2.0 → tail.0.2.0
```

**New Pipeline Formed:**
- Pipeline 3: `head.0.1.0` → `tail.0.0.0` → Trainer 3 spawned

**Status:** 1 active trainer, training resumed

### Key Insights

1. **Adaptive Resource Utilization**: Even though both original pipelines failed, the system recombines surviving servers from different pipelines.

2. **No Manual Intervention**: The entire recovery process is automatic:
   - Client continuously scans DHT
   - Detects if new pipelines can be formed
   - Spawns trainers for complete pipelines

3. **Graceful Degradation**: Training throughput reduced from 2 pipelines to 1, but training continues rather than failing completely.


### Timeline Summary

```
t=0s   : 4 servers, 2 pipelines, 2 trainers
         [T1: head.0.0.0 → tail.0.0.0]
         [T2: head.0.1.0 → tail.0.1.0]
         
t=20s  : 2 servers fail (Server 1, Server 4)
         Client detects failures
         Stops both trainers
         
t=25s  : Client scans DHT
         Discovers: head.0.1.0, tail.0.0.0
         Forms new complete pipeline
         
t=30s  : Spawns Trainer 3 for new pipeline
         [T3: head.0.2.0 → tail.0.2.0]
         
t=40s+ : Training continues with 1 pipeline
         50% throughput maintained
```

### Configuration Impact

This scenario demonstrates why these configuration parameters matter:

```yaml
# How often client scans for new experts
client:
  refresh_period: 5.0  # Lower = faster recovery detection
```
In the `dht_handler.py` the expiration time can be changed to control when the server heartbeat expires after failure. Will eventually become a config parameter as well.
