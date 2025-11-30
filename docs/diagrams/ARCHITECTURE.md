# Architecture Diagram 

```mermaid
graph TB
    subgraph DHT[DHT Network - Distributed Hash Table]
        direction LR
        PAD[" "]:::pad
        P2P[P2P Connections]
    end
    
    Monitor[Monitor<br/>- Initialize DHT<br/>- Collect metrics<br/>- WandB logging]
    Client[Client<br/>- Discover experts<br/>- Spawn trainers<br/>- Load balancing #40;to come#41;]
    DS[Data Server <br/>- Streams Data from where it's hosted<br/>- Pushes it into shared memory]
    
    subgraph Servers[Server Nodes]
        S1[Server 1<br/>Stage: head<br/>Expert: 0]
        S2[Server 2<br/>Stage: tail<br/>Expert: 0]
        S3[Server 3<br/>Stage: head<br/>Expert: 1]
        S4[Server 4<br/>Stage: tail<br/>Expert: 1]
    end
    
    subgraph Trainers[Trainer Processes]
        T1[Trainer 1<br/>Pipeline: 0]
        T2[Trainer 2<br/>Pipeline: 1]
    end
    
    DHT --> Monitor
    DHT --> Client
    DHT --> S1
    DHT --> S2
    DHT --> S3
    DHT --> S4
    DHT --> T1
    DHT --> T2
    
    Monitor -->|Writes peers| Client
    Client -->|Discovers| S1
    Client -->|Discovers| S2
    Client -->|Discovers| S3
    Client -->|Discovers| S4
    Client -->|Spawns| T1
    Client -->|Spawns| T2
    Client -->|Spawns| DS

    T1 -->|Pulls Data| DS
    T2 -->|Pulls Data| DS
    
    T1 -->|Forward| S1
    S1 -->|Forward| S2
    S2 -->|Backward| S1
    S1 -->|Backward| T1
    
    T2 -->|Forward| S3
    S3 -->|Forward| S4
    S4 -->|Backward| S3
    S3 -->|Backward| T2



    S1 <-->|Averages| S3
    S2 <-->|Averages| S4



    classDef pad fill:transparent,stroke:transparent,stroke-width:0,opacity:0;
```