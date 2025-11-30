# Training Flow Diagram

```mermaid
sequenceDiagram
    participant T as Trainer
    participant S0 as Server (Stage 0)
    participant S1 as Server (Stage 1)
    
    Note over T,S1: Forward Pass
    T->>S0: send batch (input data)
    activate S0
    S0->>S0: compute forward
    S0->>S1: send activations (hidden)
    deactivate S0
    activate S1
    S1->>S1: compute forward + loss
    S1->>T: send output/loss
    deactivate S1
    
    Note over T,S1: Backward Pass
    T->>S1: send loss gradient
    activate S1
    S1->>S1: compute backward
    S1->>S0: send gradients (hidden)
    deactivate S1
    activate S0
    S0->>S0: compute backward
    S0->>S0: update local params
    S0->>T: send gradients (input)
    deactivate S0
    
    T->>T: update learning rate
    
    Note over T,S1: Repeat inner_steps times
    
    Note over S0,S1: DiLoCo Outer Synchronization <br> if there are more than 1 <br> servers stage
    S0->>S1: all-reduce parameters
    S1->>S0: averaged parameters
    S0->>S0: apply outer optimizer
    S1->>S1: apply outer optimizer
```

## Inner vs Outer Loop

```
For outer_step in range(outer_steps):
    ├─ For inner_step in range(inner_steps):
    │   ├─ Forward pass (Trainer → Server0 → Server1)
    │   ├─ Backward pass (Server1 → Server0 → Trainer)
    │   └─ Local parameter update (Adam/SGD)
    │
    └─ Global synchronization
        ├─ Average parameters across all replicas
        └─ Apply outer optimizer (SGD + momentum)
```
