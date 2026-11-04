# Project Documentation

Index of project technical documentation for **libras-cv**. Use the links below to navigate the available documents and graph map topology.

## Documentation Index

| Document | Description | Reading |
|---|---|---|
| [**.digest.md**](./.digest.md) | Fast-path machine-readable orientation digest (stack, test commands, rules). | **Mandatory** |
| [**.graph.json**](./.graph.json) | Macro relation graph index for agent topology navigation and 1-hop routing. | **Mandatory** |
| [**ARCHITECTURE.md**](./adr/ARCHITECTURE.md) | Current implementation, planned components, and architecture constraints. | **Mandatory** |
| [**TESTS.md**](./adr/TESTS.md) | Test status, documented targets, and testing guidance. | **Mandatory** |
| [**landmark-svm.md**](./feature/landmark-svm.md) | Phase A landmark collection, SVM training, and live inference workflow. | Optional |

## Recommended Reading Order

If an exact path is supplied, read it directly. Otherwise use this order:

1. **.digest.md** — fast AI orientation (architecture pattern, stack, test commands).
2. **.graph.json** — macro relation graph index for 1-hop document lookup.
3. **ARCHITECTURE.md** and **TESTS.md** — baseline architecture and testing context.
4. Feature documents when their domain matches the task.
