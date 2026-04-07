# Risk Taxonomy

## Risk Categories

### 1. Implementation Difficulty (`implementation_difficulty`)
Technical complexity of implementing the proposed method.

| Severity | Criteria |
|----------|----------|
| LOW | Standard components, well-documented APIs, common patterns |
| MEDIUM | Custom layers or loss functions, moderate algorithmic complexity |
| HIGH | Novel algorithms, non-standard training procedures, complex data pipelines |
| CRITICAL | Requires specialized domain expertise, undocumented tricks, or extremely complex math |

### 2. Paper Clarity (`paper_clarity`)
How well the paper describes what needs to be implemented.

| Severity | Criteria |
|----------|----------|
| LOW | Clear descriptions with pseudocode or equations |
| MEDIUM | Some details missing but inferable from context |
| HIGH | Significant ambiguity, multiple valid interpretations |
| CRITICAL | Core method description is incomplete or contradictory |

### 3. Environment Dependency (`environment_dependency`)
External dependencies that may block or complicate reproduction.

| Severity | Criteria |
|----------|----------|
| LOW | Common packages, no special hardware |
| MEDIUM | Specific package versions, single GPU required |
| HIGH | Special hardware (multi-GPU, TPU), restricted datasets |
| CRITICAL | Proprietary software, unavailable datasets, extreme compute needs |

### 4. Data Availability (`data_availability`)
Access to required datasets and their usability.

| Severity | Criteria |
|----------|----------|
| LOW | Public datasets, well-documented formats |
| MEDIUM | Requires registration or agreement, minor preprocessing needed |
| HIGH | Restricted access, significant preprocessing, or large download |
| CRITICAL | Proprietary data, no public alternative, or data no longer available |

### 5. Computational Resource (`computational_resource`)
Compute requirements for training and evaluation.

| Severity | Criteria |
|----------|----------|
| LOW | Single CPU or modest GPU, trains in hours |
| MEDIUM | Single GPU, trains in 1-3 days |
| HIGH | Multi-GPU, trains in days to a week |
| CRITICAL | Multi-node, trains in weeks, requires 100+ GPU-hours |

## Gap Types

### missing
Information that should be in the paper but is completely absent.
Example: "The paper does not mention the learning rate used for fine-tuning."

### ambiguous
Information that is present but open to multiple interpretations.
Example: "The paper says 'standard augmentation' without specifying which transforms."

### contradictory
Information that conflicts between different parts of the paper.
Example: "Section 3 says batch size 32 but Table 2 footnote says batch size 64."

### omitted
Information that is intentionally left out (common for space reasons).
Example: "Hyperparameter details are said to be in supplementary material that is not available."

## Severity Levels

| Level | Score Weight | Action Required |
|-------|-------------|-----------------|
| LOW | 1 | Note and proceed with reasonable defaults |
| MEDIUM | 2 | Investigate alternatives, document assumption |
| HIGH | 3 | Requires explicit decision, may need experimentation |
| CRITICAL | 4 | Blocks progress, needs resolution before implementation |
