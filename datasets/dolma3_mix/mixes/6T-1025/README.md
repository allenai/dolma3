# Data Mixing and Sampling

## Table of Contents

- [Overview](#overview)
- [Final Mix Composition](#final-mix-composition)
- [Mixing Methodology](#mixing-methodology)
  - [Swarm-Based Optimization](#swarm-based-optimization)
  - [Conditional Mixing](#conditional-mixing)
  - [Quality-Aware Upsampling](#quality-aware-upsampling)
- [Implementation Guide](#implementation-guide)

---

## Overview

Transforming the 9T pretraining pool into an effective training dataset requires determining optimal mixing ratios across data sources and quality levels. We use a swarm-based optimization approach that trains many small proxy models with different mixture ratios, then uses their performance to predict an optimal mix. This method significantly outperforms training on the natural data distribution.

Key innovations:
- **Swarm-based mixing**: Train 30M proxy models to explore the mixture space and fit per-task regression models
- **Conditional mixing**: Efficiently update mixes when data sources evolve without recomputing from scratch
- **Quality-aware upsampling**: Oversample high-quality data within each topic rather than flat filtering

---

## Final Mix Composition

Our 6T training mix achieves the following composition:

| Source | Type | 6T Mix Tokens | 6T Mix Docs | Percentage |
|--------|------|---------------|-------------|------------|
| Common Crawl | Web pages | 4.51T | 3.15B | 76.1% |
| olmOCR Science PDFs | Academic documents | 805B | 83.8M | 13.6% |
| StackEdu (Rebalanced) | GitHub code | 409B | 526M | 6.89% |
| FineMath 3+ | Math web pages | 152B | 95.5M | 2.56% |
| arXiv | Papers with LaTeX | 50.8B | 9.10M | 0.86% |
| Wikipedia & Wikibooks | Encyclopedic | 2.51B | 4.24M | 0.04% |
| **Total** | | **5.93T** | **3.87B** | **100%** |

---

## Mixing Methodology

### Swarm-Based Optimization

Our base mixing procedure operates in three stages:

#### 1. Swarm Construction
- Train 30M-parameter models (OLMo3 architecture) for 3B tokens each
- Sample mixing ratios from Dirichlet distributions centered on natural distribution
- Swarm size ≈ 5x number of domains
- Evaluate all models on Base Easy suite

#### 2. Per-Task Regression
- Fit separate generalized linear models for each evaluation task
- Maps mixture weights → task performance (bits per byte)
- Enables prediction of performance for unseen mixture candidates

#### 3. Constrained Optimization
- Minimize average task loss across all evaluation tasks
- Constraints:
  - 6T total token budget
  - Maximum 4-7× upsampling per domain (based on available tokens)
  - Domain-specific availability limits
- Solve via guided search from natural distribution

**Performance**: On 1B models, optimized DCLM mix improved average performance by 0.056 BPB with maximum gains of 0.209 BPB. Only 13/54 tasks degraded, with maximum degradation of 0.035 BPB.

### Conditional Mixing

When data sources evolve (new filters, additional domains, quality improvements), recomputing the full swarm is expensive. Conditional mixing solves this by:

1. Freezing the ratios of previously optimized domains into a single "virtual domain"
2. Running swarm optimization over: {virtual domain + new/changed domains}
3. This reduces dimensionality and computational cost significantly

**Application**: We performed three rounds:
1. **Round 1**: Optimize over 24 WebOrganizer categories in DCLM-Baseline web data
2. **Round 2**: Fix web:code ratio at 75:25, optimize over StackEdu programming languages
3. **Round 3**: Integrate PDF WebOrganizer categories, conditioned on prior rounds

This incremental approach was essential for incorporating late-arriving data sources (especially PDFs) without restarting optimization.

#### Learned WebOrganizer Ratios

| Category | Natural Distribution | Optimized Ratio | Change |
|----------|---------------------|-----------------|--------|
| Science, Math, and Technology | 5.00% | 21.03% | +16.03% |
| Software Development | 2.07% | 11.14% | +9.07% |
| Health | 7.26% | 9.94% | +2.68% |
| Entertainment | 7.33% | 9.60% | +2.27% |
| Games | 3.67% | 6.89% | +3.22% |
| Literature | 3.68% | 6.83% | +3.15% |
| Software | 2.62% | 4.73% | +2.11% |
| Education and Jobs | 4.77% | 4.29% | -0.48% |
| Finance and Business | 9.91% | 4.07% | -5.84% |
| Electronics and Hardware | 2.42% | 3.45% | +1.03% |
| Crime and Law | 2.86% | 2.92% | +0.06% |
| History and Geography | 1.90% | 2.71% | +0.81% |
| Politics | 7.61% | 2.17% | -5.44% |
| Religion | 3.16% | 1.98% | -1.18% |
| Industrial | 1.70% | 1.57% | -0.13% |
| Food and Dining | 3.13% | 1.41% | -1.72% |
| Sports and Fitness | 6.40% | 1.31% | -5.09% |
| Art and Design | 2.20% | 1.25% | -0.95% |
| Transportation | 2.62% | 0.98% | -1.64% |
| Home and Hobbies | 7.84% | 0.92% | -6.92% |
| Social Life | 3.98% | 0.46% | -3.52% |
| Travel and Tourism | 3.66% | 0.22% | -3.44% |
| Adult Content | 0.99% | 0.11% | -0.88% |
| Fashion and Beauty | 3.24% | 0.01% | -3.23% |

### Quality-Aware Upsampling

Rather than flat filtering (e.g., "keep top 25%"), we oversample high-quality data to better utilize token budgets.

#### Concept
- Each WebOrganizer topic gets an **upsampling curve**: quality percentile → upsampling factor
- Flat filtering = step function; our approach = monotonically increasing curve
- Typical pattern: discard bottom 40%, sample top 5% up to 7×

#### Curve Generation

For each topic, we solve for an upsampling curve satisfying:

1. **Target integral**: Determined by optimal topic ratio × 6T budget ÷ available tokens
2. **Max upsampling**: 7× cap to avoid excessive repetition
3. **Monotonicity**: Higher quality → higher upsampling

**Implementation**:
- Data organized into vigintiles (5-percentile buckets) by quality score
- For each bucket, integrate curve over its percentile range
- See `/mixing_dashboard.ipynb` for mathematical details and curve generation code

**Example**: If a topic needs 2.5x average upsampling:
- Bottom 40% (percentiles 0-40): 0x (discarded)
- Middle 50% (percentiles 40-90): 1-4x
- Top 10% (percentiles 90-100): 5-7x

---

## Implementation Guide

### Prerequisites

- **Proxy models**: 30M parameters, trained for 3B tokens (5x Chinchilla)
- **Evaluation suite**: Base Easy or custom task suite
- **Quality classifiers**: Applied to all web text and PDF documents
- **Topic classifiers**: FastText WebOrganizer classifier

### Workflow

1. **Partition data**:
   - Classify all documents by topic (WebOrganizer) and quality (vigintiles)
   - Store in format: `{source}/{topic}/{quality_bucket}/`

2. **Run swarm optimization**:
   - Sample ~5× number of domains for swarm size
   - Train all proxy models with different Dirichlet-sampled mixtures
   - Evaluate on development suite

3. **Fit regression models**:
   - One GLM per evaluation task
   - Input: mixture weights, Output: task performance

4. **Optimize mix**:
   - Minimize average task loss
   - Subject to token budget and upsampling constraints
   - See `/mixing_dashboard.ipynb` for optimization code

5. **Generate upsampling curves**:
   - For each topic: solve for curve meeting integral and max constraints
   - Apply curves to quality buckets
   - See `/mixing_dashboard.ipynb` for curve generation implementation


