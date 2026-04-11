# **RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin**

## Research Objective
Project Overview: My goal is to train a Reinforcement Learning (RL) agent to optimize land-use allocation within a specific region (e.g., a 25km buffer around the lake). The agent would analyze current land cover (at 500m resolution) and decide whether to "conserve" (forest) or "develop" (crops/built area). The objective is to balance economic development against environmental impact (e.g. Evapotranspiration (ET), water cost etc.) to maximize the total reward of the region.

Scope & Feasibility: I aim to deliver a Proof of Concept (POC) within this semester. I plan to leverage the data sources already used in the repo: Sentinel-2 for fine-grained land cover and MODIS (via Planetary Computer) for 8-day composite ET data. Since handling geospatial data can be complex, I intend to simplify the data processing pipeline to ensure I can deliver a complete, end-to-end modeling solution in the end.

## Data Avaliability
Total Area (Lake + 25km Buffer):59,434.97 km²

Total Pixels (Lake + Surroundings): ~594,349,743 (approx. 594 million pixels)
Surrounding Pixels Only: ~332,102,461 (approx. 332 million pixels)
Implication for RL:f
Process 594 million pixels directly as a flat state vector is computationally infeasible for standard RL agents. You will likely need to:

Downsample significantly (e.g., to 500m or 1km resolution) for the RL agent's high-level decision making.
Use patches/tiles (e.g., train on small 64×64 cutouts).
Aggregate features into administrative zones or sub-catchments instead of raw pixels.

1. Land-cover Data:
   - Source: Sentinel-2
   - Description: Raw: 10*10 High-resolution land cover data -> Processed: 10% sampling  in mode for certain year
   - Coverage: Global, including Lake Malawi and its surroundings.
   - Format: GeoTIFF files.
2. Evapotranspiration Data:
   - Source: MOD16A2 (MODIS Global Evapotranspiration Product)
   - Description: Raw: 500*500 8-day composite evapotranspiration data -> Processed: Yearly  averages per land cover type
    - Coverage: Global, including Lake Malawi and its surroundings.
    - Format: HDF files.
3. Economic Value Data:

    Based on recent agricultural reports (SeedCo Malawi, MwAPATA Institute, and ecosystem service valuations), here is a **lookup table** you can use for your simulation. ("ha": 100m*100m)

    | Land Cover Class | Estimated Economic Value (USD/ha/year) | Notes & Data Sources |
    | --- | --- | --- |
    | **Maize (Rainfed)** | **$300 - $500** | Yields are typically 1.5–2.5 tons/ha for smallholders. Market price is ~$200-$350/ton. |
    | **Tobacco** | **$500 - $800** | High revenue but high input cost. Net returns are often tighter (~$128/ha profit), but *Gross Margin* is higher. |
    | **Wetlands / Water** | **$554** | **CRITICAL VALUE.** Derived from "Gross Financial Value of Lake Chiuta wetlands" (fisheries + services). *Do not set this to zero!* |
    | **Forest** | **$200 - $450** | Based on ecosystem services (firewood, carbon, water regulation). Lower cash value than crops, but low water cost. |
    | **Rangeland** | **$50 - $100** | Low intensity grazing value. |
    | **Built Area (Urban)** | **$2,000+** | **Constraint:** Set this very high to prevent the agent from converting cities into farms. Real estate value far exceeds crop value. |

## Methodology
### Algorithm
PPO
### Training dataset prepartion
The total AOI is a 50*50 cells at 500m resolution (coverred 25km surrouding of the lake). The training will crop a 10*10 cells sub-patch as the input state for the agent to make decision in every episode. Among the cropped sub-patch, split 7:3 as training and test dataset. At each end of episode, randomly select one sample from test dataset to evaluate the performance.
### Environment Setup and Reward Design
- **State Space**: The state is a 10×10 cell patch. Each cell's state features $s(i, j)$ describe the area. The state space of cell can be represented by an N-dimensional vector, where N represents the modifiable number of land-use types (e.g. forest, built-area, crops etc.)
- **Action Space**: Each action selects a cell $(i, j)$ and performs a **source-to-target land-use transfer** — converting a fixed quantity of pixels from one modifiable class to another within that cell. The action is a 4-tuple:

  $a = (i, j, c_{src}, c_{tgt})$ where $i \in \{1,...,M\}$, $j \in \{1,...,N\}$, $c_{src}, c_{tgt} \in \mathcal{C}_{mod}$

  where $\mathcal{C}_{mod}$ is the set of modifiable land-use classes (Trees, Crops, Built Area, Bare Ground, Rangeland — excluding protected classes). Each action transfers a fixed number of pixels (e.g. 5 out of 25 per cell) from $c_{src}$ to $c_{tgt}$, guaranteeing **zero-sum conservation** within the cell by construction. When $c_{src} = c_{tgt}$, the action is a no-op (the agent chooses to skip). The transfer is clamped so that the source fraction cannot go below 0 and the target fraction cannot exceed 1.

  **Protected classes** (Water, Flooded, Snow/Ice, Clouds) are excluded from the observation and action spaces entirely — the agent can neither observe nor modify them. **Invalid actions** (e.g. transferring from a class with 0 pixels) are masked out using MaskablePPO's action masking mechanism.
- **Reward Function**: The reward at timestamp t+1 is calculated as following:
total_values_t+1 - total_values_t
where total_values_t = sum of proporation_k * (ECO_k + ET_k) for land-use type k at time t in cell (i. j) among all i and j. Extra components will be considered as project goes on, such as contiguity of land-use types, buffer zone requirements etc.
### Renerative Argriculture Experiment
- Consider the eco_value increasing by regnerative argriculture and compare the action behaviour and final result

## Generalization
- Infer other area of the study region (e.g. other 50*50 cells) and evaluate the generalization of the trained RL agent.