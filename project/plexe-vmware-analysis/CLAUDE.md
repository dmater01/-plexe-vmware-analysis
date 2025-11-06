# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VMware ML Analyzer - Infrastructure cost optimization tool using scikit-learn to analyze VMware RVtools exports and identify VM rightsizing opportunities

## Environment Setup

```bash
# Activate the virtual environment
source plex-env/bin/activate

# Install required dependencies
pip install pandas numpy scikit-learn

# Navigate to project directory
cd plex-vmware-analysis-model
```

## Running the Analyzer

```bash
# Test installation and dependencies
python test_installation.py

# Basic analysis (uses Sample.csv in current directory)
python vmware_ml_analyzer.py Sample.csv

# CLI version with custom vCPU cost
python vmware_ml_cli.py Sample.csv --vcpu-cost 175 --output-dir ./results

# Using VXRail sample dataset
python vmware_ml_cli.py VXRail_sample.csv --vcpu-cost 150
```

**Input Requirements:**
- RVtools CSV export with minimum columns: `VM`, `CPUs`
- Optional columns enhance analysis: `Memory`, `Powerstate`, `OS according to the configuration file`, `Provisioned MiB`, `In Use MiB`, `Unshared MiB`

**Output Files:**
- `vmware_optimization_recommendations.csv` - VM-by-VM rightsizing recommendations with efficiency scores
- `analysis_results.json` - ML model metrics, feature importance, and training statistics

## Architecture

### VMware ML Analyzer Architecture

**Class:** `VMwareMLAnalyzer`
- `prepare_data(csv_file)` - Loads and preprocesses RVtools CSV data
  - Handles memory format conversion ("8,192 MB" → numeric)
  - Creates engineered features (Memory_per_CPU)
  - Encodes categorical variables (Powerstate, OS)
  - Generates simulated CPU utilization targets (for training)

- `build_model(data)` - Trains Random Forest regression model
  - Features: CPUs, Memory_per_CPU, encoded categorical fields
  - 70/30 train-test split
  - Evaluates with R² score and RMSE
  - Returns feature importance rankings

- `generate_recommendations(data)` - Produces optimization recommendations
  - Calculates efficiency scores: predicted_util / (CPUs * 25)
  - Over-provisioned: efficiency < 0.6 (60%)
  - Under-provisioned: efficiency > 1.2 (120%)
  - Calculates annual savings: vCPU_reduction * vcpu_cost

**Key Implementation Details:**
- Uses scikit-learn's `RandomForestRegressor` with 100 estimators
- Applies `StandardScaler` for feature normalization
- Default vCPU annual cost: $150 (configurable via CLI)

## vCPU Cost Calculation

The analyzer requires a vCPU annual cost parameter for ROI calculations. Accurate costing is critical for credible recommendations.

### Cost Calculation Methods

**Method 1: Direct Infrastructure Cost (Most Accurate)**
```
Annual vCPU Cost = (Total Infrastructure Costs) / (Total vCPUs)

Components to include:
- VMware licensing (vSphere, vCenter)
- Hardware amortization (3-5 year refresh cycle)
- Support contracts (vendor + internal)
- Power and cooling
- Staff allocation (admin time)
```

**Method 2: Industry Benchmarks**
- Small orgs (<500 VMs): $180-$280/vCPU/year
- Mid-market (500-2,500 VMs): $140-$220/vCPU/year
- Enterprise (2,500+ VMs): $100-$180/vCPU/year

**Method 3: Conservative Estimation**
Start with licensing costs and multiply by 1.5-2.0x for overhead:
```bash
# Example: VMware licensing is $50/vCPU
python vmware_ml_cli.py data.csv --vcpu-cost 100  # 2x multiplier
```

### Sensitivity Analysis
Run multiple scenarios to understand how cost assumptions affect recommendations:
```bash
# Conservative (high cost)
python vmware_ml_cli.py data.csv --vcpu-cost 200

# Moderate (industry average)
python vmware_ml_cli.py data.csv --vcpu-cost 150

# Aggressive (optimized environment)
python vmware_ml_cli.py data.csv --vcpu-cost 100
```

See `vcpu_cost_calculator.md` and `Determining_the_accurate_vCPU_cost.md` for detailed cost breakdowns.

## Data Files

**Sample Datasets:**
- `Sample.csv` - Example RVtools export with ~70 columns (VM name, CPUs, Memory, Powerstate, OS, storage metrics, etc.)
- `VXRail_sample.csv` - Larger VXRail environment dataset for testing at scale

**Documentation:**
- `vmware_ml_readme.md` - Comprehensive usage guide with examples and architecture details
- `vcpu_cost_calculator.md` - Detailed vCPU cost calculation formulas and industry benchmarks
- `Determining_the_accurate_vCPU_cost.md` - Practical guide for determining organization-specific vCPU costs

## Testing

```bash
cd plex-vmware-analysis-model

# Validate environment and dependencies
python test_installation.py

# Run end-to-end analysis on sample data
python vmware_ml_analyzer.py Sample.csv

# Verify outputs were generated
ls -lh vmware_optimization_recommendations.csv analysis_results.json
```

The test_installation.py script validates:
- Python environment and version
- Required packages (pandas, numpy, scikit-learn, torch, tensorflow)
- File I/O operations
- Optional: OpenAI API connectivity (if configured)

## Important Implementation Details

### Data Processing Quirks
- **Memory parsing**: RVtools exports memory as "16,384 MB" strings - the analyzer handles comma removal and numeric conversion
- **CPU utilization simulation**: Since RVtools doesn't export actual utilization, the model creates synthetic targets using `base_util = 25 + CPUs * 8 + noise` for training
- **Zero CPU handling**: VMs with 0 CPUs are normalized to 1 to prevent division errors
- **Missing columns**: The analyzer gracefully handles missing optional columns (Memory, Powerstate, OS) by using defaults

### Model Training Process
1. **Feature engineering**: Creates `Memory_per_CPU` ratio and encodes categorical variables
2. **Train/test split**: 70/30 split with random_state=42 for reproducibility
3. **Scaling**: StandardScaler applied to all numeric features before training
4. **Model**: RandomForestRegressor with 100 estimators predicts CPU utilization
5. **Evaluation**: Reports R² score and RMSE on test set

### Efficiency Score Calculation
```python
efficiency_score = predicted_cpu_util / (CPUs * 25)

# Thresholds (configurable in generate_recommendations):
# < 0.6 (60%) = over-provisioned → candidate for downsizing
# > 1.2 (120%) = under-provisioned → candidate for upsizing
# 0.6 - 1.2 = well-configured
```

### ROI Calculation
```python
vcpu_reduction = current_vcpus - recommended_vcpus
annual_savings = vcpu_reduction * vcpu_cost
total_savings = sum(annual_savings for all over-provisioned VMs)
roi = (total_savings / analysis_cost) * 100
```

### Limitations and Considerations
- **Synthetic utilization**: Model trains on simulated CPU utilization, not actual historical data
- **Relative patterns only**: Predictions indicate relative efficiency, not absolute utilization values
- **Point-in-time analysis**: RVtools exports are snapshots; doesn't account for usage patterns over time
- **Cost assumptions**: Accuracy of savings calculations depends entirely on vCPU cost input
- **No workload awareness**: Model doesn't understand application criticality or performance requirements