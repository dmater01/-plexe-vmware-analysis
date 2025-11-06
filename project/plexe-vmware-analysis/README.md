# VMware ML Analyzer

Infrastructure cost optimization tool using machine learning to analyze VMware RVtools exports and identify VM rightsizing opportunities.

## Overview

VMware ML Analyzer uses scikit-learn's Random Forest regression model to analyze virtual machine configurations and provide data-driven recommendations for rightsizing VMs. By analyzing CPU, memory, and utilization patterns, the tool identifies over-provisioned and under-provisioned VMs, calculating potential annual cost savings.

## Features

- **ML-Powered Analysis**: Random Forest model trained on VM configuration patterns
- **Cost Optimization**: Identifies over-provisioned VMs and calculates annual savings
- **RVtools Integration**: Processes standard RVtools CSV exports
- **Flexible Configuration**: Customizable vCPU cost parameters for accurate ROI calculations
- **Comprehensive Reports**: VM-by-VM recommendations with efficiency scores
- **Multiple Output Formats**: CSV recommendations and JSON analysis metrics

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- RVtools CSV export

## Installation

```bash
# Clone the repository
git clone https://github.com/dmater01/-plexe-vmware-analysis.git
cd plexe-vmware-analysis

# Create and activate virtual environment
python -m venv plex-env
source plex-env/bin/activate  # On Windows: plex-env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn
```

## Quick Start

### Test Installation
```bash
python plex-vmware-analysis-model/test_installation.py
```

### Basic Analysis
```bash
# Using sample data
python plex-vmware-analysis-model/vmware_ml_analyzer.py Sample.csv

# Using CLI with custom vCPU cost
python plex-vmware-analysis-model/vmware_ml_cli.py Sample.csv --vcpu-cost 175 --output-dir ./results
```

### Using Your Own Data
```bash
# Export data from RVtools (vInfo tab as CSV)
python plex-vmware-analysis-model/vmware_ml_cli.py your_rvtools_export.csv --vcpu-cost 150
```

## Input Requirements

Your RVtools CSV export must include:
- **Required**: `VM`, `CPUs`
- **Optional** (enhances analysis): `Memory`, `Powerstate`, `OS according to the configuration file`, `Provisioned MiB`, `In Use MiB`, `Unshared MiB`

## Output Files

### vmware_optimization_recommendations.csv
VM-by-VM recommendations including:
- Current vs. recommended vCPU allocation
- Efficiency scores
- Annual savings potential
- Optimization category (over-provisioned, under-provisioned, well-configured)

### analysis_results.json
Model performance metrics:
- R² score and RMSE
- Feature importance rankings
- Training statistics

## vCPU Cost Calculation

Accurate vCPU costing is critical for credible ROI calculations. Use one of these methods:

### Method 1: Direct Infrastructure Cost
```
Annual vCPU Cost = (Total Infrastructure Costs) / (Total vCPUs)
```
Include: VMware licensing, hardware amortization, support, power/cooling, staff

### Method 2: Industry Benchmarks
- Small orgs (<500 VMs): $180-$280/vCPU/year
- Mid-market (500-2,500 VMs): $140-$220/vCPU/year
- Enterprise (2,500+ VMs): $100-$180/vCPU/year

### Method 3: Sensitivity Analysis
Run multiple scenarios to understand cost assumption impact:
```bash
python vmware_ml_cli.py data.csv --vcpu-cost 200  # Conservative
python vmware_ml_cli.py data.csv --vcpu-cost 150  # Moderate
python vmware_ml_cli.py data.csv --vcpu-cost 100  # Aggressive
```

See `plex-vmware-analysis-model/vcpu_cost_calculator.md` for detailed calculations.

## How It Works

### 1. Data Preprocessing
- Loads RVtools CSV export
- Handles memory format conversion ("8,192 MB" → numeric)
- Creates engineered features (Memory_per_CPU ratio)
- Encodes categorical variables (Powerstate, OS)

### 2. Model Training
- Random Forest Regressor (100 estimators)
- 70/30 train-test split
- StandardScaler for feature normalization
- Predicts CPU utilization patterns

### 3. Efficiency Analysis
```python
efficiency_score = predicted_cpu_util / (CPUs * 25)

# Classification:
# < 0.6 (60%)   = over-provisioned → downsize candidate
# > 1.2 (120%)  = under-provisioned → upsize candidate
# 0.6 - 1.2     = well-configured
```

### 4. ROI Calculation
```python
vcpu_reduction = current_vcpus - recommended_vcpus
annual_savings = vcpu_reduction * vcpu_cost
```

## Sample Datasets

- **Sample.csv**: Example RVtools export (~70 columns)
- **VXRail_sample.csv**: Larger VXRail environment for testing at scale

## Documentation

- **CLAUDE.md**: Development guide and architecture details
- **vmware_ml_readme.md**: Comprehensive usage guide
- **vcpu_cost_calculator.md**: Cost calculation formulas and benchmarks
- **Determining_the_accurate_vCPU_cost.md**: Practical cost determination guide

## Limitations

- **Synthetic Training Data**: Model trains on simulated CPU utilization (RVtools doesn't export actual usage)
- **Point-in-Time Analysis**: Based on snapshots, not historical trends
- **Cost Assumptions**: Savings accuracy depends on vCPU cost input
- **No Workload Context**: Doesn't account for application criticality

For production use, validate recommendations against actual performance metrics and business requirements.

## Use Cases

- **Capacity Planning**: Optimize VM sizing before hardware refresh
- **Cost Reduction**: Identify immediate cost-saving opportunities
- **Compliance**: Right-size VMs for licensing audits
- **Migration Planning**: Optimize configurations before cloud migration

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Specify your license here]

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This tool provides recommendations based on configuration patterns. Always validate recommendations against actual workload requirements and performance metrics before implementing changes in production environments.
