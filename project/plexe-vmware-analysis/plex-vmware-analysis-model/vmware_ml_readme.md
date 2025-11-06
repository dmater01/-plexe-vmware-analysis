# VMware ML Analyzer

A machine learning-powered tool for analyzing VMware infrastructure and identifying optimization opportunities using RVtools data.

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ML Framework](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

## 🎯 Overview

The VMware ML Analyzer uses real machine learning algorithms to analyze your VMware infrastructure and provide actionable optimization recommendations. By processing RVtools CSV exports, it identifies over-provisioned and under-provisioned virtual machines, calculates potential cost savings, and generates detailed reports for infrastructure teams.

### Key Features

- **🤖 Real ML Analysis**: Uses scikit-learn Random Forest algorithms for CPU utilization prediction
- **💰 Cost Optimization**: Identifies rightsizing opportunities with ROI calculations
- **📊 Detailed Reports**: Generates CSV reports and JSON analytics for further processing
- **🔧 Easy Integration**: Command-line interface for automation and CI/CD pipelines
- **⚡ Fast Processing**: Handles large RVtools exports efficiently
- **📈 Feature Engineering**: Automatically creates relevant metrics like Memory-per-CPU ratios

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- RVtools CSV export file

### Installation

1. Clone or download the repository:
```bash
git clone <repository-url>
cd vmware-ml-analyzer
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn
```

### Basic Usage

```bash
# Analyze your RVtools export
python vmware_ml_analyzer.py your_rvtools_export.csv

# Custom vCPU cost and output directory
python vmware_ml_analyzer.py data/vm_export.csv --vcpu-cost 200 --output-dir results/
```

## 📋 Requirements

### Python Packages
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### Input Data Format
The tool expects RVtools CSV exports with the following columns:
- **Required**: `VM`, `CPUs`
- **Optional**: `Memory`, `Powerstate`, `OS according to the configuration file`, storage columns

## 💻 Command Line Options

```bash
usage: vmware_ml_analyzer.py [-h] [--vcpu-cost VCPU_COST] [--output-dir OUTPUT_DIR] [--version] csv_file

positional arguments:
  csv_file              Path to the RVtools CSV export file

optional arguments:
  -h, --help            Show help message and exit
  --vcpu-cost VCPU_COST Annual cost per vCPU in USD (default: 150)
  --output-dir OUTPUT_DIR Output directory for results (default: current directory)
  --version             Show program's version number and exit
```

### Examples

```bash
# Basic analysis
python vmware_ml_analyzer.py sample.csv

# Custom vCPU pricing
python vmware_ml_analyzer.py vm_data.csv --vcpu-cost 180

# Specify output location
python vmware_ml_analyzer.py data/rvtools.csv --output-dir ./reports/

# Enterprise analysis with custom parameters
python vmware_ml_analyzer.py enterprise_vms.csv --vcpu-cost 250 --output-dir /shared/reports/
```

## 📊 Output Files

The analyzer generates two main output files:

### 1. `vmware_optimization_recommendations.csv`
Detailed VM-level recommendations including:
- VM name and current configuration
- Predicted CPU utilization
- Efficiency scores
- Memory allocation (if available)
- Power state information

### 2. `analysis_results.json`
Machine learning model metrics and metadata:
```json
{
  "model_type": "Random Forest CPU Utilization Predictor",
  "features_used": ["CPUs", "Memory_per_CPU", "Powerstate_encoded"],
  "feature_importance": {
    "CPUs": 0.654,
    "Memory_per_CPU": 0.231,
    "Powerstate_encoded": 0.115
  },
  "r2_score": 0.847,
  "rmse": 12.3,
  "training_samples": 700,
  "test_samples": 300
}
```

## 🔍 How It Works

### 1. Data Processing
- Loads and validates RVtools CSV data
- Cleans memory and storage columns (handles formats like "8,192 MB")
- Creates derived features like Memory-per-CPU ratios
- Encodes categorical variables (OS, power state)

### 2. Machine Learning Model
- Trains a Random Forest regressor to predict CPU utilization
- Uses features like CPU count, memory allocation, and VM characteristics
- Evaluates model performance with R² score and RMSE metrics
- Provides feature importance rankings

### 3. Optimization Analysis
- Calculates efficiency scores for each VM
- Identifies over-provisioned VMs (efficiency < 60%)
- Identifies under-provisioned VMs (efficiency > 120%)
- Generates rightsizing recommendations

### 4. Business Intelligence
- Calculates annual cost savings potential
- Provides ROI analysis for optimization efforts
- Generates actionable recommendations with specific vCPU adjustments

## 📈 Sample Output

```
🎯 VMWARE INFRASTRUCTURE ANALYSIS
Real ML implementation using scikit-learn
============================================================

🔧 PREPARING VMWARE DATA FOR ANALYSIS
==================================================
✅ Loaded sample.csv: 1000 rows, 15 columns
✅ Data processing complete: 1000 VMs ready for ML

🤖 BUILDING ML MODEL FOR CPU OPTIMIZATION
==================================================
📊 Training with 3 features on 1000 VMs
Features used: ['CPUs', 'Memory_per_CPU', 'Powerstate_encoded']
✅ Model trained successfully!
   R² Score: 0.847
   RMSE: 12.30%

📊 GENERATING OPTIMIZATION RECOMMENDATIONS
==================================================
🔍 ANALYSIS RESULTS:
   Total VMs: 1000
   Over-provisioned: 234
   Under-provisioned: 45
   Well-configured: 721

⬇️ RIGHTSIZING OPPORTUNITIES:
   • web-server-01           | 4 → 2 vCPU | Save $300/year
   • db-backup-vm            | 8 → 4 vCPU | Save $600/year
   • test-environment-03     | 6 → 3 vCPU | Save $450/year

💰 POTENTIAL ANNUAL SAVINGS: $127,500
💡 ROI: 6,375% return on analysis investment
```

## 🏗️ Architecture

### Class Structure
```python
class VMwareMLAnalyzer:
    def __init__(self, vcpu_cost=150, output_dir='.')
    def prepare_data(self, csv_file)          # Data cleaning and feature engineering
    def build_model(self, data)               # ML model training and evaluation
    def generate_recommendations(self, data)   # Business recommendations
```

### Key Components
- **Data Pipeline**: Robust CSV processing with error handling
- **Feature Engineering**: Automatic creation of relevant ML features
- **ML Engine**: scikit-learn Random Forest with proper validation
- **Business Logic**: Cost calculation and ROI analysis
- **Report Generation**: CSV and JSON output for downstream processing

## 🔧 Customization

### Adjusting Efficiency Thresholds
Modify the efficiency score thresholds in `generate_recommendations()`:
```python
over_provisioned = recommendations[recommendations['Efficiency_Score'] < 0.6]  # 60%
under_provisioned = recommendations[recommendations['Efficiency_Score'] > 1.2]  # 120%
```

### Adding New Features
Extend the feature engineering in `prepare_data()`:
```python
# Add custom features
processed_data['Storage_Ratio'] = processed_data['Provisioned_MB'] / processed_data['In_Use_MB']
processed_data['VM_Density'] = processed_data['CPUs'] / processed_data['Host_CPU_Count']
```

### Model Tuning
Adjust Random Forest parameters in `build_model()`:
```python
self.model = RandomForestRegressor(
    n_estimators=200,    # More trees
    max_depth=10,        # Limit depth
    random_state=42
)
```

## 🚀 Integration Examples

### Automated Reporting
```bash
#!/bin/bash
# Weekly VM optimization report
python vmware_ml_analyzer.py /data/weekly_rvtools.csv \
    --output-dir /reports/$(date +%Y-%m-%d) \
    --vcpu-cost 175

# Email results to infrastructure team
mail -s "Weekly VM Optimization Report" infrastructure@company.com < /reports/$(date +%Y-%m-%d)/analysis_results.json
```

### CI/CD Pipeline
```yaml
# .github/workflows/vm-analysis.yml
name: VM Infrastructure Analysis
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install pandas numpy scikit-learn
      - name: Run analysis
        run: python vmware_ml_analyzer.py data/rvtools_export.csv
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: vm-analysis-results
          path: "*.csv"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RVtools** by Rob de Veij for excellent VMware data collection
- **scikit-learn** team for robust ML libraries
- VMware infrastructure teams worldwide for inspiration

## 📞 Support

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/wiki)

---

**Built with ❤️ for VMware infrastructure optimization**