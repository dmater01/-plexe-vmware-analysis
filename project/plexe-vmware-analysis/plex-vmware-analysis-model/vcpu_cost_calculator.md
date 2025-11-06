# vCPU Cost Calculator

## Comprehensive Cost Components

### 1. Software Licensing (40-60% of total cost)
| Component | Annual Cost Range | Notes |
|-----------|------------------|--------|
| VMware vSphere | $35-$85/vCPU | Depends on edition (Standard/Enterprise/Plus) |
| Windows Server | $15-$25/vCPU | Per-core licensing model |
| Backup Software | $5-$15/vCPU | Veeam, Commvault, etc. |
| Monitoring Tools | $2-$8/vCPU | vRealize, SolarWinds, etc. |

### 2. Hardware Costs (25-35% of total cost)
| Component | Annual Cost Range | Calculation Method |
|-----------|------------------|-------------------|
| Server Hardware | $30-$60/vCPU | 3-5 year amortization |
| Storage Systems | $20-$40/vCPU | SAN/NAS amortization |
| Network Equipment | $5-$15/vCPU | Switches, load balancers |

### 3. Operational Costs (15-25% of total cost)
| Component | Annual Cost Range | Notes |
|-----------|------------------|--------|
| Power & Cooling | $10-$25/vCPU | Data center utility costs |
| Support Contracts | $8-$20/vCPU | Vendor support (hardware/software) |
| Staff Allocation | $15-$35/vCPU | Admin time allocated to VM management |

## Industry Benchmarks by Organization Size

### Small Business (< 500 VMs)
```
Total vCPU Cost: $180-$280/year
- Higher per-unit costs due to limited economies of scale
- More reliance on vendor support
- Example: $220/vCPU for 200 VMs
```

### Mid-Market (500-2,500 VMs)
```
Total vCPU Cost: $140-$220/year
- Better licensing negotiations
- Shared infrastructure costs
- Example: $175/vCPU for 1,200 VMs
```

### Enterprise (2,500+ VMs)
```
Total vCPU Cost: $100-$180/year
- Volume licensing discounts
- Dedicated infrastructure teams
- Example: $135/vCPU for 8,000 VMs
```

### Cloud-First Organizations
```
Total vCPU Cost: $120-$200/year
- Hybrid cloud management overhead
- Cloud migration costs
- Example: $160/vCPU for mixed environment
```

## Quick Estimation Methods

### Method 1: Infrastructure Budget Approach
```bash
# If you know your total infrastructure budget
python vmware_ml_analyzer.py data.csv --vcpu-cost $((ANNUAL_BUDGET / TOTAL_VCPUS))
```

### Method 2: License-Based Estimation
```bash
# Conservative estimate (licensing + 50% overhead)
VCPU_COST=$((VMWARE_LICENSE_COST_PER_VCPU * 1.5))
python vmware_ml_analyzer.py data.csv --vcpu-cost $VCPU_COST
```

### Method 3: Industry Average
```bash
# Use industry averages by organization size
# Small: $230, Medium: $180, Large: $140
python vmware_ml_analyzer.py data.csv --vcpu-cost 180
```

## Cost Validation Checklist

✅ **Include ALL cost components:**
- [ ] VMware licensing (per-socket or per-CPU)
- [ ] Windows/Linux licensing
- [ ] Hardware amortization (3-5 years)
- [ ] Storage costs
- [ ] Network infrastructure
- [ ] Power and cooling
- [ ] Support contracts
- [ ] Staff time allocation

✅ **Verify calculations:**
- [ ] Check licensing model (per-socket vs per-core)
- [ ] Confirm vCPU to physical core ratios
- [ ] Include 3-year hardware refresh cycles
- [ ] Account for capacity planning overhead (20-30%)

✅ **Consider context:**
- [ ] Peak vs average utilization
- [ ] Development vs production environments
- [ ] Disaster recovery requirements
- [ ] Compliance and security requirements

## Real-World Examples

### Example 1: Financial Services Company
```
Environment: 3,000 VMs, 12,000 vCPUs
Annual Costs:
- VMware licensing: $720,000
- Hardware (amortized): $480,000
- Support: $240,000
- Operations: $360,000
Total: $1,800,000 ÷ 12,000 = $150/vCPU
```

### Example 2: Manufacturing Company
```
Environment: 800 VMs, 2,400 vCPUs
Annual Costs:
- VMware licensing: $180,000
- Hardware (amortized): $120,000
- Support: $60,000
- Operations: $90,000
Total: $450,000 ÷ 2,400 = $188/vCPU
```

### Example 3: Technology Startup
```
Environment: 200 VMs, 600 vCPUs
Annual Costs:
- VMware licensing: $45,000
- Cloud infrastructure: $36,000
- Support: $15,000
- Operations: $18,000
Total: $114,000 ÷ 600 = $190/vCPU
```

## Regional Cost Variations

### North America
- **Average**: $150-$220/vCPU
- **Factors**: Higher labor costs, premium support contracts

### Europe
- **Average**: $140-$200/vCPU  
- **Factors**: GDPR compliance overhead, energy costs

### Asia-Pacific
- **Average**: $120-$180/vCPU
- **Factors**: Lower operational costs, growing market competition

## Sensitivity Analysis

Test different cost scenarios:
```bash
# Conservative estimate
python vmware_ml_analyzer.py data.csv --vcpu-cost 200

# Moderate estimate  
python vmware_ml_analyzer.py data.csv --vcpu-cost 150

# Aggressive estimate
python vmware_ml_analyzer.py data.csv --vcpu-cost 100
```

This helps understand how cost assumptions affect ROI calculations and optimization priorities.