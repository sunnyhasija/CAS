# SCM-Arena: The World's First LLM Supply Chain Benchmark

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

**SCM-Arena** is the first standardized benchmark platform for evaluating Large Language Models on supply chain management coordination tasks. Using the classic Beer Game simulation, we provide rigorous, reproducible evaluation of LLM decision-making capabilities in multi-agent supply chain environments.

## 🎯 Quick Start

### Run Your First Benchmark (2 minutes)
```bash
# Install
git clone https://github.com/sunnyhasija/CAS
cd CAS
poetry install

# Start Ollama & pull model
ollama serve
ollama pull llama3.2

# Test
poetry run python -m scm_arena.cli test-model --model llama3.2

# Run benchmark
poetry run python -m scm_arena.cli run --model llama3.2 --rounds 10 --save-database
```

### Full Research Study (12-15 hours)
```bash
# Complete 1,080 experiment evaluation with data capture
poetry run python -m scm_arena.cli experiment \
  --models llama3.2 \
  --memory none short full \
  --prompts specific neutral \
  --visibility local adjacent full \
  --scenarios classic random shock \
  --game-modes modern classic \
  --runs 10 --rounds 30 \
  --save-results study_results.csv \
  --save-database
```

## 🔬 Key Features

### **Multi-Dimensional Evaluation**
- **6 Experimental Factors**: Models × Memory × Prompts × Visibility × Scenarios × Game Modes
- **Standardized Metrics**: Cost optimization, service level, coordination quality, consistency
- **Statistical Rigor**: n=10 replications, confidence intervals, effect size calculations

### **Complete Data Capture**
- **Every LLM interaction logged**: Prompts sent, responses received, decisions made
- **Canonical settings tracked**: Temperature, top_p, top_k, repeat_penalty
- **SQLite database**: 130k+ agent interactions per full study
- **Audit trail**: Complete reproducibility for research verification
- **Cost tracking**: Ready for expensive hosted model evaluation

### **Academic Rigor**
- **Literature-compliant costs**: $1 holding, $2 stockout (Sterman 1989)
- **Validated scenarios**: Classic step-change, random demand, shock patterns
- **Canonical LLM settings**: temperature=0.3, top_p=0.9 for reproducibility
- **Reproducible protocols**: Standardized evaluation methodology

## 🎮 The Beer Game Environment

SCM-Arena uses the classic Beer Game - a multi-agent supply chain simulation where four players (Retailer → Wholesaler → Distributor → Manufacturer) must coordinate inventory decisions without direct communication.

### **Why the Beer Game?**
- **Academic standard**: Used in business schools worldwide since 1960s
- **Coordination challenge**: Tests multi-agent decision making under uncertainty
- **Real-world relevance**: Models actual supply chain dynamics
- **Established benchmarks**: Decades of human performance data for comparison

### **Game Variants**
- **Classic Mode**: 2-turn information delays (original 1960s version)
- **Modern Mode**: Instant information sharing (Industry 4.0)
- **Multiple scenarios**: Step-change, random, shock demand patterns

## 📊 Benchmark Methodology

### **Canonical LLM Settings**
```
SCM-Arena Benchmark Standards:
├── Temperature: 0.3 (balanced decision-making)
├── Top_P: 0.9 (nucleus sampling)
├── Top_K: 40 (exploration window)
└── Repeat_Penalty: 1.1 (anti-repetition)
```

### **Information Architecture Testing**
```
Memory Strategies:
├── None: Pure reactive (0 decision history)
├── Short: Recent context (5 decisions)  
└── Full: Complete history (all decisions)

Visibility Levels:
├── Local: Own state only
├── Adjacent: See immediate partners + history
└── Full: Complete supply chain transparency

Prompt Types:
├── Specific: Position-based business prompts
└── Neutral: Generic supply chain prompts
```

### **Performance Metrics**
- **Total Supply Chain Cost** (40% weight) - Primary optimization target
- **Service Level** (30% weight) - Customer satisfaction measure
- **Bullwhip Ratio** (20% weight) - Coordination quality indicator  
- **Consistency** (10% weight) - Reliability across runs

## 🔧 Installation & Setup

### **Prerequisites**
- Python 3.9+
- [Poetry](https://python-poetry.org/) (dependency management)
- [Ollama](https://ollama.ai/) (local LLM hosting)

### **Quick Installation**
```bash
# 1. Clone repository
git clone https://github.com/sunnyhasija/CAS
cd CAS

# 2. Install dependencies  
poetry install

# 3. Start Ollama server (separate terminal)
ollama serve

# 4. Pull LLM models
ollama pull llama3.2
ollama pull llama2  
ollama pull mistral

# 5. Verify installation
poetry run python -m scm_arena.cli test-model --model llama3.2
```

### **Database Exploration**
```bash
# Explore captured experimental data
sqlite3 scm_arena_experiments.db

# View all tables
.tables

# Sample analysis queries
SELECT model_name, AVG(total_cost), AVG(service_level) 
FROM experiments 
GROUP BY model_name;

SELECT position, AVG(decision) 
FROM agent_rounds 
WHERE visibility_level = 'full'
GROUP BY position;
```

## 📈 Usage Examples

### **Single Model Evaluation**
```bash
# Quick evaluation (5 minutes)
poetry run python -m scm_arena.cli run \
  --model llama3.2 \
  --scenario classic \
  --rounds 15

# Comprehensive evaluation (2 hours)  
poetry run python -m scm_arena.cli experiment \
  --models llama3.2 \
  --memory none short full \
  --visibility local adjacent full \
  --runs 5 --rounds 20
```

### **Cross-Model Comparison**
```bash
# Compare multiple models
poetry run python -m scm_arena.cli experiment \
  --models llama3.2 llama2 mistral \
  --memory short \
  --visibility local \
  --scenarios classic random \
  --runs 3 --rounds 20 \
  --save-results model_comparison.csv
```

### **Information Architecture Study**
```bash
# Test visibility levels
poetry run python -m scm_arena.cli visibility-study \
  --model llama3.2 \
  --scenario classic \
  --runs 10

# Memory strategy analysis
poetry run python -m scm_arena.cli experiment \
  --models llama3.2 \
  --memory none short medium full \
  --visibility local \
  --runs 5
```

## 🔬 Research Applications

### **Academic Research**
- **Model Comparison**: Systematic evaluation of different LLM architectures
- **Prompt Engineering**: Testing position-specific vs. neutral prompts
- **Information Design**: Optimal memory and visibility configurations
- **Coordination Analysis**: Multi-agent decision making patterns

### **Industry Applications**  
- **Model Selection**: Choose best LLM for supply chain automation
- **Cost Estimation**: Predict hosted model costs before deployment
- **Performance Validation**: Verify LLM capabilities before production use
- **Benchmarking**: Compare custom models against established baselines

## 📊 Data & Results

### **Canonical Benchmark Results**
```
Model: llama3.2 (canonical settings: temp=0.3, top_p=0.9)
Conditions: 108 unique experimental conditions
Replications: 10 per condition (1,080 total experiments)
Database: 129,600 individual agent decisions captured

Performance Baseline:
├── Modern Mode: ~$1,600 avg cost | ~95% service level
├── Classic Mode: ~$1,700 avg cost | ~93% service level  
└── Bullwhip Ratio: 1.2-1.5 (approaching human performance)

Key Insights:
├── Modern information flow enables better coordination
├── Memory strategies show position-dependent effects
├── Visibility improvements provide measurable benefits
└── Canonical settings ensure reproducible results
```

### **Research-Ready Data**
- **Complete audit trail**: Every prompt, response, and decision logged
- **Reproducible results**: Full experimental metadata captured
- **Statistical power**: n=10 enables robust significance testing
- **Publication ready**: Comprehensive evaluation methodology

## 🤝 Contributing

### **For Researchers**
- **Add new models**: Extend the evaluation to your LLM
- **New scenarios**: Contribute additional supply chain environments
- **Analysis tools**: Develop visualization and statistical analysis components

### **For Developers**
- **Model integrations**: Add support for hosted APIs (GPT-4, Claude, etc.)
- **Performance optimizations**: Improve evaluation speed and efficiency
- **UI/Web interface**: Build web-based evaluation and visualization tools

### **Getting Started**
```bash
# Fork the repository
git fork https://github.com/sunnyhasija/CAS

# Create feature branch
git checkout -b feature/your-contribution

# Make changes and test
poetry run pytest
poetry run python -m scm_arena.cli test-model --model your_model

# Submit pull request
git push origin feature/your-contribution
```

## 📖 Citation

If you use SCM-Arena in your research, please cite:

```bibtex
@software{scm_arena_2025,
  title = {SCM-Arena: A Benchmark Platform for Large Language Model Supply Chain Coordination},
  author = {Abhinav "Sunny" Hasija and Vince Castillo},
  year = {2025},
  url = {https://github.com/sunnyhasija/CAS},
  note = {Version 1.0}
}
```



## 🔗 Links

- **Repository**: [https://github.com/sunnyhasija/CAS](https://github.com/sunnyhasija/CAS)
- **Paper**: [Coming Soon]
- **Dataset**: [Research Data Repository]
- **Web Platform**: [SCM-Arena.com] (Coming Soon)
- **Discord**: [Research Community] (Coming Soon)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive free software license that allows users to:
- Use the software for any purpose (commercial or non-commercial)
- Modify the software
- Distribute the software
- Include the software in proprietary works

The only requirements are to include the original copyright notice and license text in any copies or substantial portions of the software. Learn more about the [MIT License](https://opensource.org/licenses/MIT).

## 🙏 Acknowledgments

- **Beer Game Origins**: MIT Sloan School of Management (1960s)
- **Academic Foundation**: John Sterman's systems dynamics research
- **Open Source Community**: Ollama, Python, and Poetry ecosystems
- **Research Community**: Supply chain and AI/ML researchers worldwide

---

**SCM-Arena: Advancing the science of AI coordination in supply chain management**

*Built for researchers, by researchers. Open source, reproducible, and ready for the future of supply chain AI.*