# SCM-Arena 🏭

**Supply Chain Management LLM Benchmark Platform**

SCM-Arena is the first standardized benchmark platform for evaluating Large Language Models on supply chain management coordination tasks. Using the classic Beer Game simulation, it tests LLMs' ability to coordinate effectively in multi-agent supply chain scenarios.

## 🎯 What is the Beer Game?

The Beer Game is a classic supply chain coordination simulation used in operations research for 50+ years. Players manage a 4-tier supply chain (Retailer → Wholesaler → Distributor → Manufacturer) and must coordinate to minimize total system cost while dealing with:

- **Information delays** (2-week lag)
- **Shipping delays** (2-week lag) 
- **Demand uncertainty**
- **Bullwhip effect** (demand amplification)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) for dependency management
- [Ollama](https://ollama.ai/) for running LLMs locally

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd scm-arena

# Install dependencies
poetry install

# Start Ollama server (in separate terminal)
ollama serve

# Pull a model if needed
ollama pull llama3.2
```

### Run Your First Game

```bash
# Run a Beer Game with Ollama agents
poetry run python -m scm_arena.cli run

# Test a specific model
poetry run python -m scm_arena.cli test-model --model llama3.2

# Compare different agent types
poetry run python -m scm_arena.cli benchmark

# Run the detailed example
poetry run python examples/basic_game_example.py
```

## 📊 Available Commands

| Command | Description |
|---------|-------------|
| `run` | Run single game with Ollama agents |
| `compare` | Compare multiple models across scenarios |
| `test-model` | Quick test of model with sample scenario |
| `list-models` | Show available Ollama models |
| `benchmark` | Compare different agent types |

## 🎮 Example Usage

```python
from scm_arena import BeerGame, OllamaAgent, Position
from scm_arena.evaluation.scenarios import DEMAND_PATTERNS

# Create Ollama agents
agents = {
    Position.RETAILER: OllamaAgent(Position.RETAILER, "llama3.2"),
    Position.WHOLESALER: OllamaAgent(Position.WHOLESALER, "llama3.2"),
    Position.DISTRIBUTOR: OllamaAgent(Position.DISTRIBUTOR, "llama3.2"),
    Position.MANUFACTURER: OllamaAgent(Position.MANUFACTURER, "llama3.2"),
}

# Set up game with classic demand pattern
demand_pattern = DEMAND_PATTERNS["classic"]
game = BeerGame(agents, demand_pattern)

# Run simulation
while not game.is_complete():
    game.step()

# Analyze results
results = game.get_results()
print(f"Total Cost: ${results.total_cost:.2f}")
print(f"Bullwhip Ratio: {results.bullwhip_ratio:.2f}")
```

## 📈 Evaluation Scenarios

- **Classic**: Step change (4→8→4 units) - tests response to demand shifts
- **Random**: Stochastic demand - tests handling of uncertainty  
- **Shock**: Periodic demand spikes - tests recovery capability
- **Seasonal**: Cyclical patterns - tests pattern learning
- **Trend**: Gradual demand growth - tests adaptation
- **Complex**: Combined patterns - tests real-world scenarios

## 🤖 Agent Types

- **SimpleAgent**: Basic inventory management heuristics
- **OptimalAgent**: Advanced forecasting and safety stock
- **RandomAgent**: Random decisions (baseline)
- **OllamaAgent**: LLM-powered decision making
- **HumanAgent**: Interactive human player

## 📊 Key Metrics

- **Total Cost**: Primary performance metric
- **Bullwhip Ratio**: Demand amplification (coordination quality)
- **Service Level**: Orders fulfilled successfully
- **Individual Costs**: Per-player cost breakdown
- **Convergence Time**: Rounds to reach stability

## 🛠️ Development

### Project Structure

```
scm-arena/
├── src/scm_arena/
│   ├── beer_game/          # Core game engine
│   ├── models/             # LLM integrations  
│   ├── evaluation/         # Scenarios and metrics
│   └── cli.py             # Command-line interface
├── tests/                  # Test suite
├── examples/              # Usage examples
└── pyproject.toml         # Dependencies
```

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
poetry run black src/
poetry run isort src/
poetry run mypy src/
```

## 🎯 Roadmap

- [x] **Phase 1**: Core Beer Game + Ollama integration
- [ ] **Phase 2**: Web interface and leaderboard
- [ ] **Phase 3**: Additional LLM providers (OpenAI, Anthropic)
- [ ] **Phase 4**: Extended supply chain benchmarks
- [ ] **Phase 5**: Academic publication and community building

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📚 Citation

If you use SCM-Arena in your research, please cite:

```bibtex
@software{scm_arena_2024,
  title={SCM-Arena: A Benchmark Platform for Supply Chain Management LLMs},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/scm-arena}
}
```

---

**Happy benchmarking!** 🚀