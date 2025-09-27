# Monte Carlo Prisoner's Dilemma Simulator

A comprehensive web application for simulating thousands of Prisoner's Dilemma games using Monte Carlo sampling of probabilistic strategies. This project calculates average payoffs, visualizes outcomes, and highlights how different strategies perform under varying cooperation probabilities.

## 🎯 Features

- **Multiple Strategy Types**: Probabilistic, Always Cooperate, Always Defect
- **Monte Carlo Simulation**: Run thousands of game rounds with statistical accuracy
- **Interactive Web Interface**: Beautiful, responsive frontend with real-time visualization
- **Data Visualization**: Charts showing payoffs, cooperation rates, and outcome distributions
- **RESTful API**: Backend API for programmatic access and integration
- **Comprehensive Testing**: Unit tests for simulation logic validation

## 🏗️ Project Structure

```
monte-carlo-prisoners-dilemma/
├── backend/
│   └── app.py              # Flask API server
├── index.html              # Frontend web interface
├── test_simulation.py      # Unit tests
├── requirements.txt        # Python dependencies
├── package.json           # Node.js dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- Node.js (optional, for frontend development)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd monte-carlo-prisoners-dilemma
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies (optional)**
   ```bash
   npm install
   ```

### Running the Application

1. **Start the Flask backend**
   ```bash
   cd backend
   python app.py
   ```
   The API will be available at `http://127.0.0.1:5000`

2. **Open the frontend**
   - Open `index.html` in your web browser
   - Or serve it with a local server:
   ```bash
   # Using Python
   python -m http.server 8000
   # Then visit http://127.0.0.1:8000
   ```

## 🎮 Usage

### Web Interface

1. Open `index.html` in your browser
2. Adjust simulation parameters:
   - **Cooperation Probabilities**: Set how likely each player is to cooperate (0-1)
   - **Strategies**: Choose from Probabilistic, Always Cooperate, or Always Defect
   - **Rounds**: Number of game rounds to simulate (100-1,000,000)
3. Click "Run Simulation" to see results
4. View interactive charts showing payoffs and cooperation patterns

### API Usage

#### Single Simulation
```bash
curl -X POST http://127.0.0.1:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "player1_prob": 0.7,
    "player2_prob": 0.3,
    "rounds": 1000,
    "strategy1": "probabilistic",
    "strategy2": "always_cooperate"
  }'
```

#### Batch Simulations
```bash
curl -X POST http://127.0.0.1:5000/batch_simulate \
  -H "Content-Type: application/json" \
  -d '{
    "simulations": [
      {"player1_prob": 0.5, "player2_prob": 0.5, "rounds": 1000},
      {"player1_prob": 0.8, "player2_prob": 0.2, "rounds": 1000}
    ]
  }'
```

#### Get Available Strategies
```bash
curl http://127.0.0.1:5000/strategies
```

## 📊 Prisoner's Dilemma Payoff Matrix

|                | Player 2: Cooperate | Player 2: Defect |
|----------------|-------------------|------------------|
| **Player 1: Cooperate** | (3, 3) | (0, 5) |
| **Player 1: Defect**   | (5, 0) | (1, 1) |

- **Mutual Cooperation**: Both players get 3 points
- **Mutual Defection**: Both players get 1 point  
- **Temptation**: Defecting player gets 5 points, cooperating player gets 0
- **Sucker's Payoff**: Cooperating player gets 0 points, defecting player gets 5

## 🧪 Testing

Run the test suite to validate simulation logic:

```bash
python test_simulation.py
```

Tests cover:
- Probabilistic strategy accuracy
- Always cooperate/defect strategies
- Mixed strategy combinations
- Statistical validation of results
- Large-scale simulations

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page with API documentation |
| `/simulate` | POST | Run a single Monte Carlo simulation |
| `/batch_simulate` | POST | Run multiple simulations |
| `/strategies` | GET | Get available strategy types |

## 📈 Simulation Parameters

- **player1_prob**: Probability that Player 1 cooperates (0-1)
- **player2_prob**: Probability that Player 2 cooperates (0-1)  
- **rounds**: Number of game rounds to simulate (1-1,000,000)
- **strategy1**: Player 1's strategy type
- **strategy2**: Player 2's strategy type

## 🎨 Technologies Used

- **Backend**: Flask, PyTorch, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Testing**: Python unittest
- **Dependencies**: See `requirements.txt` and `package.json`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔬 Research Applications

This simulator can be used for:
- Game theory research
- Strategy analysis in economics
- Behavioral economics studies
- Machine learning algorithm testing
- Educational demonstrations of the Prisoner's Dilemma

## 🐛 Troubleshooting

**Flask app not starting:**
- Ensure virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version compatibility

**Frontend not connecting to backend:**
- Ensure Flask app is running on `http://127.0.0.1:5000`
- Check browser console for CORS errors
- Verify network connectivity

**Simulation errors:**
- Check that probabilities are between 0 and 1
- Ensure rounds count is reasonable (100-1,000,000)
- Review error messages in browser console or Flask logs
