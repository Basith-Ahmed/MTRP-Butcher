# Market Trend Prediction Model

## Overview
This project develops a logistic regression model to predict market trend reversals, specifically identifying potential "buy" or "sell" opportunities in financial markets. The model analyzes technical indicators and price data to make predictions.

## Features
- Utilizes Logistic Regression for binary classification of market trends.
- Processes and analyzes data using technical indicators like EMAs, RSI, MACD, and StochRSI.
- Uses data from Yahoo Finance for the period from January 2020 to January 2024.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/Basith-Ahmed/MTRP-Butcher.git
cd mtrp-butcher
```

## Usage
To run the model training and prediction script, navigate to the project directory and run:
```bash
python Butcher_UI.py
```

## Directory Structure
```bash
mtrp-butcher/
│
├── Butcher_Model.sav
├── Butcher_UI.py
├── requirements.txt
└── README.md
```

## Data
The data used in this project is sourced from Yahoo Finance, covering daily price movements of Bitcoin (BTC-USD) from January 1, 2020, to January 1, 2024 by default which you can change as required.

## Configurations
Edit the config.py file to modify the parameters of the logistic regression model, including the choice of technical indicators and the thresholds for "buy" and "sell" predictions.

## Contributing
Contributions to this project are welcome. To contribute: <br/>

- Fork the repository. <br/>
- Create a new branch (git checkout -b feature-branch).<br/>
- Make your changes.<br/>
- Commit your changes (git commit -am 'Add some feature').<br/>
- Push to the branch (git push origin feature-branch).<br/>
- Submit a new Pull Request.<br/>

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Basith Ahmed - [Link](https://www.linkedin.com/in/basith-ahmed/) <br/>
Project - [Link](https://github.com/Basith-Ahmed/MTRP-Butcher)

## Acknowledgements
Yahoo Finance for providing the data.<br/>
Contributors who have participated in this project.
