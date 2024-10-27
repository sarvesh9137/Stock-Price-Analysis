# Stock-Price-Analysis
Stock Price Analysis Using LSTM And SVM
Here's a draft README for your project:

---

 Stock Price Analysis with LSTM and SVM

This Streamlit application uses LSTM (Long Short-Term Memory) and SVM (Support Vector Machine) models to analyze and predict stock prices based on historical data retrieved from Yahoo Finance. The app provides tools for financial analysis and metrics to aid in understanding stock performance.

 Features

- Interactive Ticker Input: Users can input a stock ticker symbol (e.g., AAPL, TSLA) to fetch historical data.
- Data Visualization: The application displays the closing price trends of the selected stock over time.
- Financial Metrics: Calculates and shows the annual return, standard deviation, and risk-adjusted return (Sharpe Ratio).
- LSTM Model: A neural network model is used to predict stock prices based on past data. Performance is measured using Root Mean Square Error (RMSE).
- SVM Model: A Support Vector Machine model is also used for stock price prediction, providing a comparative approach with the LSTM model.
- Comparative Visualization: Original closing prices are plotted against predictions from both the LSTM and SVM models.
- Model Evaluation: Displays RMSE values for both models to help users understand which model performs better for their stock selection.

 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/StockPriceAnalysis.git
   cd StockPriceAnalysis
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

 Usage

1. Enter a stock ticker symbol in the sidebar, along with start and end dates.
2. View stock price data, financial metrics, and model predictions.
3. The app displays a comparison of LSTM and SVM predictions, alongside original values, to aid in evaluating model accuracy.
4. The RMSE values for both models are shown to provide insights into the model performances.

 Dependencies

- Streamlit - For creating the interactive web application interface.
- yFinance - For retrieving historical stock data from Yahoo Finance.
- NumPy & Pandas - For data manipulation and financial calculations.
- Matplotlib & Plotly - For visualizing stock prices and model predictions.
- Scikit-learn - For implementing the SVM model and preprocessing data.
- TensorFlow/Keras - For building and training the LSTM model.

 Models

- LSTM Model: Built with TensorFlow's Keras API, this model uses two LSTM layers followed by a dense output layer. The model's predictions are plotted against actual closing prices.
- SVM Model: Uses Scikit-learn’s SVR (Support Vector Regression) with an RBF kernel. This model provides a non-neural network-based comparison for predicting stock prices.

 Project Structure

```
📦StockPriceAnalysis
 ┣ 📜app.py                     Main application file
 ┣ 📜README.md                  Project documentation
 ┣ 📜requirements.txt           Dependencies
 ┗ 📂data                       Directory to store or download stock data (optional)
```

 Disclaimer

This project is for educational and illustrative purposes only. Stock predictions are complex, and these models are not intended for real financial trading or investment.

---

CODE WITH SARVESH (CWS)
