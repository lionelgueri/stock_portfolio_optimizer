# Install PyPortfolioOpt

#!pip install gradio
#!pip install PyPortfolioOpt
#!pip install yfinance
#!pip install yfinance --upgrade

# Import Libraries
import gradio as gr
import yfinance as yf

from datetime import datetime
from dateutil.relativedelta import relativedelta

#from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
#import seaborn as sns

# Init period
end_date = str(datetime.today().strftime("%Y-%m-%d"))
dtstart = datetime.today() - relativedelta(years=1)
start_date = str(dtstart.strftime("%Y-%m-%d"))
print('Date fin: '+end_date+'\nDate début: '+start_date)

# Init tickers
#tickers = ['TTE.PA', 'BNP.PA', 'SAN.PA', 'HO.PA', 'ATO.PA']
tickers_def = 'TTE.PA,BNP.PA,SAN.PA,HO.PA,ATO.PA'
ind_ticker_def = '%5EFCHI'
print('Liste des tickers: '+tickers_def+'\nTicker indice: '+ind_ticker_def)

# Init Risk free rate
rf_rate_def = '0.03'
print('Risk free rate: '+rf_rate_def)

# Plot Individual Cumulative Returns
def plot_cum_returns(data, title):
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod()*100
    fig = px.line(daily_cum_returns, title=title)
    return fig

# Efficient frontier and max sharpe
def plot_efficient_frontier_and_max_sharpe(rf_rate_str, mu, S):

    # Optimize portfolio for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(8,6))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the max sharpe portfolio
    rf_rate = float(rf_rate_str)
    ef_max_sharpe.max_sharpe(risk_free_rate = rf_rate)
    ret_tangent, std_tangent, _ =   ef_max_sharpe.portfolio_performance(risk_free_rate = rf_rate)
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r",     label="Max Sharpe")

    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Frontière efficiente avec des Portefeuilles aléatoires")
    ax.legend()
    return fig

# Get data
def output_results(rf_rate_str, start_date, end_date, tickers_string, ind_ticker_string):
    tickers = tickers_string.split(',')
    
    # Get Stock Prices
    stocks_df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if stocks_df.empty:
                stocks_df = data[['Close']].rename(columns={'Close': ticker})
            else:
                stocks_df = stocks_df.join(data['Close'].rename(ticker))
        except Exception as e:
            print(f"Failed to download data for {ticker}. Error: {e}")
    
    indice_df = yf.Ticker(ind_ticker_string).history(start=start_date, end=end_date)
    
    # convert date format
    stocks_df.index = stocks_df.index.strftime('%Y-%m-%d')
    indice_df.index = indice_df.index.strftime('%Y-%m-%d')
    
    # Plot Individual Stock Prices
    fig_indiv_prices = px.line(stocks_df, title='Prix des Actions individuelles')
        
    # Plot Individual Cumulative Returns
    fig_cum_returns = plot_cum_returns(stocks_df, "Rendements cumulés des actions individuelles en partant de 100 euros")
    
    # Calculate and Plot Correlation Matrix between Stocks
    corr_df = stocks_df.corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True, title = 'Corrélation entre les Actions')

    # Calculate expected returns and sample covariance matrix for portfolio optimization later
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)

    # Plot efficient frontier curve
    fig_efficient_frontier = plot_efficient_frontier_and_max_sharpe(rf_rate_str, mu, S)

    # Get optimized weights
    rf_rate = float(rf_rate_str)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=rf_rate)
    weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate = rf_rate)
    
    expected_annual_return, annual_volatility, sharpe_ratio = '{}%'.format((expected_annual_return*100).round(2)), \
    '{}%'.format((annual_volatility*100).round(2)), \
    '{}%'.format((sharpe_ratio*100).round(2))
    
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df = weights_df.reset_index()
    weights_df.columns = ['Tickers', 'Poids']

    # Calculate returns of portfolio with optimized weights
    stocks_df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
        stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
    
    # Fusionner les deux séries sur l'index (dates)
    print('Dimension série stocks: '+str(stocks_df['Optimized Portfolio'].shape))
    print('Dimension série indice: '+str(indice_df['Close'].shape))
    
    data_df = pd.DataFrame({
        'Optimized Portfolio': stocks_df['Optimized Portfolio'],
        'Ref Indice': indice_df['Close']
        })
    
    # Plot Cumulative Returns of Optimized Portfolio
    fig_cum_returns_optimized = plot_cum_returns(data_df, "Rendements cumulés d'un portefeuille optimisé en partant de 100 euros")
    #fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')

    return  fig_cum_returns_optimized, weights_df, fig_efficient_frontier, fig_corr,   \
            expected_annual_return, annual_volatility, sharpe_ratio, fig_indiv_prices, fig_cum_returns

# User interface with Gradio
with gr.Blocks() as app:
    with gr.Row():
        gr.HTML("<h1>Application d'optimisation de portefeuille d'actions avec PyPortfolioOpt & Gradio</h1>")
    
    with gr.Row():
        start_date = gr.Textbox(start_date, label="Date début")
        end_date = gr.Textbox(end_date, label="Date finale")
    
    with gr.Row(): 
        rf_rate_str = gr.Textbox(rf_rate_def, label="Taux d'intérêt sans risque")
        
    with gr.Row():        
        tickers_string = gr.Textbox(tickers_def, 
                                    label='Entrer tous les tickers (Yahoo Finance) à inclure dans le portefeuille séparés par\
                                    des virgules SANS espaces, ex. "META,AMZN,AAPL,TSLA"')
        ind_ticker_string = gr.Textbox(ind_ticker_def, label="Entrer le ticker de l'indice de référence \
                                       ex. pour CAC40 entrer '^FCHI'")
        btn = gr.Button("Obtenir le portefeuille optimisé")
        
    # Outputs
    with gr.Row():
        expected_annual_return = gr.Text(label="Rendement annuel attendu")
        annual_volatility = gr.Text(label="Volatilité annuelle")
        sharpe_ratio = gr.Text(label="Sharpe Ratio")            
   
    with gr.Row():        
        fig_cum_returns_optimized = gr.Plot(label="Rendements cumulés d'un portefeuille optimisé en partant de 100 euros")
        weights_df = gr.DataFrame(label="Pondérations optimisées de chaque ticker")
        
    with gr.Row():
        fig_efficient_frontier = gr.Plot(label="Frontière efficiente")
        fig_corr = gr.Plot(label="Corrélation entre les Actions")
    
    with gr.Row():
        fig_indiv_prices = gr.Plot(label="Prix des Actions individuelles")
        fig_cum_returns = gr.Plot(label="Rendements cumulés des actions individuelles en partant de 100 euros")    

    # Action button
    btn.click(fn=output_results, inputs=[rf_rate_str, start_date, end_date, tickers_string, ind_ticker_string], 
              outputs=[fig_cum_returns_optimized, weights_df, fig_efficient_frontier, fig_corr,   \
            expected_annual_return, annual_volatility, sharpe_ratio, fig_indiv_prices, fig_cum_returns])

# Lauch the app
app.launch()