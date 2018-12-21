# =============================================================================
# Using Python 3.5
# Using Spyder IDE
# FINAL PROJECT
# ISYE 6767
# DESIGN AND IMPLEMENTATION OF SYSTEMS TO SUPPORT COMPUTATIONAL FINANCE
# YASHOVARDHAN JALLAN
# =============================================================================
import datetime as dt
import numpy as np
import pandas as pd
from pandas_datareader import data
from sklearn import linear_model
import os

class Security:
    
    #Constructor
    def __init__(self,tf):
        self.ticker_file=tf               
    
    #this function is using Pandas_datareader to pull stock data
    #from the internet and store it in the Data folder as CSV file for each stock
    #not all stocks returns data, hence i have written the try and except loop
    def store_data(self,symb):
        start_date = dt.datetime(2011,1,1)
        end_date = dt.datetime(2015,7,31)
        data_source = "yahoo"        
        print(symb)
        
        max_tries=5 # the max number of times we are trying to retrieve stock data if not collected in first attempt
        count=0 #just a count variable to break the loop is max_tries reached
        
        while True:
            try:            
                web_df=data.DataReader(symb,data_source,start_date, end_date)[['Adj Close','Volume']].fillna(method='ffill')
                filepath="./data/"+str(symb)+".csv"
                web_df.to_csv(filepath)
            except:
                print("there was an exception, retrying")                
                count+=1
                if (count==max_tries):
                    print("Max tries reached")
                    return 0
                continue            
            break                 
        return 0
        
    
    def get_stock_df_dict(self):                
        ticker_df=pd.read_csv(self.ticker_file,usecols=(0, 1))
        
        #Coverting SH to SS
        #Converting Market Cap from string to float
        for i in range(len(ticker_df)):                     
            if "SH" in ticker_df["ticker"][i]:
                ticker_df["ticker"][i] = ticker_df["ticker"][i].replace("SH", "SS")
            
            ticker_df["mktshare"][i] = float(ticker_df["mktshare"][i].replace(",", ""))
                
        #keep only the stocks with market cap greater than 500,000,000
        ticker_df=ticker_df[ticker_df["mktshare"]>=500000000]        
        
        #this for loop calls the function which reads data from internet
        #I have commented it because I am storing the data in CSV and I do not
        #want to run this again as this takes a long time to run
# =============================================================================
#         for i in range(len(ticker_df)):            
#             #print(stock_df.iloc[i]["ticker"])            
#             self.store_data(stock_df.iloc[i]["ticker"])            
# =============================================================================
            
        stock_df_dict={}
        ticker_list=os.listdir("./data")
        #for i in range(10): 
        for i in range(len(ticker_list)): 
            symb=ticker_list[i].replace(".csv","")
            file_path="./data/"+ticker_list[i]
            df_stock=pd.read_csv(file_path,index_col=["Date"])
            
            if (symb[-2:]=="SS"):
                file_path="./szss/ss/"+symb+"_factor.csv"
            else:
                file_path="./szss/sz/"+symb+"_factor.csv"
            success=False
            try:
                df_accounting=pd.read_csv(file_path,index_col=["Date"])    
                df_combined=df_stock.merge(df_accounting,how="left",left_index=True,right_index=True)            
                success=True
            except:            
                #print(file_path)
                pass
            if(success):
                stock_df_dict[symb]=df_combined                
        
        return stock_df_dict       

class Security_Market:        
    #Constructor
    def __init__(self):
        pass
    def store_data(self):
        start_date = dt.datetime(2011,1,1)
        end_date = dt.datetime(2015,7,31)
        data_source = "yahoo"        
        symb="000300.SS" #this is collecting only market data
        
        max_tries=5 # the max number of times we are trying to retrieve stock data if not collected in first attempt
        count=0
        while True:
            try:            
                web_df=data.DataReader(symb,data_source,start_date, end_date)[['Adj Close']].fillna(method='ffill')
                filepath="./mkt_data/"+str(symb)+".csv"
                web_df.to_csv(filepath)
            except:
                print("there was an exception, retrying")                
                count+=1
                if (count==max_tries):
                    print("Max tries reached")
                    return 0
                continue            
            break                 
        return web_df

class Strategy:
    def __init__(self,st_dict,n_window,m_window,l_window,u_freq,index_list,start_port_val,transaction_fraction):
        self.stock_dict_df=st_dict   
        self.n_window=n_window
        self.m_window=m_window
        self.l_window=l_window
        self.u_freq=u_freq
        self.index_list=index_list
        self.start_port_val=start_port_val
        self.transaction_fraction=transaction_fraction
    
    def get_M_score_df_and_optimal_weight_list(self):
        #print("Creating the M_Score DataFrame")
        #print()
        key_list=list(self.stock_dict_df.keys())
        #print(len(key_list))
        m_score_df=pd.DataFrame(index=self.index_list)       #initializing an empty data frame
        min_avg_vol=1000000 #given in the question
        
        regress_weight_df=pd.DataFrame(columns=['PB', 'PCF', 'PE', 'PS', 'Log_Ret', 'PM', 'PRev', 'Vol'])
        
        #for i in range(3):
        
        for i in range(len(key_list)):
            #print(key_list[i])
            symb=key_list[i]
            df=self.stock_dict_df[symb]
            
            #creating columns in this dataframe
            df["Log_Ret"]=np.log(df["Adj Close"]/df["Adj Close"].shift(1))    
            df["Avg_15d_Volume"]=df["Volume"].rolling(window=15).mean()
            df["PM"]=np.log(df["Adj Close"].shift(1)/df["Adj Close"].shift(n_window))
            df["PRev"]=np.log(df["Adj Close"].shift(m_window)/df["Adj Close"].shift(1))
            df["Vol"]=df["Log_Ret"].rolling(window=l_window).std()
            #print(df)        
            #this next column stores M_score values. I have named it by the ticker name as I am joining the M_score columns in one dataframe and need to identify M_scores by their ticker
            # I have calculated M-Score for only those days when the avg_15d_volume is greater than 1 million
            
            df=df.drop(['Adj Close', 'Volume','Avg_15d_Volume'],axis=1)
            #print("head")
            #print(df.head())            
            #print("regress")
            #print(regress_weight_df.head())            
            regress_weight_df=regress_weight_df.append(df,ignore_index=True)            
        
        #Cleaning the regress weights dataframe before running regression
        regress_weight_df=regress_weight_df.dropna()
        regress_weight_df=regress_weight_df[regress_weight_df["Log_Ret"]!=0]
        
       
        #Running Regression to get optimal weights
        Y = regress_weight_df[['Log_Ret']]
        X = regress_weight_df[['PB', 'PCF', 'PE', 'PS', 'PM', 'PRev', 'Vol']]    
        model=linear_model.LinearRegression()
        Reg_Mod=model.fit(X,Y)
        #accuracy = Reg_Mod.score(X,Y)
        weights_list=Reg_Mod.coef_        
        #normalizing weights to get sum 1
        weights_list=weights_list/weights_list.sum()        
        
        #calculating M Score based on the optimized weights
        #for i in range(5):
        
        for i in range(len(key_list)):
            symb=key_list[i]
            df=self.stock_dict_df[symb]
            
            df[symb]=weights_list[0][0]*df["PB"]+weights_list[0][1]*df["PCF"]+weights_list[0][2]*df["PE"]+weights_list[0][3]*df["PS"]+weights_list[0][4]*df["PM"]+weights_list[0][5]*df["PRev"]+weights_list[0][6]*df["Vol"].where(df["Avg_15d_Volume"]>=min_avg_vol,np.nan)
            m_score_df=m_score_df.join(df[symb])              
        
        #return self.stock_dict_df[symb],weights_list  
        return m_score_df,weights_list            
    
    def top_100_stocks(self,m_df,s_df_dict,i,pos_val):        
        m_score_series=m_df.iloc[i].sort_values(ascending=False).iloc[:100]
        m_score_series=list(m_score_series.index)
        
        df=pd.DataFrame(columns=["Ticker", "Adj Close", "Shares"])
        df["Ticker"]=m_score_series                
        
        for t,ticker in enumerate(m_score_series):        
            #print(t,ticker)
            try:
                df.iloc[t]["Adj Close"]=s_df_dict[ticker]["Adj Close"].iloc[i]
                df.iloc[t]["Shares"]=pos_val/ (df.iloc[t]["Adj Close"])            
            except:
                df.iloc[t]["Adj Close"]=0
                df.iloc[t]["Shares"]=0
            
        df["Pos_Val"]=df["Shares"]*df["Adj Close"]
        return df
    
    def portfolio_value(self,previousPortfolio_df,s_df_dict, i):
        port_val = 0
        #k = 0
        new_prices=[]
        for t,ticker in enumerate(previousPortfolio_df["Ticker"]):            
            try:
                new_prices.append(s_df_dict[ticker]["Adj Close"].iloc[i])
            except:
                new_prices.append(0)
            
        previousPortfolio_df["New_Prices"]=new_prices
        previousPortfolio_df["New_Pos_Val"]=previousPortfolio_df["New_Prices"]*previousPortfolio_df["Shares"]
        
        port_val=previousPortfolio_df["New_Pos_Val"].sum()
        return previousPortfolio_df,port_val
    
    def sellEquities(self,previousPortfolio_df, equitiesToBeSold):

        #gains from sell
        #gain = 0
        temp_df=previousPortfolio_df.copy()
        temp_df=temp_df.loc[temp_df["Ticker"].isin(equitiesToBeSold)]
        return temp_df,temp_df["New_Pos_Val"].sum()
        #return temp_df["New_Pos_Val"].sum()
    
    def buyEquities(self,newPortfolio_df, equitiesToBeBought):
        
        temp_df=newPortfolio_df.copy()
        temp_df=temp_df.loc[temp_df["Ticker"].isin(equitiesToBeBought)]
        #print(temp_df)
        return temp_df,temp_df["Pos_Val"].sum()
    
    def rebalance(self,heldPortfolio_df,newPortfolio_df):
        temp1_df=heldPortfolio_df.copy()
        temp2_df=newPortfolio_df.copy()
        temp1_df=temp1_df.set_index("Ticker")
        temp2_df=temp2_df.set_index("Ticker")
        temp2_df.rename(columns={"Pos_Val": "Target"},inplace=True)
        
        temp1_df=temp1_df.join(temp2_df["Target"])
        temp1_df["Diff"]=temp1_df["New_Pos_Val"]-temp1_df["Target"]

        diff=temp1_df["Diff"].sum()
        
        gain_in_reb=0
        loss_in_reb=0
        if diff>=0:
            gain_in_reb=diff
        else:
            loss_in_reb=diff
        #return temp1_df,temp2_df
        return loss_in_reb,gain_in_reb
    
    def get_performance(self,portfolio_value_list,profitlist,u_freq):
        
        cols = ["Portfolio Value", "Annulized Return"]
        performance_df=pd.DataFrame(columns=cols)
        performance_df["Portfolio Value"]=portfolio_value_list        
        #annualized
        performance_df["Annulized Return"]=((performance_df["Portfolio Value"]/performance_df["Portfolio Value"].shift(1))-1)*np.sqrt(252/u_freq)
        
        PNL=np.sum(profitlist)
        avg_PNL=PNL/performance_df.shape[0]
        #print(PNL)
        #print(np.sum(profitlist))
        Avg_Return=(performance_df["Annulized Return"].sum())*100
        std_dev=np.std(performance_df["Annulized Return"])*100
        #print(std_dev)
        
        #assuming risk free rate to be 0
        Sharpe_ratio=Avg_Return/std_dev
        
        #Max Drawdown
        md=0
        max_val=performance_df["Portfolio Value"].iloc[0]
        for i in range(1,performance_df.shape[0]):
            
            if (performance_df["Portfolio Value"].iloc[i] > max_val):
                max_val=performance_df["Portfolio Value"].iloc[i]
                #print("Max Val",max_val)
            if (max_val-performance_df["Portfolio Value"].iloc[i]> md):
                md=max_val-performance_df["Portfolio Value"].iloc[i]   
                #print("Max Drawdown",md)
        
        #Percentage of Winning/Losing Trades
        total_trades=len(profitlist)
        winning_trades=len([element for element in profitlist if element>=0])
        losing_trades=len([element for element in profitlist if element<0])
        pct_winning_trades=winning_trades/total_trades*100
        pct_losing_trades=losing_trades/total_trades*100       
        
        return PNL,avg_PNL,Avg_Return,Sharpe_ratio,md,std_dev,pct_winning_trades,pct_losing_trades,performance_df["Portfolio Value"]
        #Avg Returns, and MD are in percentage terms

    def run_strategy(self,m_df,s_df_dict):
        #STARTS HERE
        m_df=m_df.copy()
        s_df_dict=stock_df_dict.copy()
        
        i=self.u_freq
        portfolioValue=start_port_val
        
        position_value=0.01*portfolioValue
        initialPortfolio_df=self.top_100_stocks(m_df,s_df_dict,i,position_value)
        
        portfolio_value_list=[]
        portfolio_value_list.append(portfolioValue)
        
        profitlist = []
        profitlist.append(-portfolioValue*transaction_fraction)
        
        transactionCost = []
        transactionCost.append(portfolioValue * transaction_fraction)
            
        #while (i+u_freq) < 70:
        while (i+u_freq) < m_df.shape[0]:              
            
            oldportfolioValue=portfolioValue
            i+=u_freq
            
            previousPortfolio_df=initialPortfolio_df.copy()
            
            previousPortfolio_df,portfolioValue=self.portfolio_value(previousPortfolio_df,s_df_dict,i)
            
            position_value=0.01*portfolioValue
            newPortfolio_df=self.top_100_stocks(m_df,s_df_dict,i,position_value)
            
            equitiesToBeSold=list(set(previousPortfolio_df['Ticker']) - set(newPortfolio_df['Ticker']))
            sell_df,gain=self.sellEquities(previousPortfolio_df,equitiesToBeSold)
            
            equitiesToBeBought = list(set(newPortfolio_df['Ticker']) - set(previousPortfolio_df['Ticker']))
            buy_df,loss=self.buyEquities(newPortfolio_df,equitiesToBeBought)
            
            equitiesToBeHeld=list(set(previousPortfolio_df['Ticker']).intersection(set(newPortfolio_df['Ticker'])))
            heldPortfolio_df=previousPortfolio_df.loc[previousPortfolio_df["Ticker"].isin(equitiesToBeHeld)]
            
            lossInRebalance, gainInRebalance=self.rebalance(heldPortfolio_df,newPortfolio_df)
            
            portfolio_value_list.append(portfolioValue)
            
            tc=(abs(gain) + abs(loss) + abs(lossInRebalance) + abs(gainInRebalance)) * transaction_fraction
            transactionCost.append(tc)
            
            profit = portfolioValue - oldportfolioValue - tc
            profitlist.append(profit)
            
            initialPortfolio_df=newPortfolio_df.copy()
        
        PNL,avg_PNL,Avg_Return,Sharpe_ratio,Max_Drawdown,std_dev,pct_winning_trades,pct_losing_trades,Port_val_series= self.get_performance(portfolio_value_list,profitlist,self.u_freq)
        
        return PNL,avg_PNL,Avg_Return,Sharpe_ratio,Max_Drawdown,std_dev,pct_winning_trades,pct_losing_trades,Port_val_series

if __name__ == '__main__':
    
    print("Running the Code")
    print()
    
    ticker_file="ticker_universe.csv"   
        
    #Creating an Instance of Market Security Class
    # I have run this omce and now I have commented this because it uses internet, instead I have downloaded the file and can read from my directory
# =============================================================================
#     mkt_sec_inst=Security_Market()
#     market_df=mkt_sec_inst.store_data()
# =============================================================================
    market_df=pd.read_csv("./mkt_data/000300.SS.csv",index_col=["Date"])
    
    #Strategy Class parameters initialization
    # I AM FININDING OPTIMAL WEIGHTS FOR all these parameters everytime i call the class
    n_window=20
    m_window=15
    l_window=20
    u_freq=15
    #n_window_list=[2,5,10,20]
    #m_window_list=[10,15,20,30]
    #l_window_list=[15,20,30,40]
    #u_freq_list=[20,15,10]
    start_port_val=10000000
    transaction_fraction=0.001
    index_list=list(market_df.index)   
    
    #cols=["N","M","L","U","Wt_PB","Wt_PCF","Wt_PE","Wt_PS","Wt_PM","Wt_PRev","Wt_Vol","PNL","avg_PNL","Avg_Return","Sharpe_ratio","Max_Drawdown","std_dev","pct_winning_trades","pct_losing_trades","Final_Port_val"]
    #optimization_df_insample=pd.DataFrame(columns=cols)
    #optimization_df_outsample=pd.DataFrame(columns=cols)
    
    #count=0
    # I used this for loop for running the optimization of the strategy test.
    # I have commented it and only put in the most optimized set of values I received.
    # This loop takes time to run.
    """
    for n_window in n_window_list:
        for m_window in m_window_list:
            for l_window in l_window_list:
                for u_freq in u_freq_list:    
                    """                    
                    
    print("Getting Stock Data for each ticker")
    print()
    
    #Creating an Instance of Security Class    
    sec_inst=Security(ticker_file)
    stock_df_dict=sec_inst.get_stock_df_dict()
    #len(stock_df_dict) =1321
    #we get 1321 stocks finally whose Market cap is greater than 500,000,000 and the accounting ratios are available
    
    #Creating an Instance of Strategy Class        
    
    print("Getting optimal weights using Cross-Sectional regression and Calculating M-Score for all stocks")
    print()
    
    copy_of_stock_df_dict=dict(stock_df_dict)
    str_inst=Strategy(copy_of_stock_df_dict,n_window,m_window,l_window,u_freq,index_list,start_port_val,transaction_fraction)
    m_score_df,weight_list=str_inst.get_M_score_df_and_optimal_weight_list()
     
    m_score_df_insample=m_score_df.loc[:'2014-10-31']
    m_score_df_outsample=m_score_df.loc['2014-11-01':]
    
    print("Using M-score, the trading strategy is now executed")
    print()
    
    #running the project for in-sample testing and getting the performance measures
    PNL,avg_PNL,Avg_Return,Sharpe_ratio,Max_Drawdown,std_dev,pct_winning_trades,pct_losing_trades,Port_val_series=str_inst.run_strategy(m_score_df_insample,stock_df_dict)
    """
    print("In Sample Portfolio Results")
    print()        
    print("Total Profit/Loss (in $): ",PNL)
    print("Average Profit/Loss per trade (in $): ",avg_PNL)
    print("Annualized Return on Portfolio (%): ",Avg_Return)
    print("Portfolio Sharpe Ratio: ",Sharpe_ratio)    
    print("Maximum Drawdown ($): ",Max_Drawdown)
    print("Portfolio Volatility (Annualized) (in %): ",std_dev)
    print("Percentage of winning trades: ",pct_winning_trades)
    print("Percentage of losing trades: ",pct_losing_trades)
    
    
    Port_val_series=Port_val_series/start_port_val # to create it in terms of $1
    Port_val_series.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Growth of $1 over with the trades made over time')                       
    """
    #temp_list=[[n_window,m_window,l_window,u_freq,weight_list[0][0],weight_list[0][1],weight_list[0][2],weight_list[0][3],weight_list[0][4],weight_list[0][5],weight_list[0][6],PNL,avg_PNL,Avg_Return,Sharpe_ratio,Max_Drawdown,std_dev,pct_winning_trades,pct_losing_trades,float(Port_val_series[-1:])]]
    #temp_df=pd.DataFrame(temp_list,columns=cols)
    #optimization_df_insample=optimization_df_insample.append(temp_df)
    
    #Port_val_series=Port_val_series/start_port_val # to create it in terms of $1
    #Port_val_series.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Growth of $1 over with the trades made over time')
    
    #running the project for out-sample testing and getting the performance measures
    PNL,avg_PNL,Avg_Return,Sharpe_ratio,Max_Drawdown,std_dev,pct_winning_trades,pct_losing_trades,Port_val_series=str_inst.run_strategy(m_score_df_outsample,stock_df_dict)
    print()
    print("Out Sample Portfolio Results")
    print()        
    print("Total Profit/Loss (in $): ",PNL)
    print("Average Profit/Loss per trade (in $): ",avg_PNL)
    print("Annualized Return on Portfolio (%): ",Avg_Return)
    print("Portfolio Sharpe Ratio: ",Sharpe_ratio)    
    print("Maximum Drawdown ($): ",Max_Drawdown)
    print("Portfolio Volatility (Annualized) (in %): ",std_dev)
    print("Percentage of winning trades: ",pct_winning_trades)
    print("Percentage of losing trades: ",pct_losing_trades)
    
    #temp_list=[[n_window,m_window,l_window,u_freq,weight_list[0][0],weight_list[0][1],weight_list[0][2],weight_list[0][3],weight_list[0][4],weight_list[0][5],weight_list[0][6],PNL,avg_PNL,Avg_Return,Sharpe_ratio,Max_Drawdown,std_dev,pct_winning_trades,pct_losing_trades,float(Port_val_series[-1:])]]
    #temp_df=pd.DataFrame(temp_list,columns=cols)
    #optimization_df_outsample=optimization_df_outsample.append(temp_df)
    
    Port_val_series=Port_val_series/start_port_val # to create it in terms of $1
    Port_val_series.plot(lw=2,colormap='jet',marker='.',markersize=10,title='Growth of $1 over with the trades made over time')                       
                    
    #optimization_df_insample.to_csv("optimization_df_insample.csv")
    #optimization_df_outsample.to_csv("optimization_df_outsample.csv")
    
    
    
    
    
        
     
    
    
   
    