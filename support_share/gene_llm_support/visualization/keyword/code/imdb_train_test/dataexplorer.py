import pandas as pd
from sklearn.model_selection import train_test_split

class DataExplorer():
    def __init__(self,path,test_size,random_seed):
        self.test_size = test_size
        self.random_seed = random_seed
        self.path = path
        self.dataframe = self.set_dataframe()
        self.assign_sentiment()
        self.dataframe_train = None
        self.dataframe_test = None
        self.dataframe_val = None
        self.set_train_test_val_data()

    def set_dataframe(self):
        return pd.read_csv(self.path)
    
    def print_dataframe(self):
        print(self.dataframe)
    
    def print_head(self):
        print(self.dataframe.head())

    def get_dataframe(self):
        return  self.dataframe
    
    def get_train(self):
        return self.dataframe_train
    
    def get_test(self):
        return self.dataframe_test
    
    def get_val(self):
        return self.dataframe_val
    
    def get_train_len(self):
        return self.dataframe_train.shape[0]
    
    def get_val_len(self):
        return self.dataframe_val.shape[0]
    
    def get_test_len(self):
        return self.dataframe_test.shape[0]
    
    def get_nclasses(self):
        return len(self.dataframe['sentiment'].unique())
    
    def assign_sentiment(self):
        self.dataframe['sentiment'] =  self.dataframe['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    def set_train_test_val_data(self):
        self.dataframe_train, temp_test = train_test_split(self.dataframe,test_size=self.test_size,random_state=self.random_seed)
        self.dataframe_test, self.dataframe_val = train_test_split(temp_test,test_size=0.5,random_state=self.random_seed)