import pandas as pd
import matplotlib.pyplot as plt


class Tst_class():

    # def __init__(self, filename):
    #     self.filename = filename
    #     data = 

    def get_data(self, filename):
        self.df = pd.read_excel(filename)

    def preview_data(self):
        print("Head of dataframe:\n", self.df.head())
        print("Tail of dataframe:\n", self.df.tail())


dirpath = 'test_data1.xlsx'
lr = Tst_class()
lr.get_data(dirpath)
lr.preview_data()
