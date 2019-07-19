import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class Tst_class():

    def preprocessing_data(self, filename):
        self.df = pd.read_excel(filename)
        self.df['Sale'] = self.df['amount']/self.df['quantity']  # Set the sale for each unique ID
        self.df['PoS'] = 1  # Set default value for Part of Sales for each UID

        for i in range(201500, 201553):
            self.df = self._set_PoS(i)
        for i in range(201601, 201653):
            self.df = self._set_PoS(i)
        for i in range(201701, 201754):
            self.df = self._set_PoS(i)
        for i in range(201801, 201826):
            self.df = self._set_PoS(i)

        return self.df

    def _set_PoS(self, i):
            tmp_we = i
            tmp_q = np.array(self.df[self.df['Week_ending'] == tmp_we]['quantity'])
            tmp_total = tmp_q.sum()
            self.df.loc[self.df['Week_ending'] == tmp_we, 'PoS'] = self.df['quantity']/tmp_total
            one = np.array(self.df.loc[self.df['Week_ending'] == tmp_we, 'PoS']).sum()
            if round(one) != 1:
                # print(one)
                raise ValueError
            return self.df

    def preview_data(self):
        print(self.df.tail(25))

    def get_uid_model(self, uid: int):
        main_df = self.df[self.df['Uid code'] == uid].copy()
        x = main_df[['Sale', 'ND']].values
        y = main_df['PoS'].values
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = LinearRegression().fit(X_train, Y_train)
        coeff_df = pd.DataFrame(model.coef_, main_df[['Sale', 'ND']].columns, columns=['Coefficient'])
        y_pred = model.predict(X_test)
        df_pred = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
        self.show_plt(df_pred)
        return model, coeff_df, df_pred

    def show_plt(self, df_pred):
        main_df = df_pred
        main_df.plot(kind='bar', figsize=(15, 10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()


dirpath = 'test_data1.xlsx'
lr = Tst_class()
lr.preprocessing_data(dirpath)
lr.preview_data()
for i in range(71):
    model, coef, df_pred = lr.get_uid_model(i)
    lr.show_plt(df_pred)
