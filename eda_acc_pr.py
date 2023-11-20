import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2, ttest_ind, mannwhitneyu, spearmanr
import warnings
warnings.filterwarnings("ignore")
from termcolor import colored

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


class DataAnalysis:
    
    def __init__(self, df, target_variable):
        self.df = df
        self.target_variable = target_variable
        
    def cat_columns(self):
        cat_cols = []
        for col in df.columns:
            if col != self.target_variable and (self.df[col].dtype == 'int64' or self.df[col].dtype == 'object' or self.df[col].dtype == 'category') and self.df[col].nunique() < 15:
                cat_cols.append(col)
        return cat_cols

    def num_columns(self):
        num_cols = []
        for col in df.columns:
            if col != self.target_variable and df[col].dtype != 'object' and df[col].nunique() > 20:
                num_cols.append(col)
        return num_cols
        
    def data_quality_table(self, outlier_threshold=3):
        # Calculate the counts
        null_counts = self.df.isnull().sum()
        duplicate_counts = self.df.duplicated().sum()

        # Calculate z-score for each numerical column
        exclude_types = ['object', 'category']  # Add any other data types you want to exclude

        # Select columns that are not of excluded types
        self.df.select_dtypes(exclude= exclude_types)
        z_scores = self.df.select_dtypes(exclude= exclude_types).apply(lambda x: np.abs((x - x.mean()) / x.std()))

        # Set the threshold for outlier detection (e.g., z-score > 3)
        outlier_counts = (z_scores > outlier_threshold).sum()

        # Create the table
        table_data = {
            'Null Values': null_counts,
            'Duplicate Values': duplicate_counts,
            'Outliers': outlier_counts
        }
        table_df = pd.DataFrame(table_data)

        return table_df 
    
    def bins(self, con_var, num_bins=6):
        # Calculate the bin edges
        min_val = self.df[con_var].min() - 1
        max_val = self.df[con_var].max()
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_col = f'{con_var}_bins'
        self.df[bin_col] = pd.cut(self.df[con_var], bins=bin_edges)

    def visual_1(self, var):
       
        self.df[self.target_variable] = self.df[self.target_variable].astype('category')

        # Assuming 'df' is your DataFrame and 'Age' is the continuous variable column
        # 'Exited' is the binary feature column

        # Plot the boxplot
        ax = sns.boxplot(data=self.df, x=var, y=self.target_variable)

        # Calculate and draw the median line
        overall_median = self.df[var].median()
        ax.axvline(overall_median, color='red', linestyle='--', label='Overall Median')

        # Add labels and title
        plt.xlabel(var)
        plt.ylabel(self.target_variable)
        plt.title('Boxplot: ' + var + ' by ' + self.target_variable)

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()

    def visual_2(self, var_bins):
        self.df[self.target_variable] = self.df[self.target_variable].astype('int')

        # Grouping data by age group and exited status
        grouped_data = self.df.groupby([var_bins, self.target_variable]).size().unstack()
        overall_rate = self.df[self.target_variable].mean()

        # Computing churn rate for each age group
        target_rate_col = f'{self.target_variable}_rate'
        grouped_data[target_rate_col] = grouped_data[1] / grouped_data.sum(axis=1)

        # Plotting the stacked count plot with churn rate
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plotting the stacked count bars

        bins = grouped_data.index.astype(str)
        val_0 = grouped_data[0]
        val_1 = grouped_data[1]
        ax1.bar(bins, val_0, label='0')
        ax1.bar(bins, val_1, bottom=val_0, label='1')
        ax1.set_xticklabels(bins, rotation=90)
        # Plotting the churn rate line
        line_plot = grouped_data[target_rate_col].plot(marker='o', ax=ax2, color='red', zorder=10)

        # Adding a line for overall churn rate on the secondary axis
        ax2.axhline(overall_rate, color='blue', linestyle='--', label='Overall Rate')

        ax1.set_ylabel('Count')
        ax2.set_ylabel(target_rate_col)

        # Adding data labels to the stacked count bars
        for p in ax1.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            label_text = f'{int(height)}'
            ax1.text(x + width / 2, y + height / 2, label_text, ha='center', va='center')

        # Adding data labels to the churn rate line
        for i, rate in enumerate(grouped_data[target_rate_col]):
            ax2.text(i, rate, f'{rate:.2%}', ha='center', va='bottom')

        plt.title('Comparison: ' + var_bins + ' by ' + self.target_variable)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()
        
    def perform_ttest(self, con_var):
        
        target_0 = self.df[self.df[self.target_variable] == 0][con_var]
        target_1 = self.df[self.df[self.target_variable] == 1][con_var]

        # Perform t-test
        t_stat, p_value = ttest_ind(target_0, target_1)

        return p_value

    def perform_wmtest(self, con_var):
        
        target_0 = self.df[self.df[self.target_variable] == 0][con_var]
        target_1 = self.df[self.df[self.target_variable] == 1][con_var]

        statistic, p_value = mannwhitneyu(target_0, target_1)

        return p_value

    # plot of categorical variables with respect to target variable
    def plot_categorical_cols(self, col):
        grouped_data = df.groupby([col, self.target_variable]).size().unstack()
            
        grouped_data['% outcome'] = grouped_data[1] / grouped_data.sum(axis=1)
        overall_target = df[self.target_variable].mean()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
            
        # Plot the stacked bar chart
        grouped_data.plot(kind='bar', stacked=True, ax= ax1)
        line_plot = grouped_data['% outcome'].plot(marker='o', ax=ax2, color='green', zorder=10)
        ax2.axhline(overall_target, color='blue', linestyle='--', label='Overall target %')
            
        ax1.set_ylabel('Count')
        ax2.set_ylabel('Overall %')


        for i, rate in enumerate(grouped_data['% outcome']):
            ax2.text(i, rate, f'{rate:.2%}', ha='center', va='bottom')
    
        plt.title(f"Categorical relationship between {col} and {self.target_variable}")
        plt.show()
        
    #find sub-cateegories with higher percentage of outcomes than overall outcome
    def high_subcategory(self, col):
        
        higher_subcategory = self.df.groupby(col)[self.target_variable].mean().idxmax()
        percent = f"{np.round(self.df.groupby(col)[self.target_variable].mean().max() *100,2)}%"
        return str(higher_subcategory) + " : " + percent 
    
    def low_subcategory(self,col):
        lower_subcategory = self.df.groupby(col)[self.target_variable].mean().idxmin()
        percent = f"{np.round(self.df.groupby(col)[self.target_variable].mean().min() *100,2)}%"
        return str(lower_subcategory) + " : " + percent 
        
        # if higher_outcome_categories:
        #     print("Sub-categories with higher percentage of outcomes than overall outcome: ")
        #     for category in higher_outcome_categories:
        #         print(category)
                
        # else:
        #     print("Sub-categories with lower percentage of outcomes than overall outcome: ")
        #     for category in lower_outcome_categories:
        #         print(category)

    
    # chi-square test to test the association between categorical variable and target variable
    def categorical_target_relation(self, col):
        contingency_table = pd.crosstab(df[col], df[self.target_variable])
        chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
        
                    
        # print(f"chi-Square test p- value: {p_val}")
        # if p_val <= 0.001:
        #     self.columns_with_very_strong_association.append(col)
        #     print(f"there is an evidence of very strong association between {col} and {self.target_variable}. It means that as {col} changes, the target variable {self.target_variable} also changes very strongly.")
        
        # elif p_val <= 0.01:
        #     self.columns_with_strong_association.append(col)
        #     print(f"there is an evidence of strong association between {col} and {self.target_variable}. It means that as {col} changes, the target variable {self.target_variable} also changes strongly.")
            
        # elif p_val <= 0.05:
        #     self.columns_with_mild_association.append(col)
        #     print(f"there is an evidence of mild association between {col} and {self.target_variable}. It means that as {col} changes, there is a mild effect on {self.target_variable}")
        
        
        # else:            
        #     self.columns_with_relatively_no_association.append(col)
        #     print(f"there is no evidence of association between {col} and {self.target_variable}. It means that as {col} changes, we cannot say conclusively about the nature of change in target variable {self.target_variable}")

        return p_val
            
    def create_table(self, var_bins):
        temp = self.df.groupby([var_bins, self.target_variable]).size().unstack()
        temp['total pop'] = temp[0] + temp[1]
        temp['Percentage'] = temp['total pop'] / temp['total pop'].sum()
        temp_1 = temp[temp['Percentage'] > 0.05]
        temp_1['Rate']=(temp_1[1]/temp_1['total pop'])*100
        temp_1.reset_index(inplace=True)
        return temp_1

    def get_variable_trends(self):
                        
        trends_table = pd.DataFrame()
        numerical_columns = self.num_columns()

        for var in numerical_columns:
            if var != self.target_variable:
                var_bins = f'{var}_bins'
                self.bins(var)
                temp_table = self.create_table(var_bins)
                temp_table['moving_avg'] = temp_table['Rate'].rolling(window=2).mean()
                temp_table['var_bins_en']=np.arange(1,len(temp_table)+1,1)
                spearman_corr, _ = spearmanr(temp_table[1:]['var_bins_en'], temp_table[1:]['Rate'])
                
                highest_rate = str(temp_table[temp_table['Rate'] == temp_table['Rate'].max()][var_bins].values[0]) +' : ' +str(np.round(temp_table['Rate'].max(),2))+'%'
                lowest_rate = str(temp_table[temp_table['Rate'] == temp_table['Rate'].min()][var_bins].values[0]) +' : ' +str(np.round(temp_table['Rate'].min(),2))+'%'
                avg_overall_rate = self.df[self.target_variable].mean()*100
                
                if self.perform_ttest(var)>0.05 and self.perform_wmtest(var)>0.05 and (avg_overall_rate-temp_table['Rate'].min())<5 and(temp_table['Rate'].max()-avg_overall_rate)<5:
                    insights=f"There is no relation of {var} with {self.target_variable}."
                elif self.perform_ttest(var)<0.05 and self.perform_wmtest(var)<0.05 and (avg_overall_rate-temp_table['Rate'].min())<5 and(temp_table['Rate'].max()-avg_overall_rate)<5:
                    insights=f"There is no conclusive relation of {var} with {self.target_variable}, but there lies a must see observation' refer bin bar plot"
                else:
                    insights=f'There is association between {var} and {self.target_variable}, refer barplot for more details'
                
                if spearman_corr > 0.5 and spearman_corr < 0.7:
                    trend = 'Positive'
                elif spearman_corr > 0.7:
                    trend = 'Strong Positive'
                elif spearman_corr < -0.5 and spearman_corr > -0.7:
                    trend = 'Negative'
                elif spearman_corr < -0.7:
                    trend = 'Strong Negative'
                else:
                    trend = 'None'

                
                trends_table = trends_table.append({'Variable': var,  'Highest Rate': highest_rate,
                                                    'Lowest Rate': lowest_rate, 'Average Overall_Rate':avg_overall_rate,'Trend':trend,'Observation':insights},ignore_index=True)
                

        categorical_columns = self.cat_columns()
        #chi-square test
        for col in categorical_columns:
            if "bin" not in col:
                p_value = self.categorical_target_relation(col)
                if p_value <= 0.001:
                    strength = f"very strong association between {col} and {self.target_variable}"

                elif p_value <= 0.01:
                    strength = f'strong association between {col} and {self.target_variable}'

                elif p_value <= 0.05:
                    strength = f'mild association between {col} and {self.target_variable}'

                else:
                    strength = f"no association between {col} and {self.target_variable}"

                trends_table = trends_table.append({'Variable': col,  'Highest Rate': self.high_subcategory(col),
                                                        'Lowest Rate': self.low_subcategory(col), 'Average Overall_Rate':avg_overall_rate,'Trend':"--",'Observation':strength},ignore_index=True)    



                
        def color_rows(row):
            if ('no relation' in row['Observation']) or ('no association' in row['Observation']):
                return ['background-color: #FFC0CB'] * len(row)
            elif 'no conclusive relation' in row['Observation'] or ('mild association' in row['Observation']):
                return ['background-color:#FFFFE0'] * len(row)
            elif 'very strong association' in row['Observation']:
                return ['background-color: #90EE90'] * len(row)
            else:
                return ['background-color: #90EE90'] * len(row)

        trends_table_styled = trends_table.style.apply(color_rows, axis=1)
        
        print(colored('-----------------------------------------------UNIVARIATE VS Y Summary Table-----------------------------------------------', attrs=['bold']))

        return trends_table_styled
    
    
    def analyze_data(self):
        from termcolor import colored
        import warnings
        warnings.filterwarnings("ignore")
        
        numerical_columns = self.num_columns()
        categorical_columns = self.cat_columns()

        for var in numerical_columns:
            if var!=self.target_variable:
                print(colored(f'----------------------------------------------------{var}--------------------------------------------------------', attrs=['bold']))
                self.visual_1(var)
                var_bins = f'{var}_bins'
                self.bins(var)
                self.visual_2(var_bins)
                print(colored('----------------------------------------------------------------------------------------------------------------------', attrs=['bold']))

        for var in categorical_columns:
            print(colored(f'----------------------------------------------------{var}--------------------------------------------------------', attrs=['bold']))
            self.plot_categorical_cols(var)
            self.categorical_target_relation(var)





# file_path = r"C:\Users\govin\Pangea_training\bank.csv"

df = pd.read_csv('Bank_Churn_data.csv')     
analysis = DataAnalysis(df, 'Exited')
analysis.data_quality_table()
analysis.analyze_data()
analysis.get_variable_trends()