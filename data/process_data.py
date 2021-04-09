import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge datasets
    df = messages.merge(categories, on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(";", expand=True)
    categories.columns = ["category_" + str(i) for i in categories.columns]
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: re.sub("^([a-z_A-Z]+)\-\d$","\g<1>",x))
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: re.sub("^([a-z_A-Z]+)\-(\d)$","\g<2>",x)) 

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column]) 
    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)
    
    df2 = pd.concat([df,categories],axis=1, join="inner")
    
    return df2
    


def clean_data(df):
           
        
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # recode any of the values greater than 1 to 1 (should be only from the "related" 
    # column - 192 instances)
    # need to also cast as int as well, other wise the function changes a lot more
    # values than the expect 192
    df.iloc[:,4:] = df.iloc[:,4:].applymap(lambda x: 1 if int(x) > 1 else int(x))
    
    return df
    
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages_categorized', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()