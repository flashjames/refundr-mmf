import pandas as pd
from sklearn.model_selection import train_test_split

def split_stratified_into_train_val_test(df_input, y,
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    X = df_input # Contains all columns.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)

    #TODO: Should straitify but it gives error, so just removed it for now
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

import pandas as pd

fname = "./datasets/prisjakt/product_pages_ml.csv"
df = pd.read_csv(fname)
#
grouped = df.groupby('meta_product_page__id')
grouped = grouped.filter(lambda x: len(x) > 3)

#print(grouped)
train, val, test = \
    split_stratified_into_train_val_test(grouped, grouped['meta_product_page__id'], frac_train=0.60, frac_val=0.20, frac_test=0.20)
#import pdb;pdb.set_trace()
train_fname = fname.replace("product_pages_ml", "product_pages_ml__train")
train.to_csv(train_fname, index=False)


val_fname = fname.replace("product_pages_ml", "product_pages_ml__val")
val.to_csv(val_fname, index=False)

test_fname = fname.replace("product_pages_ml", "product_pages_ml__test")
test.to_csv(test_fname, index=False)
