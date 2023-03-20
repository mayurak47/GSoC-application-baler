
import uproot
import pandas as pd

def pre_processing(input_path,output_path):

    df = pd.read_csv(input_path)
    df = df.drop(df.columns[0:2],axis=1).dropna()
    df.to_pickle(output_path)

def type_clearing(tt_tree):
    type_names = tt_tree.typenames()
    column_type = []
    column_names = []

    # In order to remove non integers or -floats in the TTree,
    # we separate the values and keys
    for keys in type_names:
        column_type.append(type_names[keys])
        column_names.append(keys)

    # Checks each value of the typename values to see if it isn't an int or
    # float, and then removes it
    for i in range(len(column_type)):
        if column_type[i] != "float[]" and column_type[i] != "int32_t[]":
            # print('Index ',i,' was of type ',Typename_list_values[i],'            # and was deleted from the file')
            del column_names[i]

    # Returns list of column names to use in load_data function
    return column_names
