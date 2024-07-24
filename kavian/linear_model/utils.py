import pandas as pd

def create_entry(col_name, row):
    row = str(round(row, 3))

    return f"{col_name} :", row

# TODO: It's probably better to include this in the class
def basic_model_info(X, y):
    date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')
    time = pd.Timestamp.now().time().strftime('%H:%M:%S')
    num_observ, num_features = str(len(X)), str(len(X.columns))

    info = [("Date: ", date),
            ("Time: ", time),
            ("No. Observations: ", num_observ),
            ("No. Features: ", num_features),
            ("Dep. Variable: ", y.name),]

    return info





