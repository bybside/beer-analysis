import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

def main():
    beers_df = read_data("beers.csv", 0)
    breweries_df = read_data("breweries.csv", 0)
    merged_df = breweries_df.merge(beers_df,
                                   left_index=True,
                                   right_on="brewery_id",
                                   how="inner")

    # get average ABVs across a range of columns                         
    abv_by_brewery = mean_abv_by_key(merged_df, "name_x")
    abv_by_style = mean_abv_by_key(merged_df, "style")
    abv_by_state = mean_abv_by_key(merged_df, "state")
    natusch_ratios_by_ipa = natusch_ratio_by_ipa(beers_df)
    abv_ibu_corrcoef = correlation_abv_ibu(beers_df)
    print(f"ABV v IBU corrcoef: {abv_ibu_corrcoef}")

    # plot data
    plot_data(data=abv_by_brewery.head(20),
              plot_type="barh",
              plot_title="Mean ABV by Brewery")
    
    plot_data(data=abv_by_style.head(20),
              plot_type="barh",
              plot_title="Mean ABV by Style")

    plot_data(data=abv_by_state.head(20),
              plot_type="barh",
              plot_title="Mean ABV by State")
    
    plot_data(data=natusch_ratios_by_ipa.tail(20),
              plot_type="barh",
              plot_title="Natusch Ratio by IPA")

# generates a plot for a given DataFrame
# parameters:
#   - data: pandas Series object
#   - plot_type: kind of plot
#   - plot_title: title of plot
# returns:
#   - None
def plot_data(data, plot_type, plot_title):
    plot_img = f"images/{plot_title}.png"
    data.plot(kind=plot_type, title=plot_title)
    plt.savefig(plot_img, bbox_inches="tight")

# read in a csv file and return a dataframe
# parameters: 
#   - file: name of the file to be read in
#   - index (default=None): the column number to be used as the index
# returns:
#   - a pandas DataFrame object representing the read in csv data
def read_data(file, index=None):
    path = f"data/{file}"
    if index != None:
        return pd.read_csv(path, index_col=index)
    return pd.read_csv(path)

# get the average ABV for a given key in the dataset
# parameters:
#   - df: pandas DataFrame object
#   - key: column header in the DataFrame
# returns:
#   - a pandas Series object
def mean_abv_by_key(df, key):
    return df.groupby(key)["abv"].mean()

# what IPA has the highest abv but lowest ibu?
# parameters:
#   - df: pandas DataFrame object
# returns:
#   - a pandas Series object containing the Natusch ratios for all IPAs
def natusch_ratio_by_ipa(df):
    df.drop_duplicates(subset=["id"], keep="first", inplace=True) # drop rows with duplicate id
    ibu_df = df.dropna(subset=["ibu"]) # drop rows where ibu column is NaN
    natusch_ratios = (ibu_df["abv"]*100) / ibu_df["ibu"] # calculate natusch ratio
    ibu_df = ibu_df.assign(natusch_ratio=natusch_ratios) # assign natusch ratio values to existing ibu_df
    idx_series = ibu_df["style"].apply(is_ipa) # check each style for "IPA", return bool 
    ipa_df = ibu_df[idx_series] # filter out ibu_df values where style does not contain "IPA"
    return ipa_df.set_index("name")["natusch_ratio"].sort_values()

# utility function for natusch_ratio_by_ipa()
def is_ipa(style):
    return "IPA" in str(style)

def correlation_abv_ibu(df):
    # ibu_df = df[["ibu", "abv", "name"]].fillna(0) # fill NaN ibu and abv values
    ibu_df = df.dropna(subset=["ibu"])
    idx_series = ibu_df["style"].apply(is_ipa) # check each style for "IPA", return bool 
    ipa_df = ibu_df[idx_series] # filter out ibu_df values where style does not contain "IPA"
    r_value = np.corrcoef(ipa_df["ibu"].values, ipa_df["abv"].values)[0, 1]
    # plt.scatter(ibu_df["name"].values, ibu_df["ibu"].values)
    # plt.scatter(ibu_df["name"].values, (ibu_df["abv"]*10000).values)
    # plt.title(f"IBU v. ABV, r-value: {r_value}")
    # plt.savefig(f"images/IBU v ABV.png", bbox_inches="tight")
    return r_value

main()
