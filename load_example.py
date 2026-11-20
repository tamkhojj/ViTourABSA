import pandas as pd

ASPECTS = ["room", "service", "location", "price", "food_and_beverage", "amenities",
           "cleanliness", "transportation", "policy", "others"]

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/{split}.csv")
        print(split, df.shape)
    print(df[["review"] + ASPECTS].head())
