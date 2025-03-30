
import pandas as pd

# Example DataFrame: assume df has 'PrimaryUse' and 'AreaHa' columns
def classify_by_name(row):
    name = row['PrimaryUse']
    # Step 1: Classification by name
    if name in ["Country park"]:
        return "1 Regional Parks"
    elif name in ["Park", "Nature reserve"]:
        return "2 Metropolitan Parks"
    elif name in ["Recreation ground", "Playing fields"]:
        return "3 District Parks"
    elif name in ["Amenity green space", "Formal garden", "Village green", "Public woodland"]:
        return "4 Local Parks and Open Spaces"
    elif name in ["Play space", "Community garden", "Allotments", "Youth area"]:
        return "5 Small Open Spaces"
    elif name in ["Road island/verge", "Civic/market square"]:
        return "6 Pocket Parks"
    elif name in ["Canal", "River", "Walking/cycling route"]:
        return "7 Linear Open Spaces"
    else:
        return "Unclassified"
    
# --- Step 2: Area-Based Classification (Column D) ---
def classify_by_area(row):
    area = row['AreaHa']
    if pd.isna(area):
        return "Unclassified"
    elif area > 400:
        return "1 Regional Parks"
    elif area > 60:
        return "2 Metropolitan Parks"
    elif area > 20:
        return "3 District Parks"
    elif area > 2:
        return "4 Local Parks and Open Spaces"
    elif area > 0.4:
        return "5 Small Open Spaces"
    elif area > 0:
        return "6 Pocket Parks"
    else:
        return "Unclassified"

# --- Step 3: Compare and finalize (Column E) ---
def final_classification(row):
    if row["TextClass"] == row["AreaClass"]:
        return row["TextClass"]
    else:
        return row["TextClass"]  # Prefer text-based if disagreement
    

# # --- Apply the functions ---
# df["TextClass"] = df["LandCover"].apply(classify_by_name)      # Column B
# df["AreaClass"] = df["AreaHa"].apply(classify_by_area)         # Column D
# df["FinalClass"] = df.apply(final_classification, axis=1)      # Column E


