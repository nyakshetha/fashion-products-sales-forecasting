# Data dictionary

| Column | Type | Description |
|---|---|---|
| Customer_ID | text | Anonymous customer identifier (one fitting-room visit per customer). |
| Garment_ID | text | Identifier of a garment design; its colour/pattern/collar/sleeve/fit/price never change. |
| Size | int | Size tried on by the customer (36, 38, 39, 40, 42, 44, 46, 48). |
| Sleeve_Type | category | Half or Full. |
| Color | category | Beige, Black, Blue, Brown, Cream, Green, Grey, Khaki, Light Blue, Lilac. |
| Collar_Type | category | Button-down, Regular, Band. |
| Fit | category | Slim or Ultra Slim. |
| Pattern | category | Print, Patterned, Graphic, Checks, Dots, Stripes, Solid, Textured. |
| Price | int (INR) | Garment price, INR 934-3,999. |
| Trial_Count | int | Number of EARLIER fitting-room trials of this garment (current trial excluded). 0 = cold start. |
| Purchased | 0/1 | TARGET. 1 = bought after trying it on, 0 = did not. |
| Trial_Timestamp | datetime | Simulated trial time; defines the order used to compute Trial_Count. |
| Trend_Index | float 0-100 | Weekly search-interest index for the garment's pattern group and fit (simulated here; PyTrends in live mode). |
