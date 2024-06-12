import pandas as pd

# Load the dataset
file_path = '/mnt/data/HousePrice.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head(), data.info()



# Convert the 'Area' column to numeric
data['Area'] = pd.to_numeric(data['Area'], errors='coerce')

# Drop rows with missing values in 'Address' column (or we could handle it differently)
data = data.dropna(subset=['Address'])

# Drop the 'Price(USD)' column
data = data.drop(columns=['Price(USD)'])

# Convert boolean columns to integers
data['Parking'] = data['Parking'].astype(int)
data['Warehouse'] = data['Warehouse'].astype(int)
data['Elevator'] = data['Elevator'].astype(int)

# Drop the 'Address' column as it's categorical and may not be useful for our regression
data = data.drop(columns=['Address'])

# Check the cleaned data
data.head(), data.info()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Drop rows with missing values in 'Area' column
data = data.dropna(subset=['Area'])

# Split the data into features (X) and target (y)
X = data.drop(columns=['Price'])
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, mse, rmse, r2