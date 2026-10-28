"""A small house-price prediction example using demo data."""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Demo data only; replace this with real local housing data for meaningful estimates.
FEATURES = [
    [area, bedrooms, bathrooms, age]
    for area, bedrooms, bathrooms, age in [
        (900, 2, 1, 26),
        (1100, 2, 1, 15),
        (1250, 3, 2, 20),
        (1400, 3, 2, 12),
        (1550, 3, 2, 8),
        (1700, 3, 2, 15),
        (1850, 4, 2, 10),
        (2000, 4, 3, 5),
        (2200, 4, 3, 12),
        (2400, 4, 3, 7),
        (2600, 5, 3, 10),
        (2800, 5, 4, 4),
    ]
]
PRICES = [
    180_000,
    210_000,
    245_000,
    285_000,
    315_000,
    320_000,
    365_000,
    420_000,
    415_000,
    475_000,
    510_000,
    590_000,
]


features_train, features_test, prices_train, prices_test = train_test_split(
    FEATURES, PRICES, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(features_train, prices_train)

test_predictions = model.predict(features_test)
print(f"Mean absolute error: ${mean_absolute_error(prices_test, test_predictions):,.0f}")

# Feature order: square feet, bedrooms, bathrooms, and age in years.
example_home = [[1800, 3, 2, 10]]
estimated_price = model.predict(example_home)[0]
print(f"Estimated demo price: ${estimated_price:,.0f}")
