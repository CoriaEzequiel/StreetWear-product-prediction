import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


df = pd.read_csv("saleOrders.csv")
#Normalizo
df["color"] = df["color"].str.strip().str.lower()
colores_map = {
    "negra": "negro",
    "blanca": "blanco",
    "azules": "azul",
    "verdes": "verde",
    "rosas": "rosa",
    "amarillas": "amarillo",
    "grises": "gris",
    "rojas": "rojo"
}
df["color"] = df["color"].replace(colores_map)

df["talle"] = df["talle"].str.strip().str.upper()

#  Features y target
X = df[["marca", "categoria", "talle", "color", "precio"]]
y = df["cantidad_vendidas"]

#  Columnas categóricas y numéricas
cat_features = ["marca", "categoria", "talle", "color"]
num_features = ["precio"]

#  Preprocesamiento: "OneHot" para categorías, "passthrough" para numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ]
)

#  Pipeline con modelo "Random Forest"
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

#  Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Entreno
model.fit(X_train, y_train)

#  Evalúo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Resultados del modelo:")
print(f" - MAE: {mae:.2f}")
print(f" - R2 (coeficiente de determinación): {r2:.2f}")

# 9. Guardar modelo entrenado
joblib.dump(model, "modelo_producto_estrella.pkl")

# predicción de prueba
nuevo_item = pd.DataFrame([{
    "marca": "Nike",
    "categoria": "remera_dri-fit",
    "talle": "M",
    "color": "Blanca",
    "precio": 50
}])

remera_nike_x_duki = pd.DataFrame([{
    "marca": "Nike",
    "categoria": "remera_dri-fita",
    "talle": "XL",
    "color": "Negra",
    "precio": 65
}])
#agrego nuevos productos
nuevos_items = pd.DataFrame([
    {"marca": "Nike", "categoria": "remera_dri-fit", "talle": "M", "color": "Blanca", "precio": 50},
    {"marca": "Nike", "categoria": "remera_dri-fita", "talle": "XL", "color": "Negra", "precio": 65},
    {"marca": "Adidas", "categoria": "remera_climalite", "talle": "L", "color": "Negro", "precio": 58}
])

predicciones = model.predict(nuevos_items)
for item, pred in zip(nuevos_items.to_dict(orient="records"), predicciones):
    print(f"Predicción para {item['marca']} {item['categoria']} {item['talle']} {item['color']}: {pred:.0f} unidades")