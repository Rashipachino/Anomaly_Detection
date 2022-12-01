import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score
import gradio as gr


print("loading data")
DATA_PATH = "conn_attack.csv"
df = pd.read_csv(DATA_PATH, header=None, names=["record ID","duration_", "src_bytes","dst_bytes"])
print("done loading data")

data = df.drop(columns=["record ID"], axis=1).copy()

print("training model")
model = IsolationForest(contamination=float(0.004), n_estimators=1000, max_samples=205, max_features=3)
model.fit(data.values)
print("done training model")

print("testing model")
df["iforest"] = pd.Series(model.predict(data.values))
df["iforest"] = df["iforest"].map({1: 0, -1: 1})
print("done testing model")

PATH_TO_LABELS = 'conn_attack_anomaly_labels.csv'
data_labels = pd.read_csv(PATH_TO_LABELS, header=None, names=["record ID","label"])

print("Confusion Matirx:")
confusion_matrix(df["iforest"], data_labels["label"], labels=[0,1])

accuracy = accuracy_score(data_labels["label"], df["iforest"])
print("accuracy score: {0:.2f}%".format(accuracy*100))

recall = recall_score(data_labels["label"], df["iforest"])
print("recall score: {0:.2f}%".format(recall*100))

def annomaly(duration_, src_bytes, dst_bytes):
    x = np.array([duration_, src_bytes, dst_bytes])
    prediction = model.predict(x.reshape(1, -1))
    return prediction

outputs = gr.outputs.Textbox()

app = gr.Interface(fn=annomaly, inputs=['number','number','number'], outputs=outputs,description="This is a cyber annomaly model")

app.launch(share=True, server_name="0.0.0.0", server_port=8080)