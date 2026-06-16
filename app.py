import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

print("Files in directory:", os.listdir("."))

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]
df.columns = ['label', 'message']
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, _, y_train, _ = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tf = tfidf.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tf, y_train)

print("Model trained successfully")

def predict(message):
    vec = tfidf.transform([message])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "🚨 SPAM" if result == 1 else "✅ HAM (not spam)"
    return f"{label} — {max(prob):.1%} confidence"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Type a message here..."),
    outputs=gr.Textbox(label="Result"),
    title="📧 Spam Email Detector",
    description="Trained on the UCI SMS Spam Collection dataset using TF-IDF + Naive Bayes."
)

demo.launch()
