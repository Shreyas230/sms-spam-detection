{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a88c7593-6865-49fc-8ef7-1bdf70ebc43d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import gradio as gr\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "# Train model on load\n",
    "df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]\n",
    "df.columns = ['label', 'message']\n",
    "df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})\n",
    "X_train, _, y_train, _ = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)\n",
    "tfidf = TfidfVectorizer(stop_words='english', max_features=3000)\n",
    "X_train_tf = tfidf.fit_transform(X_train)\n",
    "model = MultinomialNB()\n",
    "model.fit(X_train_tf, y_train)\n",
    "\n",
    "def predict(message):\n",
    "    vec = tfidf.transform([message])\n",
    "    result = model.predict(vec)[0]\n",
    "    prob = model.predict_proba(vec)[0]\n",
    "    label = \"🚨 SPAM\" if result == 1 else \"✅ HAM (not spam)\"\n",
    "    return f\"{label} — {max(prob):.1%} confidence\"\n",
    "\n",
    "demo = gr.Interface(\n",
    "    fn=predict,\n",
    "    inputs=gr.Textbox(lines=4, placeholder=\"Type a message here...\"),\n",
    "    outputs=gr.Textbox(label=\"Result\"),\n",
    "    title=\"📧 Spam Email Detector\",\n",
    "    description=\"Trained on the UCI SMS Spam Collection dataset using TF-IDF + Naive Bayes.\"\n",
    ")\n",
    "\n",
    "demo.launch()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
