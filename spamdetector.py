import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Prepare training data
emails = [
    "Win a free iPhone now",                   # spam
    "Limited time offer! Claim your prize",    # spam
    "Let's have lunch tomorrow",               # not spam
    "Reminder: your appointment at 3pm",       # not spam
    "Congratulations! You have won $1000",     # spam
    "Can you send me the report?",             # not spam
    "Cheap meds online, no prescription",      # spam
    "Your Amazon order has shipped",           # not spam
    "You have been selected for a free gift",  # spam
    "Meeting scheduled with HR at 2PM",        # not spam
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# 2. Train the spam classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

# 3. Set up Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "🚀 Spam Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"error": "No email text provided"}), 400

    email_text = data['email']
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]

    return jsonify({
        "email": email_text,
        "prediction": "SPAM" if prediction == 1 else "NOT SPAM"
    })

if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host="0.0.0.0", port=port, debug=True)
