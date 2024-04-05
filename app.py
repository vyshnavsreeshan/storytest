import os
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import openai
from openai.error import RateLimitError

openai.api_key = os.environ.get('API_KEY')
nltk.download('vader_lexicon')
nltk.download('punkt')

emotions = {
    'neutral': {'rate': 0.5, 'volume': 1.0, 'pitch': 1.0},
    'happy': {'rate': 0.5, 'volume': 0.6, 'pitch': 1.2},
    'sad': {'rate': 0.4, 'volume': 0.4, 'pitch': 0.8},
    'angry': {'rate': 0.6, 'volume': 0.75, 'pitch': 1.5},
    'excited': {'rate': 0.6, 'volume': 0.6, 'pitch': 1.2},
    'fearful': {'rate': 0.5, 'volume': 0.5, 'pitch': 1.0},
    'calm': {'rate': 0.45, 'volume': 0.4, 'pitch': 0.8},
    'surprised': {'rate': 0.6, 'volume': 0.8, 'pitch': 1.0},
    'tender': {'rate': 0.4, 'volume': 0.45, 'pitch': 0.9}
}

def map_to_emotion(sentiment_score):
    compound_score = sentiment_score['compound']
    if compound_score >= 0.2:
        return 'excited'
    elif compound_score <= -0.2:
        return 'fearful'
    elif compound_score >= 0.1:
        return 'happy'
    elif compound_score <= -0.1:
        return 'sad'
    elif -0.1 < compound_score < 0.1:
        return 'neutral'
    elif compound_score >= 0.05:
        return 'surprised'
    elif compound_score <= -0.05:
        return 'angry'
    else:
        return 'calm'


def generate_story(selected_likes):
    prompt = "Create a prompt for generating a short story with themes including: " + ", ".join(selected_likes)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000
    )

    story_prompt = "You are a short story writer don't exceed 200 words " + response.choices[0].text.strip()
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=story_prompt,
        max_tokens=1000
    )

    story = response.choices[0].text.strip()
    sentences = nltk.sent_tokenize(story)

    sid = SentimentIntensityAnalyzer()
    sentences_data = []

    for sentence in sentences:
        sentiment_score = sid.polarity_scores(sentence)
        emotion = map_to_emotion(sentiment_score)
        rate = emotions[emotion]['rate']
        volume = emotions[emotion]['volume']
        pitch = emotions[emotion]['pitch']
        sentences_data.append({
            'sentence': sentence,
            'emotion': emotion,
            'rate': rate,
            'volume': volume,
            'pitch': pitch
        })

    return sentences_data

app = Flask(__name__)

@app.route('/generate_story', methods=['POST'])
def generate_story_endpoint():
    request_json = request.get_json()
    selected_likes = request_json['selected_likes']
    try:
        result = generate_story(selected_likes)
        return jsonify({'result': result})
    except RateLimitError as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 429  # Return HTTP status code 429 for rate limit exceeded

if __name__ == "__main__":
    app.run(debug=True)
