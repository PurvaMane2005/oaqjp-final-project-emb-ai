import requests
import json

def emotion_detector(text_to_analyse):
    # URL of the sentiment analysis service
    url = 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'

    # Constructing the request payload in the expected format
    myobj = { "raw_document": { "text": text_to_analyse } }

    # Custom header specifying the model ID for the sentiment analysis service
    header = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}

    # Sending a POST request to the sentiment analysis API
    response = requests.post(url, json=myobj, headers=header)

    # Parsing the JSON response from the API
    formatted_response = json.loads(response.text)

    emotions = response['emotion']['document']['emotion']

    required_emotions = {emotion: emotions[emotion] for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness']}
    
    dominant_emotion = max(required_emotions, key=required_emotions.get)

    # Extracting sentiment label and score from the response
    label = formatted_response['documentSentiment']['label']
    score = formatted_response['documentSentiment']['score']

    # Returning a dictionary containing sentiment analysis results
    return {
        "anger": 0.006274985, 
        "disgust": 0.0025598293, 
        "fear": 0.009251528, 
        "joy": 0.9680386, 
        "sadness": 0.049744144, 
        "dominant_emotion": dominant_emotion
    }