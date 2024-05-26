import pandas as pd
import sqlite3
from tensorflow.keras.models import load_model
import joblib

def preprocess_data(data):
    '''Strip whitespace from 'status' and 'source' columns'''
    data['status'] = data['status'].str.strip()
    data['source'] = data['source'].str.strip()

    '''Convert 'status' and 'source' to binary values'''
    status_mapping = {'Closed': 1, 'Open': 0}
    source_mapping = {'Phone': 1, 'Non-phone': 0}

    data['status'] = data['status'].map(status_mapping)
    data['source'] = data['source'].map(source_mapping)

    '''Selecting the relevant columns for testing the sample'''
    data = data[['diff_days_int', 'source']]
    
    return data

def load_and_predict(input_file, db_file='classification_results.db'):
    '''Load the model'''
    model = load_model('models/best_ann_model.h5')

    '''Load the scaler'''
    scaler = joblib.load('models/scaler.pkl')

    '''Read the input file'''
    data = pd.read_csv(input_file)
    
    '''Generate IDs 'id' column because is missing'''
    if 'id' not in data.columns:
        data['id'] = range(1, len(data) + 1)
    
    ids = data['id']
    features = preprocess_data(data)

    '''Normalize/scale the features'''
    features_scaled = scaler.transform(features)

    '''The predictions'''
    predictions_prob = model.predict(features_scaled)
    predictions = (predictions_prob > 0.5).astype(int)

    '''The results for storage'''
    results = pd.DataFrame({'id': ids, 'class': predictions.flatten()})

    '''Saving and sending the outputs to SQLite database'''
    conn = sqlite3.connect(db_file)
    results.to_sql('requests_predictions', conn, if_exists='replace', index=False)
    conn.close()

    print(f'Results stored in {db_file}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Classify samples and store results in SQLite.')
    parser.add_argument('input_file', type=str, help='Path to the input file with samples to classify.')
    parser.add_argument('--db_file', type=str, default='classification_request.db', help='SQLite database file to store results.')
    args = parser.parse_args()

    load_and_predict(args.input_file, args.db_file)
