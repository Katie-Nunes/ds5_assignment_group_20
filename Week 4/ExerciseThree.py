def apply_sentiment_analysis(df, tweet_col="Tweet", language_col="Language"):
    """
    Apply language-specific sentiment analysis to each tweet in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets and their detected languages
        tweet_col (str): Name of the column containing tweet text
        language_col (str): Name of the column containing detected languages
        
    Returns:
        pd.DataFrame: DataFrame with added 'sentiment' column
        
    hala
    """
    sentiments = []
    
    for idx, row in df.iterrows():
        tweet = row[tweet_col]
        language = row[language_col]
        
        try:
            # Choose sentiment analysis function based on language
            if language == "en":
                sentiment = analyze_sentiment_english(tweet)
            else:
                sentiment = analyze_sentiment_other(tweet)
        except Exception as e:
            # Handle any errors during sentiment analysis
            print(f"Error analyzing sentiment for tweet {idx}: {e}")
            sentiment = "unknown"
        
        sentiments.append(sentiment)
    
    # Add sentiment column to DataFrame
    result_df = df.copy()
    result_df["sentiment"] = sentiments
    
    return result_df