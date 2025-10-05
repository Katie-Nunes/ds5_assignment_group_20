import pandas as pd
from langdetect import detect, DetectorFactory
# for the sentiment functions
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer


# Make language detection deterministic
DetectorFactory.seed = 0

def detect_tweet_languages(tweet: str) -> str:
    try:
        return detect(tweet)
    except:
        return "Unknown"

def _simple_clean(text: str) -> str:
    """
    Tiny helper to reduce obvious noise:
    - ensure it's a string
    - remove '@' and '#' symbols (keep the word)
    - crudely strip url bits ('http', 'www.') to avoid bias
    - collapse repeated spaces
    """
    if not isinstance(text, str):
        return ""
    t = text.replace("@", "").replace("#", "")
    t = t.replace("http", " ").replace("www.", " ")
    t = " ".join(t.split())
    return t

def analyze_sentiment_english(tweet: str) -> str:
    """
    ENGLISH sentiment (TextBlob).
    Spec rule:
      positive if polarity > 0
      negative if polarity < 0
      neutral  otherwise
    """
    text = _simple_clean(tweet)
    polarity = float(TextBlob(text).sentiment.polarity)
    if polarity > 0:
        return "positive"
    if polarity < 0:
        return "negative"
    return "neutral"

# reuse a single VADER instance (faster)
_vader = None
def _get_vader():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader

def analyze_sentiment_other(tweet: str) -> str:
    """
    NON-English (fallback) sentiment (NLTK VADER).
    Spec rule:
      positive if compound >= 0.05
      negative if compound <= -0.05
      neutral  otherwise
    """
    text = _simple_clean(tweet)
    compound = float(_get_vader().polarity_scores(text)["compound"])
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"

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

def analyze_tweet_sentiments(df, tweet_col="Tweet"):
    """
    Main function to perform complete tweet analysis (language detection + sentiment analysis).
    
    Args:
        df (pd.DataFrame): DataFrame containing tweet data
        tweet_col (str): Name of the column containing tweet text
        
    Returns:
        pd.DataFrame: DataFrame with added 'language' and 'sentiment' columns
        
    hala
    """
    print("=== TWEET SENTIMENT ANALYSIS PIPELINE ===")
    
    # Step 
    print("1. Detecting languages...")
    df_with_language = detect_tweet_languages(df, tweet_col)
    
    # Show language distribution
    language_counts = df_with_language['Language'].value_counts()
    print(f"   Language distribution: {dict(language_counts.head())}")
    
    # Step 2
    print("2. Analyzing sentiments...")
    df_with_sentiment = apply_sentiment_analysis(df_with_language)
    
    # Show sentiment distribution
    sentiment_counts = df_with_sentiment['sentiment'].value_counts()
    print(f"   Sentiment distribution: {dict(sentiment_counts)}")
    
    print("✓ Analysis completed successfully!")
    
    return df_with_sentiment

def main():
    """
    Main function to run the complete tweet sentiment analysis.
    """
    try:
        
        df = pd.read_excel("tweets-1.xlsx")
        df['language'] = df['Tweet'].apply(detect_tweet_languages)
        print("tt")
        
        results = analyze_tweet_sentiments(df)
        
        print("\nFinal Results:")
        print(results[['Tweet', 'Language', 'sentiment']].head())
        
        
        
        return results
        
    except FileNotFoundError:
        print("Error: tweets.xlsx file not found. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None

if __name__ == "__main__":
    main()