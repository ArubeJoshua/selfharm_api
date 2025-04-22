# app/models/suicidal_thought_classifier.py
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from collections import defaultdict
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger("suicidal-thought-classifier")

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class SuicidalThoughtClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the classifier with the trained model.
        
        Args:
            model_path (str, optional): Path to the trained model pickle file.
                If None, uses the default path.
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent / "static" / "mental_health_advanced_ensemble.pkl"
        else:
            model_path = Path(model_path)
            
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        # Define class mapping for the 5 categories in the dataset
        self.class_mapping = {
            0: "Attempt",
            1: "Behavior",
            2: "Ideation", 
            3: "Indicator",
            4: "Supportive"
        }
    
    def clean_text(self, text):
        """
        Performs text cleaning and preprocessing
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text):
        """
        Tokenizes, removes stopwords and lemmatizes text
        """
        tokens = word_tokenize(text)
        
        # Setup for lemmatization
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        
        # Keep negation words as they are important for sentiment
        negation_words = {'no', 'not', 'none', 'never', 'nothing'}
        stop_words = set(stopwords.words('english')) - negation_words
        
        final_words = []
        word_lemmatizer = WordNetLemmatizer()
        
        # Tag words and lemmatize based on POS
        for word, tag in pos_tag(tokens):
            if word not in stop_words and word.isalpha():
                word_final = word_lemmatizer.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        
        return ' '.join(final_words)
    
    def preprocess_text(self, text):
        """
        Apply full preprocessing pipeline to input text
        """
        cleaned = self.clean_text(text)
        lemmatized = self.lemmatize_text(cleaned)
        return lemmatized
    
    def predict(self, text):
        """
        Predicts mental health risk category in the provided text.
        
        Args:
            text (str): The input text to analyze.
            
        Returns:
            dict: Prediction results with risk level and assessment.
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Use the trained model
        try:
            # Get numeric prediction
            prediction_idx = self.model.predict([processed_text])[0]
            
            # Convert to category name
            category = self.class_mapping.get(prediction_idx, "Unknown")
            
            # Get probability scores
            try:
                probabilities = self.model.predict_proba([processed_text])[0]
                confidence = float(probabilities[prediction_idx])
            except:
                confidence = 0.8  # Default confidence if predict_proba is not available
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
        
        # Perform self-harm risk assessment
        self_harm_assessment = self.assess_self_harm_risk(text)
        
        # Determine if content is concerning (anything not supportive)
        concerning_content = category != "Supportive"
        
        # Determine overall risk level
        risk_level = self.determine_risk_level(confidence, self_harm_assessment, category)
        
        # Generate recommendation based on risk level
        recommendation = self.generate_recommendation(risk_level, category)
        
        return {
            "category": category,
            "concerning_content": concerning_content,
            "confidence": confidence,
            "risk_level": risk_level,
            "self_harm_assessment": self_harm_assessment,
            "processed_text": processed_text,
            "recommendation": recommendation
        }
    
    def determine_risk_level(self, confidence, self_harm_assessment, category):
        """
        Determine overall risk level based on model confidence, self-harm assessment, and category
        """
        sh_risk = self_harm_assessment["risk_level"]
        
        # Determine if content is concerning (anything not supportive)
        concerning_content = category != "Supportive"
        
        # Higher risk categories (Attempt, Behavior) with high confidence or self-harm indicators
        if category in ["Attempt", "Behavior"] and (confidence > 0.8 or sh_risk in ["high", "medium"]):
            return "severe"
        
        # High confidence Attempt/Behavior OR medium confidence with self-harm indicators
        elif (category in ["Attempt", "Behavior"] and confidence > 0.7) or \
             (category == "Ideation" and confidence > 0.8 and sh_risk == "high"):
            return "high"
        
        # Ideation with decent confidence OR any category with medium self-harm risk
        elif (category == "Ideation" and confidence > 0.6) or \
             sh_risk == "medium" or \
             (category == "Indicator" and confidence > 0.8):
            return "moderate"
        
        # Lower confidence concerning categories OR elevated self-harm risk
        elif concerning_content and confidence > 0.5 or sh_risk == "elevated":
            return "low"
        
        # Supportive content or low confidence predictions with no self-harm indicators
        else:
            return "minimal"
    
    def assess_self_harm_risk(self, text):
        """
        Analyze text for self-harm indicators using keyword detection
        """
        text_lower = text.lower()
        
        # Define categories of concerning phrases based on C-SSRS assessment
        active_intent_phrases = [
            "want to die", "kill myself", "end my life", "suicide", 
            "hurt myself", "self harm", "cut myself", "hang myself",
            "take all my pills", "overdose", "don't want to live",
            "can't take it anymore", "end the pain"
        ]
        
        method_phrases = [
            "gun", "knife", "pills", "rope", "bridge", "jump", "cut", 
            "hang", "suffocate", "drown", "razor", "blade", "wrists",
            "shoot", "poison", "carbon monoxide", "train", "crash", "car"
        ]
        
        planning_phrases = [
            "wrote a note", "suicide note", "goodbye letter",
            "made a plan", "planned how", "when i'm gone", 
            "after i'm dead", "no one will miss me", "better off without me",
            "giving away", "final goodbye", "last time"
        ]
        
        timeframe_phrases = [
            "tonight", "tomorrow", "soon", "this week", 
            "can't take it anymore", "given up", "end it all",
            "no future", "nothing left", "last day"
        ]
        
        # Check for matches in each category
        active_intent = [phrase for phrase in active_intent_phrases if phrase in text_lower]
        methods = [phrase for phrase in method_phrases if phrase in text_lower]
        planning = [phrase for phrase in planning_phrases if phrase in text_lower]
        timeframe = [phrase for phrase in timeframe_phrases if phrase in text_lower]
        
        # Determine overall self-harm risk level
        risk_level = "low"
        
        if active_intent and (methods or planning):
            risk_level = "high"
        elif active_intent or (methods and planning):
            risk_level = "medium"
        elif methods or planning or timeframe:
            risk_level = "elevated"
        
        return {
            "risk_level": risk_level,
            "active_intent_indicators": active_intent,
            "method_indicators": methods,
            "planning_indicators": planning,
            "timeframe_indicators": timeframe,
            "requires_followup": risk_level != "low"
        }
    
    def generate_recommendation(self, risk_level, category):
        """Generate appropriate recommendation based on risk level and category."""
        
        if risk_level == "severe":
            return "URGENT ACTION RECOMMENDED: This content strongly indicates high risk of self-harm or suicide. Consider immediate intervention such as contacting emergency services or a crisis hotline."
        
        elif risk_level == "high":
            return "IMMEDIATE FOLLOW-UP RECOMMENDED: This content indicates significant risk. Direct outreach and connection to mental health resources should be prioritized."
        
        elif risk_level == "moderate":
            return "FOLLOW-UP RECOMMENDED: This content shows concerning signs that warrant attention. Consider providing mental health resources and follow-up support."
        
        elif risk_level == "low":
            return "MONITORING SUGGESTED: Some concerning elements present but lower immediate risk. Consider offering general mental health resources."
        
        else:  # minimal
            return "MINIMAL CONCERN: No significant indicators of immediate risk detected in this content."