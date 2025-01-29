from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class TopicParser(HuggingfaceClassifierParser):
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all"
    LABEL_MAP = {
        "arts_&_culture": "Arts & Culture",
        "business_&_entrepreneurs": "Business & Entrepreneurs",
        "celebrity_&_pop_culture": "Celebrity & Pop Culture",
        "diaries_&_daily_life": "Diaries & Daily Life",
        "family": "Family",
        "fashion_&_style": "Fashion & Style",
        "film_tv_&_video": "Film, TV & Video",
        "fitness_&_health": "Fitness & Health",
        "food_&_dining": "Food & Dining",
        "gaming": "Gaming",
        "learning_&_educational": "Learning & Educational",
        "music": "Music",
        "news_&_social_concern": "News & Social Concern",
        "other_hobbies": "Other Hobbies",
        "relationships": "Relationships",
        "science_&_technology": "Science & Technology",
        "sports": "Sports",
        "travel_&_adventure": "Travel & Adventure",
        "youth_&_student_life": "Youth & Student Life",
    }

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "topic_analysis", self.classifier_operator, True
        )
