from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class LanguageParser(HuggingfaceClassifierParser):
    MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
    LABEL_MAP = {
        "ja": "Japanese",
        "nl": "Dutch",
        "ar": "Arabic",
        "pl": "Polish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "tr": "Turkish",
        "es": "Spanish",
        "hi": "Hindi",
        "el": "Greek",
        "ur": "Urdu",
        "bg": "Bulgarian",
        "en": "English",
        "fr": "French",
        "zh": "Chinese",
        "ru": "Russian",
        "th": "Thai",
        "sw": "Swahili",
        "vi": "Vietnamese",
    }

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "language_analysis", self.classifier_operator, True
        )
