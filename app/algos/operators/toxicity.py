from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class ToxicityParser(HuggingfaceClassifierParser):
    MODEL_NAME = "unitary/toxic-bert"
    LABEL_MAP = {
        "toxic": "Toxic",
        "identity_hate": "Identity Hate",
        "insult": "Insult",
        "obscene": "Obscene",
        "severe_toxic": "Severe Toxicity",
        "threat": "Threat",
    }

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "toxicity_analysis", self.classifier_operator, True
        )
