from app.algos.operators.huggingface_classifier import HuggingfaceClassifierParser


class ModerationParser(HuggingfaceClassifierParser):
    MODEL_NAME = "KoalaAI/Text-Moderation"
    LABEL_MAP = {
        "S": "sexual",
        "H": "hate",
        "V": "violence",
        "HR": "harassment",
        "SH": "self-harm",
        "S3": "sexual/minors",
        "H2": "hate/threatening",
        "V2": "violence/graphic",
        "OK": "OK",
    }

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation(
            "content_moderation", self.classifier_operator, True
        )
