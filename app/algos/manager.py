import time
from app.logic_evaluator import LogicEvaluator
from app.algos.operators.entity import EntityParser
from app.algos.operators.regex import RegexParser
from app.algos.operators.transformer import TransformerParser
from app.algos.operators.attribute import AttributeParser
from app.algos.operators.logic import LogicParser
from app.algos.operators.social import SocialParser
from app.algos.operators.moderation import ModerationParser
from app.algos.operators.emotion_sentiment import EmotionSentimentParser
from app.algos.operators.financial_sentiment import FinancialSentimentParser
from app.algos.operators.language import LanguageParser
from app.algos.operators.sentiment import SentimentParser
from app.algos.operators.toxicity import ToxicityParser
from app.algos.operators.topic import TopicParser
from app.algos.operators.image_nsfw import ImageNSFWParser
from app.algos.operators.image_arbitrary import ImageArbitraryParser
from app.algos.operators.text_arbitrary import TextArbitraryParser
from app.algos.operators.combo import ComboParser


class AlgoManager:
    def __init__(
        self,
        algorithm_manifest,
        gpu_embedding_workers,
        gpu_classifier_workers,
        network_workers,
        cache,
    ):
        # Store the version and initialize parsers with shared model references
        self.algorithm_manifest = algorithm_manifest
        self.gpu_embedding_workers = gpu_embedding_workers
        self.gpu_classifier_workers = gpu_classifier_workers
        self.network_workers = network_workers
        self.cache = cache

    @classmethod
    async def initialize(
        cls,
        algorithm_manifest,
        gpu_embedding_workers,
        gpu_classifier_workers,
        network_workers,
        cache,
    ):
        """Async factory method to handle async initialization."""
        instance = cls(
            algorithm_manifest,
            gpu_embedding_workers,
            gpu_classifier_workers,
            network_workers,
            cache,
        )
        instance.parsers = {}
        instance.logic_evaluator = await LogicEvaluator.initialize()
        instance.parsers["entity"] = await EntityParser.initialize(instance)
        instance.parsers["regex"] = await RegexParser.initialize(instance)
        instance.parsers["transformer"] = await TransformerParser.initialize(instance)
        instance.parsers["attribute"] = await AttributeParser.initialize(instance)
        instance.parsers["social"] = await SocialParser.initialize(instance)
        instance.parsers["moderation"] = await ModerationParser.initialize(instance)
        instance.parsers["logic"] = await LogicParser.initialize(instance)
        instance.parsers["emotion_sentiment"] = await EmotionSentimentParser.initialize(
            instance
        )
        instance.parsers["financial_sentiment"] = (
            await FinancialSentimentParser.initialize(instance)
        )
        instance.parsers["language"] = await LanguageParser.initialize(instance)
        instance.parsers["sentiment"] = await SentimentParser.initialize(instance)
        instance.parsers["toxicity"] = await ToxicityParser.initialize(instance)
        instance.parsers["topic"] = await TopicParser.initialize(instance)
        instance.parsers["image_nsfw"] = await ImageNSFWParser.initialize(instance)
        instance.parsers["image_arbitrary"] = await ImageArbitraryParser.initialize(
            instance
        )
        instance.parsers["text_arbitrary"] = await TextArbitraryParser.initialize(
            instance
        )
        instance.parsers["combo"] = await ComboParser.initialize(instance)
        # instance.algorithm_manifest = instance.logic_evaluator.sort_conditions(instance.algorithm_manifest)
        return instance

    async def condition_keys(self):
        conditions = await self.logic_evaluator.extract_conditions(
            self.algorithm_manifest.get("filter", []) or []
        )
        condition_keys = {
            key for keys in [e.keys() for e in conditions] for key in keys
        }
        return condition_keys

    async def is_operable(self):
        condition_keys = await self.condition_keys()
        return len(condition_keys - (set(self.logic_evaluator.operations.keys()) | set('metadata'))) == 0

    async def is_gpu_accelerated(self):
        condition_keys = await self.condition_keys()
        return len(condition_keys & self.logic_evaluator.gpu_accelerable_operations) > 0

    async def matching_records(self, records):
        """Evaluate the manifest against a given record."""
        start_time = time.time()  # Record the start time
        matching_bools = await self.logic_evaluator.evaluate(
            self.algorithm_manifest["filter"], records
        )
        trace = []
        result = [
            {
                "uri": f"at://{record['did']}/{record['commit']['collection']}/{record['commit']['rkey']}",
                "cid": record["commit"]["cid"],
                "author": record["did"],
                "reply_parent": record["commit"]["record"]
                .get("reply", {})
                .get("parent", {})
                .get("uri"),
                "reply_root": record["commit"]["record"]
                .get("reply", {})
                .get("root", {})
                .get("uri"),
                "bluesky_created_at": record["commit"]["record"]["createdAt"],
                "raw_record": record,
            }
            for record, matches in zip(records, matching_bools)
            if matches
        ]
        end_time = time.time()  # Record the end time
        execution_time_microseconds = int((end_time - start_time) * 1_000_000)
        return result, trace, execution_time_microseconds

    async def audit_records(self, records, diagnostic_mode=False):
        """Evaluate the manifest against a given record."""
        return await self.logic_evaluator.evaluate_with_audit(
            self.algorithm_manifest["filter"], records
        )
