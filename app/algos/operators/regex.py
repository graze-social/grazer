import numpy as np
import re
import re2

from app.algos.base import BaseParser
from app.algos.operators.attribute import resolve_path_batch
from app.helpers import is_likely_domain_or_url

class RegexParser(BaseParser):
    def get_record_texts(self, records, field_selector):
        return np.array([
            (
                " ".join(map(str, e))
                if isinstance(e, list)
                else (
                    "" if isinstance(e, np.ndarray) and np.all(e == None)
                    else (str(e) if str(e) else "")
                )
            )
            for e in resolve_path_batch(records, field_selector)
        ])

    async def matches_operator(
        self, records, field_selector, pattern, insensitive=True
    ):
        """
        Returns a NumPy array of booleans indicating whether the field in each
        record matches the pattern.
        """
        field_values = self.get_record_texts(records, field_selector)
        if insensitive:
            regex = re2.compile(pattern, re2.IGNORECASE)
        else:
            regex = re2.compile(pattern)
        using_re2 = type(regex) != re2.PythonRePattern
        if using_re2:
            return np.vectorize(lambda x: bool(re2.search(regex, x)))(field_values)
        else:
            return np.vectorize(lambda x: bool(re.search(pattern, x)))(field_values)


    async def negation_matches_operator(
        self, records, field_selector, pattern, insensitive=True
    ):
        """
        Returns a NumPy array of booleans indicating whether the field in each
        record does NOT match the pattern.
        """
        response = await self.matches_operator(
            records, field_selector, pattern, insensitive
        )
        return ~response

    async def allow_operator(
        self, records, field_selector, terms_list, insensitive=True, treat_as_regex=False
    ):
        """
        Returns a NumPy array of booleans indicating whether the field in each
        record matches any term in the terms_list, considering enhanced word boundaries.
        """
        # Adjust boundary pattern to avoid matches around invalid delimiters like '.'
        if treat_as_regex:
            combined_pattern = "|".join(terms_list)
        else:
            combined_pattern = "|".join(
                rf"{re2.escape(term)}(?![\w-])" if is_likely_domain_or_url(term)
                else rf"(?<![\w\-/])(?<!\.){re2.escape(term)}(?![\w-])"
                for term in terms_list
            )
        field_values = self.get_record_texts(records, field_selector)
        flags = re2.IGNORECASE if insensitive else 0
        return np.vectorize(lambda x: bool(re2.search(combined_pattern, x, flags)))(
            field_values
        )

    async def deny_operator(
        self, records, field_selector, terms_list, insensitive=True, treat_as_regex=False
    ):
        """
        Returns a NumPy array of booleans indicating whether the field in each
        record matches none of the terms in the terms_list, considering enhanced word boundaries.
        """
        response = await self.allow_operator(
            records, field_selector, terms_list, insensitive, treat_as_regex
        )
        return ~response

    async def register_operations(self, logic_evaluator):
        await logic_evaluator.add_operation("regex_matches", self.matches_operator)
        await logic_evaluator.add_operation(
            "regex_negation_matches", self.negation_matches_operator
        )
        await logic_evaluator.add_operation("regex_any", self.allow_operator)
        await logic_evaluator.add_operation("regex_none", self.deny_operator)
