# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.zh.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.zh.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.zh.verbalizers.fraction import FractionFst

# from nemo_text_processing.text_normalization.zh.verbalizers.math_symbol import MathSymbolFst
# from nemo_text_processing.text_normalization.zh.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.zh.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.zh.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.zh.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.zh.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst

        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst

        decimal = DecimalFst(deterministic=deterministic)
        decimal_graph = decimal.fst

        fraction = FractionFst(deterministic=deterministic)
        fraction_graph = fraction.fst

        date = DateFst(deterministic=deterministic)
        date_graph = date.fst

        # measure = MeasureFst(deterministic=deterministic)
        # measure_graph = measure.fst

        whitelist_graph = WhiteListFst(deterministic=deterministic).fst

        money_graph = MoneyFst(decimal=decimal, deterministic=deterministic).fst

        time_graph = TimeFst(deterministic=deterministic).fst

        # math_graph = MathSymbolFst(deterministic=deterministic).fst

        graph = (
            cardinal_graph
            # | measure_graph
            | decimal_graph
            | ordinal_graph
            | date_graph
            | money_graph
            | fraction_graph
            | whitelist_graph
            | time_graph
            # | math_graph
        )
        self.fst = graph
