#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================== #
# Author:  Philipp Sandhaas                                                  #
# GitHub:  github.com/psandhaas                                              #
# Created: Thu, 18.09.25                                                     #
# ========================================================================== #

"""LangGraph implementation of guided RST parsing with an LLM, using structured
prompting."""

from __future__ import annotations
from enum import Enum
from glob import glob
from IPython.display import Image, display
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.main import GraphRecursionError
from operator import add
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Literal, Optional, Union
from typing_extensions import Annotated

from output_formats import Mononuclear, Multinuclear, EDU, Segmentation
from tree import binarize, Node


## Data Models ##
class State(MessagesState):
    text: str
    edus: Dict[int, EDU] = None
    spans: Annotated[List[Span], add] = []
    queue: List[Optional[Span]] = []
    current_parent: Optional[Span] = None
    current_children: List[Optional[Span]] = []


class Span(BaseModel, use_enum_values=True):
    start: int
    end: int
    nuclearity: Union[Literal["N", "S"], None] = None
    relation: Union[Mononuclear, Multinuclear, Literal["span"], None] = None
    type: Union[Literal["rst", "multinuc"], None] = None

    id: Union[int, None] = None
    parent: Union[int, None] = None

    @model_validator(mode="after")
    def check_span(self):
        if self.start < 0:
            raise ValueError("Start must be a non-negative integer")
        if self.end < self.start:
            raise ValueError("End must be greater than or equal to start")
        return self


class SplitArgs(BaseModel):
    k: int = Field(
        description=str(
            "Der EDU-Index, an dem der Span geteilt werden soll. Muss im Bereich" +
            " (span.start=i <= k < span.end=j) liegen. Der erste " +
            "resultierende Span inkludiert das EDU mit Index k und der " +
            "zweite Span beginnt mit k+1. " +
            "Beispiel: span=(1,3), k=2 -> resultierende Spans: (1,2) & (3,3)"
        ),
        gt=0
    )


class NuclearityArgs(BaseModel):
    nuclearity: Literal["N-N", "N-S", "S-N"] = Field(
        description=str(
            "Die Nuklearität, die den beiden Spans zugewiesen werden soll. " +
            "'N' steht für Nukleus, 'S' für Satellit. Die Reihenfolge der " +
            "Nuklearität entspricht der Reihenfolge der Spans."
        )
    )


class RelationArgs(BaseModel):
    """
    Zur Prüfung, ob für zwei Textsegmente eine bestimmte Relation anwendbar
    ist, sollte folgendermaßen vorgegangen werden:
        • Gibt es einen Hinweis durch einen Konnektor, der die Relation anzeigt
            oder die Menge der möglichen Relationen zumindest einschränkt? (In
            manchen Fällen ist allerdings die an der Oberfläche signalisierte
            Relation nicht die pragmatisch „wichtige“.)
        • Interpunktionszeichen liefern zwar keine sehr klaren Hinweise auf
            bestimmte Relationen, doch es gibt Tendenzen, wie z.B. der
            Zusammenhang zwischen dem Semikolon und kontrastiven Relationen.
        • Möchte der Autor durch die Juxtaposition der Segmente den jeweils
            genannten Effekt beim Leser erreichen? Dies ist eine notwendige
            Bedingung für die Anwendung einer Relation.
        • Wenn die Relationsdefinition Beschränkungen für den Typ oder die
            Funktion von Nukleus, Satellit, oder ihrer Kombination nennt, sind
            diese erfüllt? Dies sind (soweit vorhanden) ebenfalls notwendige
            Bedingungen.
        • Wird die mit der Relation verbundene Nukleus/Satellit-Verbindung der
            Rolle beider Segmente für die Textfunktion gerecht? Dies ist ein
            weniger striktes Kriterium als die beiden vorgenannten, kann aber
            oft die Entscheidung erleichtern, wenn mehrere Relationen anwendbar
            erscheinen.

    ## Relations-Definitionen
    Die Definitionen für mononukleare Relationen wie folgt:
        • N: Charakterisierung des Typs und/oder der Funktion des Nukleus (als
            Beschreibung der Haltung des Autors, nicht des Ausdrucks im Text)
        • S: Charakterisierung des Typs und/oder der Funktion des Satellits
            (als Beschreibung der Haltung des Autors, nicht des Ausdrucks im
            Text)
        • N/S: Charakterisierung der Funktion der Nukleus/Satellit Kombination.
            Wenn es Beschränkungen oder Tendenzen für die textuelle Abfolge von
            N und S gibt, sind sie hier ebenfalls genannt.
        • Effekt: Charakterisierung des mit der Verwendung der Relation vom
            Autor intendierten Effekts, formuliert als „vorher–nachher“
            Veränderung.
        • Typische Konnektoren
        • Beispiel: N und S sind jeweils markiert. Wenn vorab ein Kontext
            charakterisiert wird, geschieht das in Kursivschrift.
        • Bemerkung: (optional)
    Die Felder N, S und N/S bleiben frei, wenn es für eine Relation keine
    entsprechenden Beschränkungen gibt. Bei multinuklearen Relationen entfallen
    die Felder S und N/S.

    ### Mononukleare Relationen
    #### Attribution
    • N: Inhalt der berichteten Nachricht (muss in einem separaten Satzteil
        stehen)
    • S: Quelle der Attribution (ein Satzteil mit einem berichtenden Verb oder
        eine Phrase, die z.B. mit "entsprechend" oder "gemäß" beginnt)
    • N/S: Um einen Satz in Attributionsquelle und Inhalt zu segmentieren,
        müssen zwei Bedingungen erfüllt sein:
        1) Es muss eine explizite Quelle für die Attribution vorhanden sein.
        Wenn der Satz, der das berichtende Verb enthält, die Quelle der
        Zuschreibung nicht angibt und die Quelle auch nicht an anderer Stelle
        im Satz oder im näheren Kontext identifiziert werden kann, besteht
        keine Attribution-Relation, und der berichtende und der berichtete Satz
        werden als eine Einheit behandelt. (Dies kommt häufig bei
        Passivkonstruktionen oder generischen Ausdrücken vor.)
        2) Der Nebensatz darf kein Infinitivkomplement sein.
    • Effekt: Der Leser erkennt, dass eine Quelle in S über den Inhalt in N
        berichtet und diesen einer anderen Quelle zuschreibt.
    • Beispiel: [Analysten schätzten]S [dass die Umsätze in US-amerikanischen
        Geschäften im Quartal ebenfalls zurückgingen.]N
    • Bemerkung: Die Relation wird auch mit kognitiven Prädikaten verwendet,
        um Gefühle, Gedanken, Hoffnungen usw. einzuschließen. Sie gilt auch
        im Fall der negativen Formulierung (z.B. "er bestritt, dass...").
    #### Background
    • N/S: Das Verstehen von S erleichtert dem Leser das Verständnis für den
        Inhalt von N; S enthält orientierende Hintergrundinformation, ohne die
        N nicht oder nur schwer verständlich wäre. Im Text geht S meist dem N
        voraus, aber nicht immer. Ein Background-Satellit am Anfang eines
        Textes hat oftmals auch die Funktion, das Thema kurz einzuführen.
    • Effekt: Die Fähigkeit des Lesers, den Inhalt von N zu verstehen, wird
        verbessert.
    • Typische Konnektoren: (selten durch Konnektoren angezeigt)
    • Beispiel: [Burkina Faso hieß bis 1984 noch Obervolta.]S [Nach einer
        EMNID-Umfrage glauben viele Europäer bis heute, dass es sich um zwei
        verschiedene Länder handelt.]N
    • Bemerkung: Die Relation besteht eher selten zwischen EDUs, sondern in der
        Regel zwischen größeren Segmenten. Viele Kommentare sind so
        strukturiert, dass ein Background-Satellit den Textanfang bildet, also
        den inhaltlichen Ausgangspunkt für die nachfolgende Kommentierung
        darstellt.
    #### Cause
    • N: ein realer Sachverhalt in der Welt.
    • S: ein realer Sachverhalt in der Welt.
    • N/S: der in N beschriebene Sachverhalt wird durch den in S beschriebenen
        Sachverhalt verursacht.
    • Effekt: Leser erkennt den Kausalzusammenhang in der Welt.
    • Typische Konnektoren: weil; da; deshalb; ...
    • Beispiel: [Überrascht reagierte auch Bürgermeister Jochen Wagner.]N
        [Schließlich gaben die Stadtverordneten erst Montagabend grünes Licht
        für die weitere Erschließung des neuen Ortsteils Diepensee.]S
    #### Comparison
    • N: Entität/Sachverhalt/Bereich, der als Vergleichsobjekt dient.
    • S: Entität/Sachverhalt/Bereich, der hinsichtlich eines sich
        unterscheidenden Aspekts mit N verglichen wird.
    • N/S: N & S gleichen einander in einer oder mehr Dimensionen und
        unterscheiden sich in einem oder mehreren Aspekten. Die
        Bereiche/Entitäten/usw. stehen nicht im Gegensatz zueinander.
    • Effekt: Leser erkennt den Vergleichscharakter der Relation.
    • Bemerkung: Comparison vergleicht zwei Textbereiche anhand einer
        Dimension, die abstrakt sein kann. Die Relation kann vermitteln, dass
        einige abstrakte Entitäten, die sich auf die Vergleichsrelation
        beziehen, ähnlich, unterschiedlich, größer als, kleiner als usw. sind.
    #### Condition
    • N: Eine hypothetische, künftige oder anderweitig irreale Situation.
    • S: Eine hypothetische, künftige oder anderweitig irreale Situation.
    • N/S: S beeinflusst die Realisierung von N: N wird nur dann (nicht)
        realisiert, wenn S (nicht) realisiert wird.
    • Effekt: Leser erkennt die Abhängigkeit der (Nicht-)Realisierung von N von
        der (Nicht-)Realisierung von S.
    • Typische Konnektoren: es sei denn; ...
    • Beispiel: [Morgen wird der Satellit in den Pazifik stürzen.]N [Es sei
        denn, er verglüht doch noch vollständig in der Erdatmosphäre.]S
    #### Contrast
    • N: Ein „wichtigerer“ Inhalt, der mit S vergleichbar aber nicht identisch
        ist.
    • S: Ein weniger „wichtiger“ Inhalt, der mit N vergleichbar aber nicht
        identisch ist.
    • N/S: Die Inhalte sind einander ähnlich, miteinander vergleichbar; sie
        sind aber nicht identisch, sondern unterscheiden sich in für den Autor
        wichtigen Aspekten.
    • Effekt: Leser erkennt die Vergleichbarkeit von N & S und die Betonung
        des Unterschieds.
    • Typische Konnektoren: demgegenüber; hingegen; während; aber; ...
    • Beispiel: [Mein erstes Auto war ein Kleinwagen.]N/S [Das zweite hingegen
        ein ausgewachsener Kombi.]S/N
    #### Elaboration
    • N/S: S liefert genauer Information bzw. Details zum Inhalt von N. N geht
        S im Text voraus. Typische Zusammenhänge zwischen N und S sind
        Menge::Element, Ganzes::Teil, Abstraktion::Instanz,
        Vorgang::Einzelschritt.
    • Effekt: Leser erkennt, dass S genauere Information zu N liefert.
    • Typische Konnektoren: besonders; beispielsweise; ...
    • Beispiel: [Diepensee siedelt um.]N [Ohne Wenn und Aber.]S
    #### Enablement
    • N: Eine vom Leser auszuführende Tätigkeit.
    • N/S: Das Verstehen von S erleichtert dem Leser, die von N beschriebene
        Tätigkeit auszuführen.
    • Effekt: Die Fähigkeit des Lesers, die Tätigkeit in N auszuführen, wird
        gesteigert.
    • Typische Konnektoren: damit; ...
    • Beispiel: [Wechseln Sie die Zündkerzen aus.]N [Ein Vierkantschlüssel
        befindet sich unter der Abdeckung.]S
    #### Evaluation
    • N: Beschreibung eines Sachverhalts, oder eine subjektive Aussage
        (allerdings nicht aus Perspektive des Autors)
    • S: Eine subjektive Bewertung (positiv/negativ, erstrebenswert/nicht
        erstrebenswert) aus Perspektive des Autors
    • N/S: S bewertet N
    • Effekt: Leser erkennt die Bewertungsrelation zwischen S und N
    • Typische Konnektoren: (selten durch Konnektoren angezeigt)
    • Beispiel: [Seine Vergangenheit schien wie ein Fluch über dem Hotelkomplex
        zu liegen.]S [Jahrelang hatte das Amtsgericht Potsdam umsonst versucht,
        es an den Mann zu bringen.]N
    • Bemerkung: Meist folgt das evaluierende Segment auf das evaluierte;
        manchmal ist es jedoch umgekehrt, wie im obigen Beispiel.
    #### Explanation
    • N: Eine Aussage/Einschätzung/These, die der Leser möglicherweise nicht
        akzeptiert oder als nicht genügend wichtig oder positiv einschätzt.
    • S: Eine Aussage, die der Leser wohl akzeptieren wird; in der Regel die
        „objektive“ Beschreibung eines Faktums.
    • N/S: Durch das Verstehen von S akzeptiert der Leser die Aussage von N
        leichter, bzw. teilt die damit verbundene Einschätzung des Autors.
    • Effekt: Leser glaubt eher, dass die in N getroffene Aussage zutrifft.
    • Typische Konnektoren: (kausale Konnektoren)
    • Beispiel: [Und nun scheint sogar unsere Landesregierung entschlossen,
        diese scheinbare Gleichbehandlung der beiden Fächer zu beseitigen.]N
        [Stolpe, Reiche und Co. sagen zwar Ja zu einem möglichen
        Kompromissangebot aus Karlsruhe, dekretieren aber: Einen
        Wahlpflichtbereich LER/Religion kann es nicht geben.]S
    • Bemerkung: Explanation verbindet oft ein längeres Satellit-Segment mit
        einem kürzeren Nukleus (der These).
    #### Manner-Means
    • N: Eine Handlung/Aktivität
    • N/S: S gibt Informationen, die die Realisierung/Ausführung von N
        wahrscheinlicher/einfacher machen (z. B. ein Instrument)
    • Effekt: Leser erkennt den Zusammenhang der höheren Wahrscheinlichkeit
        oder der Vereinfachung der Handlungsausführung
    • Typische Konnektoren: dazu; damit; ...
    • Beispiele: [Berliner fahren im August immer gern nach Lichtenrade.]N
        [Dazu nehmen sie meistens die S25.]S
    #### Summary
    • N: N umfasst mehr als nur eine EDU.
    • N/S: S folgt im Text auf N und wiederholt die Information von N, ist
        jedoch kürzer.
    • Effekt: Leser erkennt die zusammenfassende Funktion von S.
    • Typische Konnektoren: in Kürze; ...
    #### TextualOrganization
    • N/S: S verknüpft N mit einem funktional-strukturell übergeordneten
        Element. Weder S noch N haben eine rhetorische Beziehung zueinander,
        sondern dienen der Organisation des Texts.
    • Effekt: Leser erkennt die zusammenfassende Funktion von S.
    • Bemerkung: TextualOrganization verknüpft funktional-strukturelle
        Elemente, wie z. B. Titel, Autor oder Signaturblock.
    #### Topic-Change
    • N: Das Thema von N ist das bedeutendere Element, zu dem der Fokus
        wechselt.
    • S: Das Thema von S ist das weniger bedeutende Element, das aus dem Fokus
        rückt.
    • Bemerkung: Topic-Change wird verwendet, um größere Textabschnitte zu
        verknüpfen, wenn der Fokus von einem Abschnitt zum anderen wechselt.
    #### Topic-Comment
    • S: Der Inhalt von S kann als „Problem“ aufgefasst werden.
    • N/S: Der Inhalt von N kann als Lösung des in S dargestellten Problems
        aufgefasst werden. N geht in der Regel S im Text voraus.
    • Effekt: Leser erkennt N als Lösung des Problems in S.
    • Typische Konnektoren: (selten durch Konnektoren angezeigt)
    • Beispiel: [Mit der Verabschiedung des Nichtraucherschutzgesetzes sitzen
        viele Kneipen in der Falle.]S [Es empfiehlt sich, früh genug auf die
        Einrichtung abtrennbarer Räume zu achten.]N

    ### Multinukleare Relationen
    #### Joint
    • N: Die nicht unbedingt typgleichen Nuklei geben separate Informationen,
        stehen aber in keiner klar identifizierbaren semantischen oder
        pragmatischen Relation zueinander, haben gemeinsam auch nicht den
        Charakter einer Aufzählung. Nichtsdestotrotz besteht eine kohärente
        Verbindung, weil sie der übergeordneten Textfunktion gleichermaßen
        dienen.
    • Effekt: Leser erkennt, dass jeder Nukleus „eine eigene Botschaft“ hat,
        die aber jeweils derselben Textfunktion dienlich sind.
    • Typische Konnektoren: additive Konnektoren wie "zudem", "auch"
    • Bemerkung: Joint ist zu verwenden, wenn eine multinukleare Relation
        gesucht wird und keine der übrigen passt.
    #### Same-Unit
    • N: Die in bestimmtem Sinne typgleichen Nuklei geben Informationen, die
        als zusammengehörig, aufzählend erkennbar sind, mithin eine gemeinsame
        Rolle für die Textfunktion spielen.
    • Effekt: Leser erkennt die gemeinsame Funktion der Nuklei.
    • Typische Konnektoren: Komma, Nummerierungen, "und", "oder",
        "je A, desto B", ...
    • Beispiel: Was ich gestern getan habe? [Essen kochen,]N [Kinder
        versorgen,]N [Bad putzen.]N
    • Bemerkung: Same-Unit findet vor allem Anwendung, um zwei, z.B. durch
        Relativsätze oder Parenthesen unterbrochene, Teile des gleichen EDUs
        zu verbinden.
    #### Temporal
    • N: Die Nuklei beschreiben Sachverhalte der Welt, die in einer bestimmten
        zeitlichen Abfolge stattfinden.
    • Effekt: Leser erkennt die temporale Relation zwischen den Nuklei.
    • Typische Konnektoren: "dann"; "anschließend"; "und"; "zuvor", ...
    • Beispiel: [Um neun betrat die Lehrerin den Klassenraum.]N [Fünf Minuten
        später verkündete sie, dass ein Test geschrieben wird.]N
    • Bemerkung: Die Nennung der Ereignisse kann der zeitlichen Abfolge
        entsprechen ("dann") oder gegenäufig sein ("zuvor").
    """
    relation: Union[Mononuclear, Multinuclear] = Field(
        description=str(
            "Die RST-Relation, die die beiden Spans verbindet. Falls beide " +
            "Spans Nuklei sind, muss die Relation eine Multinuclear-Relation " +
            "sein. Falls einer der beiden Spans ein Satellit ist, muss die " +
            "Relation eine Mononuclear-Relation sein."
        )
    )


## Functions ##
def split_span(span: Span, k: int) -> List[Span]:
    """Teile einen `span[i:j]` in zwei konsekutive Teil-Spans `span[i:k]` &
    `span[k+1:j]`.
    
    Args:
        span (Span): Der zu teilende Span.
        k (int): Der Index, an dem der Span geteilt werden soll. Muss im
            Bereich (span.start, span.end) liegen.

    Returns:
        List[Span]: Eine Liste mit den beiden neuen Spans.
    """
    # if k <= span.start or k >= span.end:
    #     raise ValueError("k must be within the span range")
    return [Span(start=span.start, end=k),
            Span(start=k + 1, end=span.end)]


def assign_nuclearity(
    span1: Span, span2: Span,
    nuclearity: Literal["N-N", "N-S", "S-N"]
) -> List[Span]:
    """Weise zwei Spans ihre Nuklearität zu.
    
    Args:
        span1 (Span): Der erste Span.
        span2 (Span): Der zweite Span.
        nuclearity (Literal["N-N", "N-S", "S-N"]): Die Nuklearität der Spans.
            "N" steht für Nukleus, "S" für Satellit. Die Reihenfolge der
            Nuklearität entspricht der Reihenfolge der Spans.

    Returns:
        List[Span]: Eine Liste von [span1, span2], deren Span.nuclearity
        Attribute die entsprechende Nuklearität zugewiesen bekommen haben.
    """
    nuc1, nuc2 = nuclearity.split("-")
    span1.nuclearity, span2.nuclearity = nuc1, nuc2
    return [span1, span2]


def label_relation(
    span1: Span, span2: Span,
    relation: Union[Mononuclear, Multinuclear]
) -> List[Span]:
    """Verbinde zwei Spans mit einer RST-Relation.
    
    ## Beachte:
    - Falls beide Spans Nuklei sind, muss die Relation eine
    Multinuclear-Relation sein.
    - Falls einer der beiden Spans ein Satellit ist, muss die Relation eine
    Mononuclear-Relation sein.
    
    Args:
        span1 (Span): Der erste Span.
        span2 (Span): Der zweite Span.
        relation (Union[Mononuclear, Multinuclear]): Die RST-Relation, die die
            beiden Spans verbindet.

    Returns:
        List[Span]: Eine Liste von [span1, span2], deren Span.relation
        Attribute die entsprechende RST-Relation zugewiesen bekommen haben.
    """
    span1.relation, span2.relation = relation, relation
    return [span1, span2]


def set_id(span: Span, state: State) -> Span:
    if span.id is not None:
        return span
    elif span.start == span.end:
        span.id = span.start
    else:
        current_ids = [
            s.model_dump().get("id", 0) for s in state["spans"]
        ] + [
            s.model_dump().get("id", 0) for s in state["queue"]
        ] + [0]
        if (current_parent := state.get("current_parent")) is not None:
            current_ids.extend(
                [current_parent.model_dump().get("id", 0)]
            )
        if len(current_children := state.get("current_children", [])) > 0:
            current_ids.extend(
                [s.model_dump().get("id", 0) for s in current_children
                 if s is not None]
            )
        if (edus := state.get("edus")) is not None:
            current_ids.append(len(edus))
        current_ids = sorted([
            int(id_) for id_ in current_ids if id_ is not None
        ] + [0])
        if (current_max := current_ids[-1]) == 0:
            return span
        span.id = current_max + 1
    return span


def construct_examples():
    def load_texts_and_trees() -> Dict[str, Dict[str, Union[str, Node]]]:
        examples_dir = "C:/Users/SANDHAP/Repos/rst-parsing/data/examples"
        texts = {}
        for p in glob(f"{examples_dir}/*.txt"):
            with open(p, "r", encoding="utf-8") as f:
                texts[Path(p).stem] = f.read().split("\n\n")[-1]  # no headings
        examples = {}
        for file in texts.keys():
            examples[file] = {
                "text": texts[file],
                "tree": Node.from_rs3(
                    f"{examples_dir}/{file}.rs3",
                    exclude_disjunct_segments=True
                )
            }
        return examples
    
    def segmentation(example: Dict[str, Union[str, Node]]) -> Segmentation:
        return Segmentation(
            document=example["text"], edus_=example["tree"].edus
        )

    def init_state(example: Dict[str, Union[str, Node]]) -> State:
        return enqueue_parent(State(
            text=example["text"],
            edus=segmentation(example).edus,
            queue=[Span(start=1, end=len(segmentation(example).edus))],
        ))

    def get_split_args(example: Dict[str, Union[str, Node]]):
        for i, node in enumerate(example["tree"]):
            if i > 0:  # skip root


    examples = load_texts_and_trees()
    pass


## Nodes ##
def segment_text(state: State):
    """Segmentiere den Input-Text in EDUs."""
    prompt = f"Segmentiere den folgenden Text in EDUs:\n{state['text']}"
    res = llm.with_structured_output(Segmentation).invoke(prompt)
    return {"edus": res.edus}



def create_root(state: State):
    """Erstelle einen Wurzel-Span, der alle EDUs umfasst und füge ihn der
    Queue hinzu.

    Args:
        edus (Segmentation): Die Segmentierung, die alle EDUs eines Texts
        enthält.

    Returns:
        Span: Ein Span mit folgenden Attributen:
            - start: Der Index des ersten EDUs (1).
            - end: Der Index des letzten EDUs (len(edus.edus)).
            - nuclearity: Die Nuklearität des Spans (None).
            - relation: None.
            - type: "span".
    """
    root = Span(
        start=1,
        end=len(state['edus']),
        nuclearity=None,
        relation="span",
        type=None,
        parent=None
    )
    return {"queue": [root]}


def enqueue_parent(state: State):
    """Füge den aktuellen Parent-Span der Queue hinzu und setze seine ID."""
    spans = []
    queue = state["queue"]
    next_parent = None
    while queue:
        candidate = queue.pop(0)
        if candidate.start < candidate.end:
            next_parent = candidate
            break
        elif candidate.start == candidate.end:
            # shouldn't happen
            spans.append(candidate)
    if next_parent is not None:
        next_parent = set_id(next_parent, state)
    return {
        "spans": spans,
        "queue": queue,
        "current_parent": next_parent
    }


def split_parent(state: State):
    """Teile den aktuellen Parent-Span in zwei Teil-Spans."""
    parent = state["current_parent"]
    prompt = str(
        f"Gegeben den aktuellen Span, der die EDUs {parent.start} bis " +
        f"einschließlich {parent.end} umfasst, teile diesen Span in zwei " +
        "Teil-Spans an einem geeigneten Index k, sodass ein linker " +
        "Tochter-Span ([i:k]) und ein rechter Tochter-Span ([k+1:j]) " +
        f"entstehen.\nHier ist der aktuelle Span:\n{parent.model_dump_json()}"
    )
    res = llm.with_structured_output(SplitArgs).invoke(prompt)
    left, right = split_span(parent, res.k)
    return {"current_children": [left, right]}


def assign_nuc_to_children(state: State):
    """Weise den beiden aktuellen Kind-Spans ihre Nuklearität zu. Auf Basis
    der Nuklearität wird auch die Parent-ID gesetzt:
    - Falls beide Kinder Nuklei sind:
        - `span.parent = state["current_parent"].id`
    - Falls einer der beiden Kinder ein Satellit ist:
        - Nukleus: `span_N.parent = state["current_parent"].id`
        - Satellit: `span_S.parent = span_N.id`
    """
    left, right = state["current_children"]
    left_text = " ".join([
        state["edus"][i].text for i in range(left.start, left.end)
    ])
    right_text = " ".join([
        state["edus"][i].text for i in range(right.start, right.end)
    ])
    prompt = str(
        "Weise den beiden Spans ihre Nuklearität zu.\n" +
        f"Text des linken Spans (EDUs[{left.start}:{left.end}]): {left_text}\n" +
        f"Text des rechten Spans (EDUs[{right.start}:{right.end}]): {right_text}"
    )
    res = llm.with_structured_output(NuclearityArgs).invoke(prompt)
    left, right = assign_nuclearity(left, right, res.nuclearity)
    
    if left.nuclearity == "N" and right.nuclearity == "N":
        left.parent = state["current_parent"].id
        right.parent = state["current_parent"].id
    elif left.nuclearity == "N" and right.nuclearity == "S":
        left.parent = state["current_parent"].id
        right.parent = left.id
    else:  # left.nuclearity == "S" and right.nuclearity == "N"
        right.parent = state["current_parent"].id
        left.parent = right.id
    
    return {"current_children": [left, right]}


def assign_relation_and_types(state: State):
    """Verbindet die beiden Töchter-Spans mit einer RST-Relation und weist
    dem Mutter-Span den Typ der RST-Relation zu."""
    parent = state["current_parent"]
    left, right = state["current_children"]
    left_nuc, right_nuc = left.nuclearity, right.nuclearity
    left_text = " ".join([
        state["edus"][i].text for i in range(left.start, left.end)
    ])
    right_text = " ".join([
        state["edus"][i].text for i in range(right.start, right.end)
    ])
    prompt = str(
        "Verbinde die folgenden zwei Spans mit einer passenden RST-Relation." +
        f"\nLinker Span (EDUs[{left.start}:{left.end}], Nuklearität: {left_nuc}): \n{left_text}" +
        f"\nRechter Span (EDUs[{right.start}:{right.end}], Nuklearität: {right_nuc}): \n{right_text}"
    )
    res = llm.with_structured_output(RelationArgs).invoke(prompt)
    if (relation := res.relation) in Mononuclear:
        parent.type = "rst"
        if left.nuclearity == "S":
            left.relation = relation
            right.relation = "span"
        else:
            left.relation = "span"
            right.relation = relation
    else:
        parent.type = "multinuc"
        left.relation = relation
        right.relation = relation
    
    # Füge Parent-Span zum Baum hinzu
    parsed_spans = [parent]

    # Prüfe ob Töchter terminal sind oder weiter aufgeteilt werden müssen
    queue = state["queue"]
    if left.start == left.end:
        left.id = left.start
        parsed_spans.append(left)
    else:
        queue.append(left)
    if right.start == right.end:
        right.id = right.start
        parsed_spans.append(right)
    else:
        queue.append(right)

    return {
        # Setze aktuellen Parent und Kinder zurück
        "current_parent": None,
        "current_children": [],
        # Füge Spans zur nächsten Iteration der Queue hinzu
        "queue": queue,
        # Aktualisiere die Liste der geparsten Spans
        "spans": parsed_spans
    }


# Conditional edge to check if parsing is complete
def should_continue(state: State) -> Literal["parse", END]:
    """Decide if we should continue the loop or stop based upon whether the
    LLM made a tool call
    """
    if len(state["queue"]) > 0:
        return "parse"
    return END


## Graph ##
def build_graph(parse_only: bool = False) -> CompiledStateGraph:
    """Compile the state graph for RST parsing.
    
    :param parse_only: If `True`, only perform parsing, i.e. without segmenting
        the input text into EDUs. Defaults to `False`.
    :type parse_only: `bool`, optional

    :return: The compiled state graph.
    :rtype: `CompiledStateGraph`
    """
    
    graph: StateGraph[State, None, State, State] = StateGraph(State)

    graph.add_node("Create root span", create_root)
    graph.add_node("Enqueue parent span", enqueue_parent)
    graph.add_node("Split parent into child-spans", split_parent)
    graph.add_node("Assign nuclearity to children", assign_nuc_to_children)
    graph.add_node("Assign relation and types", assign_relation_and_types)

    graph.add_edge("Create root span", "Enqueue parent span")
    graph.add_edge("Enqueue parent span", "Split parent into child-spans")
    graph.add_edge("Split parent into child-spans", "Assign nuclearity to children")
    graph.add_edge("Assign nuclearity to children", "Assign relation and types")
    graph.add_conditional_edges(
        "Assign relation and types",
        should_continue,
        {
            "parse": "Enqueue parent span",
            END: END
        }
    )

    if parse_only:
        graph.add_edge(START, "Create root span")
    else:
        graph.add_node("Segmentation", segment_text)
        graph.add_edge(START, "Segmentation")
        graph.add_edge("Segmentation", "Create root span")

    return graph.compile()


## Misc ##
def visualize_graph(graph: CompiledStateGraph):
    """Visualisiere den Graphen."""
    display(Image(graph.get_graph().draw_mermaid_png()))


def structured_output_to_node(llm_output: Dict) -> Node:
    """Converts structured LLM output to a `Node` object.
    
    :param llm_output: The structured output from the LLM, with the following
        keys:

        - `'spans'`: list of `Span` objects.
        - `'edus'`: dict, mapping EDU indices to `EDU` objects.
    :type llm_output: Dict

    :return: The root `Node` of the constructed tree.
    :rtype: Node
    """
    spans: List[Dict] = [s.model_dump() for s in llm_output["spans"]]
    nodes = {}
    for parsed in spans:
        if (span := (parsed.pop("start"), parsed.pop("end")))[0] == span[1]:
            if (edu := llm_output["edus"].get(span[0])) is not None:
                parsed["text"] = edu.text
        parsed["span"] = span
        if (nuc := parsed.pop("nuclearity", None)) is not None:
            parsed["nuc"] = "Nucleus" if nuc == "N" else "Satellite"
        if (relname := parsed.pop("relation", None)) is not None:
            parsed["relname"] = relname.value if isinstance(
                relname, Enum
            ) else relname
        nodes[span] = parsed
    nodes = {
        span: nodes[span] for span in sorted(
            nodes.keys(), key=lambda x: (x[0], x[0] - x[1])
        )
    }

    try:
        return binarize(nodes)
    except Exception as e:
        print(
            f"Something went wrong during binarization: {e}\n" +
            "Returning unconverted LLM output as a dictionary."
        )
        return nodes


def parse_rst(
    model: BaseChatModel,
    text: Optional[Union[str, List[str]]] = None,
    edus: Optional[Union[List[str], List[List[str]]]] = None,
    return_structured_output: bool = False
) -> Union[List[Node], List[Dict]]:
    """Convenience wrapper for parsing the RST trees of texts, using the
    provided LLM.
    
    :param model: The language model to use for parsing. *Note that the underlying
        model must support structured output.*
    :type model: `langchain_core.chat_models.BaseChatModel`
    :param text: The input text(s) to be parsed. If provided, the text will be
        segmented into EDUs before parsing. If a list of texts is provided, it
        is treated as a batch of documents. If `None`, pre-segmented EDUs must
        be provided via the `edus` parameter.
    :type text: `str | List[str] | None`
    :param edus: A list of pre-segmented EDUs to use for parsing. If provided,
        the LLM will skip the segmentation step and directly parse an RST-tree
        from the EDUs. If a list of lists is provided, it is treated as a batch
        of documents. If `None`, the input text(s) must be provided via the
        `text` parameter.
    :type edus: `List[str] | List[List[str]] | None`
    :param return_structured_output: Whether to return the raw structured
        output. If `False`, the structured output will be converted to a binary
        tree of `Node` objects. Defaults to `False`.
    :type return_structured_output: `bool`, optional

    :return: The root `Node` of the constructed tree or the raw structured
        output as a dict with keys `'text'`, `'edus'`, `'spans'`, `'queue'`,
        `'current_parent'`, and `'current_children'`. If a batch of inputs is
        provided, a list is returned.
    :rtype: `Node` | `Dict[str, Union[
        str,
        List[EDU],
        Dict[int, EDU],
        List[Span],
        Optional[Span],
        List[Optional[Span]
    ]`

    :raises ValueError: If neither `text` nor `edus` or both is provided.
    :raises NotImplementedError: If the provided model does not support
        structured output.
    """
    if text is None and edus is None:
        raise ValueError("Either `text` or `edus` must be provided.")
    if text is not None and edus is not None:
        raise ValueError("Only one of `text` or `edus` can be provided.")
    
    # prepare input and graph
    if edus is not None:
        graph = build_graph(parse_only=True)
        if not isinstance(edus, list):
            edus = [edus]
        input: List[Dict[str, Dict[int, EDU]]] = [
            {"edus": Segmentation(
                document=None,
                edus_=[EDU(text=seg) for seg in edu]
            ).edus} for edu in edus
        ]
    else:
        graph: CompiledStateGraph = build_graph()
        if not isinstance(text, list):
            text = [text]
        input = [{"text": txt} for txt in text]

    global llm
    llm = model
    max_recursion_limit = 150
    if len(input) == 1:
        try:
            res = [graph.invoke(input[0], {"recursion_limit": 100})]
        except GraphRecursionError:
            print(
                "Graph recursion limit reached at 100 steps. " +
                f"Increasing limit to {max_recursion_limit} and retrying..."
            )
            res = [graph.invoke(
                input[0], {"recursion_limit": max_recursion_limit}
            )]
    else:
        try:
            res = graph.batch(
                [inp for inp in input], {"recursion_limit": 100}
            )
        except GraphRecursionError:
            print(
                "Graph recursion limit reached at 100 steps. " +
                f"Increasing limit to {max_recursion_limit} and retrying..."
            )
            res = graph.batch(
                [inp for inp in input],
                {"recursion_limit": max_recursion_limit}
            )

    if return_structured_output:
        return res
    return [structured_output_to_node(r) for r in res]
