from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Dict, List, Literal, Optional, Union


class Mononuclear(StrEnum):
    ATTRIBUTION = "Attribution"
    BACKGROUND = "Background"
    CAUSE = "Cause"
    COMPARISON = "Comparison"
    CONDITION = "Condition"
    CONTRAST = "Contrast"
    ELABORATION = "Elaboration"
    ENABLEMENT = "Enablement"
    EVALUATION = "Evaluation"
    EXPLANATION = "Explanation"
    MANNER_MEANS = "Manner-Means"
    SUMMARY = "Summary"
    TEXTUAL_ORGANIZATION = "TextualOrganization"
    TOPIC_CHANGE = "Topic-Change"
    TOPIC_COMMENT = "Topic-Comment"


class Multinuclear(StrEnum):
    JOINT = "Joint"
    SAME_UNIT = "Same-Unit"
    TEMPORAL = "Temporal"


class RSTNode(BaseModel):  # TODO: RST-Knoten beschreiben & validieren
    """
    """
    
    node_type: Union[EDU, RSTRelation] = Field(
        description=str(
            "Der Typ des RST-Knoten, der entweder ein EDU oder eine " +
            "RST-Relation ist."
        )
    )
    nuclearity: Literal["Nucleus", "Satellite", "<ROOT>"] = Field(
        description=str(
            "Die Nuklearität des Knotens in Bezug auf die übergeordnete " +
            "RST-Relation. Falls es sich um die Wurzel aller Knoten " +
            "innerhalb eines Texts handelt, ist die Nuklearität '<ROOT>'. " +
            "Falls der Knoten Teil einer mononuklearen Relation ist und " +
            "der wichtigere Knoten (hinsichtlich der Gesamtbedeutung des " +
            "Textes) ist, ist die Nuklearität 'Nucleus', andernfalls " +
            "'Satellite'. Falls der Knoten Teil einer multinuklearen " +
            "Relation ist, ist die Nuklearität 'Nucleus'."
        )
    )

    parent: Optional[RSTNode] = Field(
        description=str(
            "Der übergeordnete Knoten in der RST-Baumstruktur. Falls es " +
            "sich um die Wurzel aller Knoten innerhalb eines Texts handelt, " +
            "ist der Wert None."
        ),
        default=None
    )
    children: Optional[List[RSTNode]] = Field(
        description=str(
            "Die Liste der untergeordneten Knoten in der RST-Baumstruktur. " +
            "Falls der Knoten ein EDU ist, hat er keine untergeordneten " +
            "Knoten und der Wert ist None. Falls der Knoten eine RST-Relation " +
            "ist, enthält die Liste die Knoten, die durch diese Relation " +
            "verbunden werden."
        )
    )

    # TODO: Werte validieren
    @model_validator(mode="after")
    def validate_node(self):
        return self


class RSTRelation(BaseModel, use_enum_values=True):
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

    nuclei: List[Union[EDU, RSTRelation]] = Field(
        description=str(
            "Die Knoten, die durch die RST-Relation verbunden werden. Falls " +
            "es sich um eine mononukleare Relation handelt, muss genau " +
            "ein Knoten in dieser Liste sein, der den Nukleus bildet. " +
            "Falls es sich um eine multinukleare Relation handelt, müssen " +
            "mindestens zwei Nuklei in dieser Liste sein."
        )
    )
    satellite: Optional[Union[EDU, RSTRelation]] = Field(
        description=str(
            "Der Knoten, der als Satellit einer mononuklearen Relation " +
            "dient. Falls es sich um eine multinukleare Relation handelt, " +
            "gibt es keinen Satelliten und dieser Wert ist None."
        ),
        default=None
    )
    relation: Union[Mononuclear, Multinuclear] = Field(
        description=str(
            "Die RST-Relation, die zwei oder mehr RST-Knoten miteinander " +
            "verbindet. Für mononukleare Relationen müssen genau zwei Knoten" +
            " verbunden werden, für multinukleare Relationen mindestens zwei."
        )
    )

    @computed_field
    @property
    def type(self) -> Literal["span", "multinuc"]:
        if self.relation in Multinuclear:
            return "multinuc"
        return "span"

    @model_validator(mode="after")
    def check_nodes(self):
        if self.type == "span":
            if len(self.nuclei) != 1:
                raise ValueError(
                    "Mononukleare Relationen müssen genau einen Nukleus haben."
                )
            if self.satellite is None:
                raise ValueError(
                    "Mononukleare Relationen müssen einen Satelliten haben."
                )
        else:
            if len(self.nuclei) < 2:
                raise ValueError(
                    "Multinukleare Relationen müssen mindestens zwei Nuklei haben."
                )
            if self.satellite is not None:
                raise ValueError(
                    "Multinukleare Relationen dürfen keinen Satelliten haben."
                )


class EDU(BaseModel):
    """
    EDUs (Elementary Discourse Unit) sind die minimalen Diskurseinheiten auf
    Satz- bzw. Phrasen-Ebene und bilden damit die granularste Ebene von
    RST-Bäumen (RSTTree).
    EDUs verbinden sich miteinander und bilden weitere, komplexe
    Diskurseinheiten (RSTNode).
    """

    text: str = Field(
        description="Der korrespondierende Text-Ausschnitt der EDU."
    )
    start: int = Field(
        description=str(
            "Index des ersten Tokens des EDU.text innerhalb des Dokuments."
        )
    )
    end: int = Field(
        description=str(
            "Folge-Index des letzten Tokens des EDU.text innerhalb des " +
            "Dokuments."
        )
    )

    @model_validator(mode="after")
    def check_indices(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                "Ungültige Start- und End-Indizes für EDU: " +
                f"start={self.start}, end={self.end}."
            )
        return self


# NOTE: ggf. Beispiele für Segmentierungs-Schritte hinzufügen
class Segmentation(BaseModel):
    """
    Ziel der Segmentierung ist es, den Text in eine lineare Folge von EDUs zu
    partitionieren.
    
    ## Grundregeln der Segmentierung
    Die Segmentierung geschieht auf Basis der folgenden Grundregeln:
    - Eine EDU entspricht einer erkennbaren, selbstständigen Sprechhandlung
        (Illokution). Diese muss aber nicht im engen Sinne strukturell
        „vollständig“ sein: Etwaige Elisionen sind bei der Beurteilung
        aufzufüllen, anaphorische/ kataphorische Verweise durch ihre
        Antezedenten zu ersetzen.
        - Ausnahme: Parenthetische Einschübe, die in der Mitte eines Segments
            stehen, werden auch dann nicht als EDU abgegrenzt, wenn sie
            eigentlich eine Illokution darstellen.
    - Eine EDU muss ein Verb enthalten.
        - Ausnahme: Präpositionalphrasen werden als eigene EDU abgegrenzt,
            wenn folgende Bedingungen erfüllt sind:
            - Die eingebettete Nominalphrase ein Substantiv enthält, mit dem
                auf einen Sachverhalt referiert wird (z. B., aber nicht
                unbedingt, ein nominalisiertes Verb).
            - Die Präpositionalphrase steht in einer klar erkennbaren
                Kohärenzrelation zum umgebenden Haupt- bzw. Nebensatz.
    - EDUs dürfen sich nicht überschneiden.
    - Eine EDU erstreckt sich über einen vollständigen Satz oder einen
        satzwertigen/ phrasalen Teil eines Satzes. D.h. EDUs erstrecken sich
        nicht über Satzgrenzen hinweg.
    
    ## Unterscheidung nach strukturellen Typen
    ### Strukturelle Typen
    - Hauptsatz
        - vollständig (HS)
        - unvollständig (HSF)
    - Nebensatz
        - Satzgliederweiterung:
            - Subjektsatz (SUB)
            - Objektsatz (OBJ)
            - Adverbialsatz (ADV)
            - Prädikativsatz (PRD)
        - Attributsatz:
            - restr. Relativsatz (ARR)
            - nichtrestr. Relativsatz (ANR)
            - Partizipialkonstruktion (AKP)
            - sonstige (ATT)
        - weiterführender Nebensatz (WEI): nimmt Bezug nicht auf ein einzelnes
            Element des vorangehenden (Teil-)Satzes, sondern auf dessen gesamte
            Aussage.
    - Fragment
        - einleitend (FRE)
        - beendend (FRB)

    Die Grundregeln werden für die unterschiedlichen Segmenttypen wie folgt
    umgesetzt:
        - Hauptsätze (HS) und -fragmente (HSF) bilden stets eine eigenständige
            EDU. (HSF stellen dabei eine gewisse „Grauzone“ für die Grundregel
            dar, da hier das Problem der Elision auftritt.)
        - Fragmente: FRE & FRB sind auf den Status einer selbstständigen
            Illokution hin zu überprüfen und dann entweder als eigene EDU
            abzugrenzen oder dem benachbarten HS/HSF anzufügen.
        - Nebensätze:
            - SUB, OBJ, PRD bilden nur dann eine eigenständige EDU, wenn der
                Autor darin eine eingebettete Proposition persönlich bewertet
                bzw. beurteilt
            - im Fall der Reihung zweier/ mehrerer koordinierte SUB, OBR oder
                PRD (mit oder ohne Konjunktion) müssen auch zwei/ mehrere EDUs
                geschaffen werden
            - ARR und ATT dienen der näheren Charakterisierung eines
                Diskursreferenten, übermitteln also keine eigenständige
                Informationseinheit und bilden daher keine EDU.
            - ANR und WEI kommunizieren selbstständige Informationseinheiten
                und bilden eine EDU.
            - ADV stehen in einer durch einen Konnektor markierten inhaltlichen
                Relation zum übergeordneten HS/HSF, damit eine eigenständige
                Informationseinheit und bilden eine EDU.
            - AKP müssen im Einzelfall anhand der Grundregel beurteilt werden.

    ## Beispiel-Segmentierung
    Die Nummern im nachfolgenden Text kennzeichnen EDUs.

    (1) Die Lausitzer Braunkohle AG ( Laubag ) hat im abgelaufenen
    Geschäftsjahr mit der Förderung und dem Verkauf von Braunkohle
    erstmals Verluste gemacht. (2) Damit wird erneut deutlich,
    dass eine Neuordnung der Energiewirtschaft in Ostdeutschland
    überfällig ist. (3) Denn die Laubag und ihr wichtigster Kunde,
    die Veag, hängen wie siamesische Zwillinge voneinander ab.
    (4) Als die Veag wegen der Liberalisierung des Strommarktes
    unter Druck geriet und ihre Strompreise senken musste, (5) hielt
    sie sich bei ihrem Lieferanten schadlos. (6) Die Laubag musste
    Preiszugeständnisse machen, (7) die sie nun selbst in Bedrängnis
    bringen. (8) Einziger Weg aus dem Dilemma ist die Bildung eines
    neuen Energiekonzerns aus Rohstofflieferanten, Stromerzeugern
    und Endversorgern, (9) in dem Risiken besser verteilt werden
    können. (10) Die Idee, auch einen Gasversorger mit ins Boot zu
    holen, ist im Prinzip nicht schlecht. (11) Allerdings gilt auch hier,
    dass zu viele Köche den Brei verderben können. (12) Zumindest
    muss klar sein, wer im neuen Konzern das Sagen hat.
    """

    edus: List[EDU] = Field(
        description=str(
            "Die Liste der EDU-Objekte, die die Segmentierung des Textes " +
            "in EDUs repräsentieren. Die EDUs sind in der Reihenfolge " +
            "ihres Auftretens im Text sortiert."
        )
    )

    @field_validator("edus")
    @classmethod
    def check_edus(cls, edus: List[EDU]) -> List[EDU]:
        sorted_edus = sorted(edus, key=lambda edu: edu.start)
        for i in range(1, len(sorted_edus)):
            if sorted_edus[i].start <= sorted_edus[i - 1].end:
                raise ValueError("EDUs dürfen sich nicht überschneiden.")
        return sorted_edus


# FIXME: Ausgabe gibt keinen Baum, sondern nur EDU-Segmentierung zurück
class RSTTree(BaseModel):
    """
    Die Rhetorical Structure Theory geht davon aus, dass ein kohärenter Text
    durch eine Baumstruktur charakterisiert werden kann, deren Blätter die
    minimalen Diskurseinheiten (EDUs) sind und deren interne Knoten (RSTNodes)
    durch je eine Kohärenzrelation bezeichnet sind. Für die meisten dieser
    Relationen gilt, dass die beiden miteinander verbundenen Einheiten nicht
    „gleichgewichtig“ sind, sondern eines jeweils die zentrale Funktion erfüllt
    und das andere eine nurmehr unterstützende Rolle spielt. Erstere Einheit
    heißt dann Nukleus, letztere Satellit.

    Der Baum (RSTTree) ist also die Rekonstruktion der Absicht des Autors aus
    Sicht des Lesers. RST ist damit ein Instrument zur Explizierung des
    Textverständnisses: Die Baumstruktur macht klar, was nach Ansicht des
    Lesers der Autor mit dem Text intendiert hat.

    ## Aufgabe
    Die eigentliche RST-Annotation besteht aus drei Teilaufgaben:
        • Festlegen der hierarchischen Struktur des Texts: Welche benachbarten
            Abschnitte sind miteinander zu verbinden - wie entsteht daraus ein
            Baum, der den Text komplett überspannt?
        • Festlegen der inhaltlichen Relation bei jeder Verbindung zweier
            Segmente.
        • Festlegen des Nukleus/Satellit-Status der verbundenen Segmente. Dies
            ergibt sich bei den meisten Relationen automatisch durch die Wahl
            der Relation. Umgekehrt kann aber auch der Wunsch nach einer
            angemessenen Nukleus-Zuweisung die Menge der passenden Relationen
            einschränken.

    Beginnen Sie mit der Annotation erst, nachdem Sie den gesamten Text
    gründlich gelesen und den Verlauf seiner Argumentation verstanden haben.
    Dann gehen Sie in den nachfolgend beschriebenen Schritten vor:
        1. Zerfällt der Text in erkennbare thematische Einheiten? Markieren Sie
            etwaige Grenzen zwischen solchen Segmenten, die verschiedene
            Aspekte des Themas des Kommentars behandeln. Diese Grenzen werden
            später Anhaltspunkte für Grenzen auch im RST-Baum sein.
        2. Entscheiden Sie, welche EDUs für den Text zentrale Rollen spielen,
            am Ende also im Baum stark-nuklear sein sollten. Wenn eine einzelne
            EDU als „Kernaussage“ klar heraussticht, markieren Sie diese sowie
            andere, ebenfalls besonders wichtige EDUs. Um das Ergebnis zu
            überprüfen, machen Sie dann den „Paraphrasentest“: Reihen Sie die
            markierten EDUs aneinander (unter Löschung etwaiger unangebundener
            Konnektoren und Ersetzung von Anaphern durch ihre Antezedenten) und
            beurteilen Sie, ob der entstehende Kurztext eine adäquate
            Zusammenfassung des Texts ist. Wenn nötig, überarbeiten Sie die
            Nukleus-Auswahlen.
        3. Untersuchen Sie der Reihe nach jede EDU und ihre unmittelbaren
            Nachbarn. Gibt es eine klar erkennbare Relation zwischen zwei
            benachbarten EDUs? Dies ist insbesondere bei den meisten
            Haupt-/Nebensatz Verbindungen der Fall, und mitunter auch, wenn
            zwei syntaktisch unabhängige EDUs durch einen Konnektor verbunden
            sind. Für zu verbindende Nachbar-EDUs:
                • Entscheiden Sie, ob eines der beiden Segmente für den Text
                    ein größeres Gewicht besitzt als das andere, oder ob beide
                    gleichberechtigt nebeneinander stehen.
                • Entscheiden Sie, welche Relation zwischen den beiden EDUs
                    besteht. Dies wird durch die vorher getroffene
                    Nuklearitätsentscheidung mit beeinflusst.
                • Im Falle der multinuklearen Relationen können auch mehr als
                    zwei nebeneinander stehende EDUs verbunden werden; prüfen
                    Sie, ob dies zutrifft.
                • Notieren Sie die gefundene Relation.
            4. Nachdem alle benachbarten EDU-Paare untersucht sind, fahren Sie
                mit der Betrachtung größerer Einheiten fort: Ein Konnektor kann
                auch mehr als eine EDU als Segment nehmen, und natürlich können
                Relationen sowohl zwischen EDUs als auch zwischen größeren
                Abschnitten nicht durch Konnektoren angezeigt sein.
                Identifizieren Sie zunächst die leicht erkennbaren Zusammenhänge
                zwischen solchen größeren Abschnitten. Wenn Sie in Schritt 1
                thematische Grenzen markiert haben, ziehen Sie diese als
                Segmentgrenzen in Betracht. Bei der Zuweisung der Nuklearität
                kann nun die Rolle des Nukleus für den Gesamttext (Schritt 2)
                eine wichtige Rolle spielen – Ihre Entscheidungen zur
                Relationszuweisung an größere Segmente sollen berücksichtigen,
                dass am Ende die gewünschten EDUs streng-nuklear sind.
                Hilfreich für die Relationsentscheidung ist oft der Test, einen
                Konnektor einzufügen; z. B. bei der Probe, ob eine Begründung
                vorliegt, dem begründeten Segment ein „Also:“ voranzustellen.
                Generell empfiehlt es sich, dass Sie Ihre Hypothesen zunächst
                provisorisch notieren und dann prüfen, ob sie sich jeweils in
                eine adäquate Baumstruktur einfügen würden.
            5. Wenn Sie sich auf die Relationen zwischen größeren Abschnitten
                festgelegt haben, markieren Sie diese. Dabei sollten Sie
                tendenziell bottom-up vorgehen, also immer nur benachbarte
                Segmente (EDUs oder größere) verbinden und so sukzessive den
                Baum von unten nach oben aufbauen. Dementsprechend wichtig ist
                es für komplexe Texte, den Baum zuvor als Ganzes zu skizzieren.
    """

    segmentation: Segmentation = Field(
        description="Die Segmentierung des Textes in EDUs."
    )
    discourse_tree: RSTNode = Field(
        description=str(
            "Die Wurzel des RST-Baums, der die hierarchische Struktur " +
            "der EDUs und ihre Beziehungen zueinander repräsentiert."
        )
    )
