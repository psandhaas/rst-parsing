from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, Field, computed_field
from typing import Dict, List, Union


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

    # WICHTIG
    Der Text des Dokuments darf **nicht** verändert werden! D.h. der
    ursprüngliche Text muss sich lückenlos aus der Liste der generierten EDUs
    wieder zusammensetzen lassen. Z.B.:
        document = '''Dies ist ein Beispieltext, der der Illustration dient.
        Dies ist wichtig, um Missverständnisse zu vermeiden.'''
        edus = [
            EDU(text="Dies ist ein Beispieltext,", start=0, end=6),
            EDU(text="der der Illustration dient.", start=6, end=11),
            EDU(text="Dies ist wichtig,", start=11, end=15),
            EDU(text="um Missverständnisse zu vermeiden.", start=15, end=20)
        ]
        assert document == ''.join(edu.text for edu in edus)
    """

    document: Union[str, None] = Field(
        description=str(
            "Der unsegmentierte Text, der in EDUs partitioniert wurde."
        ),
        exclude=True,
        default=None
    )
    edus_: List[EDU] = Field(
        description=str(
            "Die EDU-Objekte, die die Segmentierung des Dokuments " +
            "in EDUs repräsentieren. Die EDUs sind in der Reihenfolge " +
            "ihres Auftretens im Text sortiert."
        ),
        exclude=True
    )

    @computed_field
    @property
    def edus(self) -> Dict[int, EDU]:
        """Segmentierte EDUs, indiziert nach ihrer Reihenfolge im Text."""
        return {i + 1: edu for i, edu in enumerate(self.edus_)}
