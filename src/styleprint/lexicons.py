"""Language-specific resources for the style analysis, critic and style card.

Markers are (label, regex). Patterns run on the lowercased utterance with its
punctuation kept, so position can disambiguate: French "bon" is a discourse marker
at the start of an utterance or after a comma ("bon, on y va"), not in "c'est bon".
`^` and `$` anchor to the utterance.

Each language provides:
    spacy_model       parser used for POS, morphology and dependencies
    markers           {category: [(label, regex)]}; category "channel" is measured but never prescribed
    familiar          regexes for familiar-register words
    negation          how informal negation looks: "fr_ne" (dropping "ne") or "en_contraction" (n't)
    preferences       pairs where a speaker may strongly favour one form: on/nous, yeah/yes...
    address           regex for words addressing the listener
    first_person      the main first-person pronoun form, for "talks about themself a lot"
    pronouns          token forms reported in the profile, compared with general usage
    tags              regex for check-in tag questions at the end of an utterance
    imperative        rule used to detect commands: "fr_endings" or "en_base"
    passive           rule used to detect passive voice: "fr" or "en"
    emphatic          words whose doubling is emphasis ("très très"), not hesitation
    repeat_ok         words that legitimately repeat ("vous vous lavez", "had had")
    grammar_words     n-grams made only of these are grammar, not signature phrases
    formal_connectors listed in the style card as things the speaker would not say
    french_spacing    space before ? and ! when joining transcript words
    text              user-facing strings for the card and the critic
"""

from __future__ import annotations

W = r"(?<![\w'’])"  # word start
E = r"(?![\w'’])"  # word end
INITIAL = r"(?:^|[,;:] )"  # start of utterance or after a comma
FINAL = r"[ ,.!…]*$"  # end of utterance, not a question


def w(*words: str) -> str:
    return W + "(?:" + "|".join(words) + ")" + E


# --- French ------------------------------------------------------------------

FR_MARKERS = {
    "discourse_markers": [
        ("voilà", w("voilà", "voila")),
        ("alors", w("alors")),
        ("donc", w("donc")),
        ("bon", INITIAL + w("bon") + r"(?=[,.!…]|$| (?:voilà|alors|ben|bah|eh|écoutez|on|je|j'|là|ok|allez|après|ça|maintenant|donc|et)\b)"),
        ("allez", INITIAL + w("allez")),
        ("du coup", w("du coup")),
        ("en fait", w("en fait")),
        ("bref", w("bref")),
        ("ben / bah", w("ben", "bah", "eh ben", "eh bien")),
        ("hein", w("hein")),
        ("quoi (final)", w("quoi") + FINAL),
        ("ok / d'accord", w("ok", "okay", "d'accord")),
        ("ouais", w("ouais")),
        ("attention", w("attention")),
        ("là (deictic)", w("là")),
    ],
    "transitions": [
        ("ensuite", w("ensuite")),
        ("puis", w("puis", "et puis")),
        ("après (initial)", INITIAL + "(?:et |donc )?" + w("après")),
        ("maintenant", w("maintenant")),
        ("d'abord", w("d'abord", "tout d'abord", "premièrement")),
        ("et là", w("et là", "et puis là")),
        ("une fois que", w("une fois que", "une fois qu'")),
        ("pendant ce temps", w("pendant ce temps", "pendant que")),
        ("pour finir", w("pour finir", "pour terminer", "pour conclure", "dernière étape")),
    ],
    "hedging": [
        ("un peu", w("un peu", "un petit peu")),
        ("à peu près / environ / plus ou moins", w("à peu près", "environ", "plus ou moins", "grosso modo")),
        ("peut-être", w("peut-être", "peut être")),
        ("je pense / je crois / je trouve", w("je pense", "je crois", "je trouve")),
        ("on va dire", w("on va dire", "disons")),
        ("normalement / en général", w("normalement", "en général", "généralement")),
        ("si vous voulez / préférez / aimez", w("si vous voulez", "si vous préférez", "si vous aimez", "comme vous voulez")),
        ("genre / une sorte de / une espèce de", w("genre", "une sorte de", "une espèce de")),
    ],
    "intensifiers": [
        ("vraiment", w("vraiment")),
        ("très", w("très")),
        ("bien (+adj)", w("bien") + r" (?=\w+(?:é|ée|és|ées|ant|ante|ants|antes|eux|euse|if|ive|ible|able|ide)\b)"),
        ("super / hyper / ultra", w("super", "hyper", "ultra")),
        ("extrêmement / énormément / tellement", w("extrêmement", "énormément", "tellement")),
        ("absolument / complètement / parfaitement", w("absolument", "complètement", "parfaitement", "totalement")),
        ("carrément / vachement", w("carrément", "vachement")),
        ("trop", w("trop")),
    ],
    "contrast_qualification": [
        ("mais", w("mais")),
        ("par contre / en revanche", w("par contre", "en revanche")),
        ("quand même", w("quand même")),
        ("sauf que", w("sauf que", "sauf qu'", "sauf si")),
        ("même si", w("même si", "même s'")),
        ("pourtant / cependant", w("pourtant", "cependant", "néanmoins", "toutefois")),
        ("c'est-à-dire", w("c'est-à-dire", "c'est à dire")),
    ],
    "audience_framing": [
        ("vous voyez / tu vois", w("vous voyez", "tu vois", "vous avez vu", "t'as vu", "tu as vu")),
        ("regardez / regarde", w("regardez", "regarde", "voyez")),
        ("je vous montre / explique", w("je vous montre", "je vais vous montrer", "je vous explique", "je vais vous expliquer")),
        ("je vous conseille / recommande", w("je vous conseille", "je vous recommande", "je vous invite")),
        ("vous allez / pouvez / verrez", w("vous allez", "vous pouvez", "vous pourrez", "vous verrez")),
        ("n'hésitez pas", w("n'hésitez pas")),
        ("on va (inclusive)", w("on va")),
        ("c'est parti / on y va / hop", w("c'est parti", "on y va", "allez hop", "et hop", "hop")),
    ],
    "repair": [
        ("enfin (repair)", r"\w " + w("enfin") + r"(?!,? (?:bref|voilà))"),
        ("je veux dire / pardon / ou plutôt", w("je veux dire", "pardon", "ou plutôt", "non mais")),
    ],
    "evaluative": [
        ("magnifique / superbe / sublime", w("magnifique", "superbe", "sublime", "splendide")),
        ("extraordinaire / incroyable", w("extraordinaire", "incroyable", "exceptionnel", "exceptionnelle")),
        ("génial / top / nickel / parfait", w("génial", "géniale", "top", "nickel", "excellent", "excellente", "parfait", "parfaite")),
        ("délicieux / succulent / une tuerie", w("délicieux", "délicieuse", "succulent", "succulente", "un régal", "une tuerie", "une merveille")),
        ("c'est (très) bon", w("c'est bon", "c'est très bon", "trop bon", "très bon")),
        ("wow", w("wow", "waouh", "whaou", "oh là là", "ouh là")),
    ],
    "channel": [
        ("abonnez-vous / like / commentaires", w(r"abonn\w*", "like", "pouce", "commentaire", "commentaires", "la cloche")),
    ],
}

FR = {
    "name": "French",
    "spacy_model": "fr_core_news_md",
    "markers": FR_MARKERS,
    "familiar": [w(x) for x in (
        "ouais", "bah", "ben", "truc", "trucs", "machin", "super", "génial", "nickel", "top", "tuerie", "sympa",
        "hyper", "vachement", "bouffe", "mec", "pote", "boulot", "galère", "cool", "kiffer", "kif", "ouf", "dingue",
        "chiant", "bordel", "putain", "fringues", "flic", "fric")],
    "negation": "fr_ne",
    "negation_words": {"pas", "jamais", "rien", "personne", "aucun", "aucune", "guère"},
    "ne": {"ne", "n'", "n’"},
    "preferences": [
        {"use": "on", "use_re": w("on"), "avoid": "nous", "avoid_re": w("nous")},
        {"use": "ça", "use_re": w("ça", "ç'"), "avoid": "cela", "avoid_re": w("cela")},
        {"use": "vous", "use_re": w("vous"), "avoid": "tu", "avoid_re": w("tu", "t'", "toi")},
        {"use": "tu", "use_re": w("tu", "t'", "toi"), "avoid": "vous", "avoid_re": w("vous")},
    ],
    "address": w("vous", "votre", "vos", "tu", "t'", "toi", "ton", "ta", "tes"),
    "first_person": "je",
    "pronouns": ["je", "j'", "moi", "tu", "t'", "toi", "vous", "on", "nous", "il", "elle", "ils", "elles",
                 "ça", "ç'", "cela", "ce", "c'"],
    "tags": r"(?:d'accord|ok|okay|hein|non|vous voyez|tu vois|n'est-ce pas|voilà|oui)\s*\?\s*$",
    "imperative": "fr_endings",
    "imperative_endings": ("ez", "ons"),
    # Mostly indicative in practice ("si vous avez", "vous savez"); their imperatives are rare or differ (sachez).
    "imperative_stop": {"chez", "assez", "nez", "rez", "avez", "êtes", "savez", "voulez", "pouvez", "devez",
                        "avons", "sommes", "savons", "voulons", "pouvons", "devons"},
    "imperative_after": {"et", "puis", "alors", "donc", "allez", "bon", "n'", "ne"},
    "imperative_display": {"allez": "Allez", "écoutez": "Écoutez", "regardez": "Regardez", "attendez": "Attendez",
                           "hésitez": "N'hésitez pas", "imaginez": "Imaginez", "pensez": "Pensez à…"},
    "passive": "fr",
    "emphatic": {"très", "vraiment", "bien", "trop", "non", "oui", "si", "tout", "beaucoup"},
    "repeat_ok": {"vous", "nous"},  # vous vous lavez
    "grammar_words": {"il", "y", "a", "que", "qu'", "j'", "ai", "et", "c'", "c'est", "est", "avec", "de", "l'",
                      "vous", "êtes", "ce", "si", "avez", "la", "le", "les", "un", "une", "à", "en", "je", "on",
                      "ça", "des", "du", "d'", "qui"},
    "formal_connectors": ["cependant", "néanmoins", "en outre", "ainsi", "par conséquent"],
    "french_spacing": True,
    "text": {
        "negation_card": "Drop the *ne* of negation {freq} (« c'est pas », « faut pas »), keep it otherwise.",
        "negation_too_formal": "Too formal: drop the « ne » in some negations (« c'est pas », « faut pas »).",
        "negation_too_casual": "Too casual: keep the « ne » in most negations.",
        "negation_label": "negations keeping *ne*",
        "negation_written": "written French: ~100%",
        "chain_words": "« et », « donc », « là »",
        "fragment_example": "« Voilà. »",
    },
}

# --- English -----------------------------------------------------------------

EN_MARKERS = {
    "discourse_markers": [
        ("so (initial)", INITIAL + w("so") + r"(?=[, ]|$)"),
        ("well (initial)", INITIAL + w("well") + r"(?=[,.!]|$)"),
        ("you know", w("you know") + r"(?! (?:what|how|why|that|the|a|it)\b)"),
        ("I mean", w("i mean")),
        ("like (filler)", r"(?:,|^) ?" + w("like") + r",? "),
        ("actually", w("actually")),
        ("basically / literally", w("basically", "literally")),
        ("okay / alright", w("ok", "okay", "alright", "all right")),
        ("right (tag)", w("right") + r" ?[?,]"),
        ("now (initial)", INITIAL + w("now") + r"(?=,| )"),
        ("look / listen", INITIAL + w("look", "listen") + r"(?=[,.!]| )"),
        ("anyway", w("anyway", "anyways")),
        ("oh", w("oh")),
        ("yeah", w("yeah", "yep", "yup")),
    ],
    "transitions": [
        ("then / and then", w("then", "and then")),
        ("next", w("next")),
        ("after that", w("after that", "afterwards")),
        ("first", INITIAL + w("first", "firstly", "first of all")),
        ("finally", w("finally", "lastly", "in the end")),
        ("meanwhile", w("meanwhile", "in the meantime")),
        ("once", INITIAL + w("once")),
    ],
    "hedging": [
        ("kind of / sort of", w("kind of", "sort of", "kinda", "sorta")),
        ("a bit / a little", w("a bit", "a little", "a little bit")),
        ("maybe / perhaps / probably", w("maybe", "perhaps", "probably")),
        ("I think / I guess", w("i think", "i guess", "i feel like", "i suppose", "i reckon")),
        ("pretty much / more or less", w("pretty much", "more or less", "roughly")),
        ("if you want", w("if you want", "if you like", "if you prefer")),
    ],
    "intensifiers": [
        ("really", w("really")),
        ("very", w("very")),
        ("so (+adj)", w("so") + r" (?=\w+(?:ful|ous|ive|able|ible|ing|ed|y|ic|al)\b)"),
        ("super / totally", w("super", "totally", "utterly")),
        ("absolutely / completely", w("absolutely", "completely", "perfectly", "entirely")),
        ("extremely / incredibly", w("extremely", "incredibly", "hugely", "massively")),
        ("too", w("too") + r"(?! ?[.,!?]|$)"),
    ],
    "contrast_qualification": [
        ("but", w("but")),
        ("however", w("however", "nevertheless", "nonetheless")),
        ("though / although", w("though", "although", "even though")),
        ("still (initial)", INITIAL + w("still")),
        ("on the other hand", w("on the other hand", "whereas", "instead")),
        ("that said", w("that said", "having said that", "mind you")),
    ],
    "audience_framing": [
        ("you see / you'll see", w("you see", "you'll see", "you can see")),
        ("let me show you", w("let me show you", "i'll show you", "i'm gonna show you", "i'm going to show you", "let me explain")),
        ("you can / you could", w("you can", "you could", "you might", "you're gonna", "you're going to")),
        ("let's", w("let's", "let us")),
        ("make sure / don't forget", w("make sure", "don't forget", "remember")),
        ("trust me / believe me", w("trust me", "believe me", "i promise")),
    ],
    "repair": [
        ("sorry / or rather", w("or rather", "sorry", "what i mean is")),
    ],
    "evaluative": [
        ("amazing / incredible", w("amazing", "incredible", "unbelievable", "insane")),
        ("great / fantastic / brilliant", w("great", "fantastic", "brilliant", "excellent", "awesome", "terrific")),
        ("beautiful / lovely / gorgeous", w("beautiful", "lovely", "gorgeous", "stunning")),
        ("perfect", w("perfect", "spot on")),
        ("delicious / tasty", w("delicious", "tasty", "yummy")),
        ("wow", w("wow", "whoa", "oh my god", "oh my gosh")),
    ],
    "channel": [
        ("subscribe / like / comment", w(r"subscrib\w*", "like this video", "hit the bell", "comment below", "the comments")),
    ],
}

EN = {
    "name": "English",
    "spacy_model": "en_core_web_md",
    "markers": EN_MARKERS,
    "familiar": [w(x) for x in (
        "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "nope", "stuff", "guys", "awesome", "cool",
        "dude", "ain't", "y'all", "folks", "mate", "bloody", "crap", "freaking", "cos", "lots of",
        "a bunch of", "loads of")],
    "negation": "en_contraction",
    "preferences": [
        {"use": "yeah", "use_re": w("yeah", "yep", "yup"), "avoid": "yes", "avoid_re": w("yes")},
        {"use": "gonna", "use_re": w("gonna"), "avoid": "going to", "avoid_re": w("going to") + r"(?! the| a\b| my| your)"},
        {"use": "wanna", "use_re": w("wanna"), "avoid": "want to", "avoid_re": w("want to")},
        {"use": "I'm", "use_re": w("i'm"), "avoid": "I am", "avoid_re": w("i am")},
        {"use": "it's", "use_re": w("it's"), "avoid": "it is", "avoid_re": w("it is")},
        {"use": "you", "use_re": w("you"), "avoid": "one (generic)",
         "avoid_re": w("one") + r" (?=can|should|must|might|would|could|has|needs)"},
    ],
    "address": w("you", "your", "yours", "yourself", "yourselves", "you're", "you'll", "you've", "you'd"),
    "first_person": "i",
    "pronouns": ["i", "me", "my", "you", "your", "we", "us", "our", "he", "she", "they", "it", "this", "that", "one"],
    "tags": r"(?:right|okay|ok|yeah|you know|isn't it|aren't you|don't you|huh|eh)\s*\?\s*$",
    "imperative": "en_base",
    "imperative_after": {"and", "so", "now", "just", "then", "please", "okay", "ok", "alright", "well", "oh",
                         "first", "next", "but"},
    "imperative_display": {},
    "passive": "en",
    "emphatic": {"very", "really", "so", "too", "no", "yes", "much", "more", "never", "way"},
    "repeat_ok": {"that", "had", "is"},  # "that that", "had had", "what it is is"
    "grammar_words": {"i", "you", "it", "is", "it's", "the", "a", "an", "and", "of", "to", "in", "that", "this",
                      "we", "be", "have", "do", "don't", "i'm", "you're", "on", "for", "with", "so", "but", "what",
                      "there", "there's", "if", "can", "just", "are", "was", "at", "'s", "'m", "'re", "n't"},
    "formal_connectors": ["furthermore", "moreover", "therefore", "consequently", "thus", "hence", "whereby"],
    "french_spacing": False,
    "text": {
        "negation_card": "Contract negations {freq} (*don't*, *isn't*, *can't* rather than *do not*).",
        "negation_too_formal": "Too formal: contract negations the way they do (don't, isn't, can't, won't).",
        "negation_too_casual": "Too casual: they spell out « not » more often than this.",
        "negation_label": "negations written out as *not*",
        "negation_written": "formal writing: high",
        "chain_words": "\"and\", \"so\", \"then\"",
        "fragment_example": "\"Right.\"",
    },
}

LEXICONS = {"fr": FR, "en": EN}
