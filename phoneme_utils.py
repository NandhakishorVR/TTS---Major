import csv
import re
import inflect
import nltk
from nltk.corpus import cmudict
from torchtext.vocab import build_vocab_from_iterator

# Load CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Number converter
num_engine = inflect.engine()

# Load custom phoneme dataset from CSV
def load_phoneme_dataset(file_path="phoneme_dataset.csv"):
    phoneme_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header if exists
            for row in reader:
                if len(row) == 2:
                    word, phoneme = row
                    phoneme_dict[word.lower()] = phoneme
    except FileNotFoundError:
        print("")
    return phoneme_dict

custom_phoneme_dict = load_phoneme_dataset()

# Expanded abbreviations
ABBREVIATIONS = {
    "dr.": "doctor",
    "mr.": "mister",
    "mrs.": "misses",
    "ph.d.": "doctor of philosophy",
    "u.s.": "united states",
    "it's": "it is",
    "you're": "you are",
    "they're": "they are",
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "vs.": "versus",
    "jr.": "junior",
    "sr.": "senior",
    "prof.": "professor",
    "inc.": "incorporated",
    "dept.": "department",
    "est.": "established",
    "a.k.a.": "also known as",
    "b.c.": "before christ",
    "a.d.": "anno domini",
    "r.i.p.": "rest in peace",
    "approx.": "approximately",
    "appt.": "appointment",
    "asap": "as soon as possible",
    "min.": "minute",
    "sec.": "second",
    "hr.": "hour",
    "w/": "with",
    "w/o": "without",
    "gov.": "governor",
    "gen.": "general",
    "rep.": "representative",
    "sen.": "senator",
    "lt.": "lieutenant",
    "col.": "colonel",
    "cmdr.": "commander",
    "maj.": "major",
    "no.": "number",
    "rev.": "reverend",
    "st.": "saint",
    "b.y.o.b.": "bring your own bottle",
    "c/o": "care of",
    "misc.": "miscellaneous",
    "univ.": "university",
    "yr.": "year",
    "vol.": "volume",
    "lib.": "library",
    "bros.": "brothers",
    "corp.": "corporation",
    "intl.": "international",
    "adj.": "adjective",
    "adv.": "adverb",
    "pp.": "pages",
    "pg.": "page",
    "dept.": "department",
    "a.m.": "ante meridiem",
    "p.m.": "post meridiem",
    "c.v.": "curriculum vitae",
    "t.b.d.": "to be decided",
    "t.b.c.": "to be continued",
    "e.o.m.": "end of message",
    "e.t.a.": "estimated time of arrival",
    "f.y.i.": "for your information",
    "r.s.v.p.": "respondez s'il vous plait",
    "b.c.e.": "before common era",
    "c.e.": "common era",
    "u.k.": "united kingdom",
    "e.u.": "european union",
    "n.b.": "nota bene",
    "a.d.c.": "aide-de-camp",
    "f.a.o.": "for the attention of",
    "f.t.w.": "for the win",
    "q.e.d.": "quod erat demonstrandum",
    "etcetera": "and so on",
    "g.m.t.": "greenwich mean time",
    "a.i.": "artificial intelligence",
    "f.b.i.": "federal bureau of investigation",
    "c.i.a.": "central intelligence agency",
    "n.a.s.a.": "national aeronautics and space administration",
    "o.s.h.a.": "occupational safety and health administration",
    "a.c.": "alternating current",
    "d.c.": "direct current",
    "l.t.d.": "limited",
    "pvt.": "private",
    "n.y.c.": "new york city",
    "tba": "to be announced",
    "s.o.s.": "save our souls",
    "c.o.d.": "cash on delivery",
    "s.o.p.": "standard operating procedure",
    "t.i.a.": "thanks in advance",
    "c.i.f.": "cost, insurance, and freight",
    "f.o.b.": "free on board",
    "f.i.f.o.": "first in, first out",
    "l.c.d.": "least common denominator",
    "gcd": "greatest common divisor",
    "i.q.": "intelligence quotient",
    "f.w.i.w.": "for what it's worth",
    "m.i.a.": "missing in action",
    "t.b.h.": "to be honest",
    "n.d.": "no date",
    "e.d.": "electronic distribution",
    "d.i.y.": "do it yourself",
    "d.o.c.": "department of corrections",
    "i.m.o.": "in my opinion",
    "f.o.r." : "for",
    "n.y." : "new york",
    "l.a." : "los angeles",
    "d.c." : "district of columbia",
    "f.b.o." : "for the benefit of",
    "q.v." : "quod vide",
    "m.o." : "modus operandi",
    "a.s.a.p." : "as soon as possible",
    "s.t." : "something",
    "c.i.": "companion information",
    "d.a." : "district attorney",
    "p.s." : "postscript",
    "l.o.l." : "laugh out loud",
    "l.t." : "long term",
    "p.t." : "part time",
    "s.s." : "social security",
    "d.b.a." : "doing business as",
    "w.r.t." : "with respect to",
    "r.i.d." : "revenue in detail",
    "b.v." : "based on view",
    "m.m.o." : "most market opportunity",
    "r.g.": "radio group",
    "p.b." : "personal best",
    "i.g." : "instagram",
    "f.m." : "frequency modulation",
    "p.m.": "post meridiem",
    "b.a." : "bachelor of arts",
    "b.s." : "bachelor of science",
    "m.s." : "master of science",
    "m.a." : "master of arts",
    "ph.d." : "doctor of philosophy",
    "m.b.a." : "master of business administration",
    "c.f." : "compare",
    "d.c.a." : "direct current amplifier",
    "c.m.s." : "content management system",
    "i.s.o." : "in search of",
    "i.e." : "in other words",
    "b.o.m." : "bill of materials",
    "b.p." : "blood pressure",
    "s.b.a." : "small business administration",
    "s.a.e." : "society of automotive engineers",
    "o.a.r." : "of all records",
    "t.s.a." : "transportation security administration",
    "e.s.o." : "electronic switching office",
    "c.o.s." : "change of status",
    "r.d.s." : "reliable data systems",
    "g.o.c." : "general operating condition",
    "f.o.s." : "free of service",
    "r.o.i." : "return on investment",
    "t.o.s." : "terms of service",
    "d.o.s." : "denial of service",
    "l.s.d." : "lysergic acid diethylamide",
    "c.b.o." : "congressional budget office",
    "e.l.i.": "end of life indicator",
    "h.e.l.p." : "human empowerment, learning, progress",
    "e.p.o." : "electric power outlet",
    "b.a.r.": "bar association",
    "c.c.": "credit card",
    "c.d.": "compact disc",
    "f.a.s.": "free alongside ship",
    "g.a.s.": "gasoline",
    "i.h.t.": "in his time",
    "j.k.l.": "just kidding, lol",
    "k.o." : "knock out",
    "l.m.t.": "local mean time",
    "n.t.s.b.": "national transportation safety board",
    "o.t.o.": "on the other",
    "p.b.x.": "private branch exchange",
    "q.l.s.": "quality of life",
    "r.a.f.": "royal air force",
    "s.a.m.": "surface to air missile",
    "t.s.l.": "time series library",
    "u.s.s.": "united states ship",
    "v.c.": "venture capital",
    "w.c.": "water closet",
    "x.o.": "hugs and kisses",
    "y.m.c.a.": "young men's christian association",
    "z.o.": "zero",
    "a.b.a.": "american bar association",
    "d.p.": "distribution point",
    "e.f.t.": "electronic funds transfer",
    "f.s.b.": "federal savings bank"
}

# Function to clean text
def clean_text(text):
    text = text.lower()
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    text = re.sub(r"\b\d+\b", lambda x: num_engine.number_to_words(x.group()), text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Function to convert text to phonemes
def text_to_phonemes(text):
    words = text.split()
    phoneme_output = []
    for word in words:
        if word in cmu_dict:
            phoneme_output.append(" ".join(cmu_dict[word][0]))
        else:
            phoneme_output.append(" ".join(list(word)))
    return " ".join(phoneme_output)

# Create vocabulary for phoneme-to-index mapping
def phoneme_vocab_generator():
    for word in cmu_dict:
        for phoneme_seq in cmu_dict[word]:
            yield phoneme_seq

phoneme_vocab = build_vocab_from_iterator(phoneme_vocab_generator(), specials=["<unk>", "<pad>"])
phoneme_vocab.set_default_index(phoneme_vocab["<unk>"])

# Function to convert phonemes to index sequence
def phonemes_to_index(phoneme_str):
    phonemes = phoneme_str.strip().split()
    return [phoneme_vocab[phoneme] for phoneme in phonemes]

# Full pipeline function
def process_text_to_phonemes(text):
    cleaned_text = clean_text(text)
    phonemes = text_to_phonemes(cleaned_text)
    phoneme_indices = phonemes_to_index(phonemes)
    return phonemes, phoneme_indices

# Optional test block
if __name__ == "__main__":
    input_text = "hello how are you"
    phoneme_result, phoneme_indices = process_text_to_phonemes(input_text)
    print("Original Text:", input_text)
    print("Cleaned Text:", clean_text(input_text))
    print("Phonemes:", phoneme_result)
