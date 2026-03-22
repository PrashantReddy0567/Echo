import spacy
from spellchecker import SpellChecker
from transformers import pipeline
import re

# ========== ✨ Load Resources ==========
print("[🔍] Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# ========== 🔧 Utilities ==========

def correct_spelling(text):
    words = re.findall(r"\b\w+\b", text.lower())
    misspelled = spell.unknown(words)
    corrections = {}
    for word in misspelled:
        candidates = list(spell.candidates(word))
        if candidates:
            corrections[word] = candidates[0]
    return corrections

def suggest_contextual(text):
    words = text.split()
    suggestions = {}
    for i, word in enumerate(words):
        masked = words.copy()
        masked[i] = '[MASK]'
        masked_sentence = ' '.join(masked)
        preds = fill_mask(masked_sentence)

        # If top prediction doesn't match current word and has high score
        if preds[0]['token_str'].lower() != word.lower() and preds[0]['score'] > 0.4:
            suggestions[word] = preds[0]['token_str']
    return suggestions

def check_syntax(text):
    doc = nlp(text)
    issues = []

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            # Basic subject-verb agreement check
            subj = token
            verb = token.head
            if subj.tag_ in ["NN", "NNP"] and verb.tag_ not in ["VBZ"]:
                issues.append(f"⚠️ Subject '{subj.text}' may not agree with verb '{verb.text}'")
            elif subj.tag_ in ["NNS", "NNPS"] and verb.tag_ in ["VBZ"]:
                issues.append(f"⚠️ Plural subject '{subj.text}' may not agree with singular verb '{verb.text}'")
    return issues

# ========== 🎯 Main Function ==========

def analyze_text(text):
    print(f"\n📘 Analyzing: '{text}'\n")

    # Step 1: Spelling Correction
    spelling_corrections = correct_spelling(text)
    if spelling_corrections:
        print("📝 Spelling Suggestions:")
        for wrong, correct in spelling_corrections.items():
            print(f" - '{wrong}' ➜ '{correct}'")
    else:
        print("✅ No spelling mistakes detected.")

    # Step 2: Contextual Analysis
    contextual = suggest_contextual(text)
    if contextual:
        print("\n🔎 Contextual Word Suggestions:")
        for word, suggestion in contextual.items():
            print(f" - '{word}' doesn't fit well. Try '{suggestion}'")
    else:
        print("✅ All words seem contextually appropriate.")

    # Step 3: Syntactic Structure
    syntax_issues = check_syntax(text)
    if syntax_issues:
        print("\n⚙️ Syntactic Warnings:")
        for issue in syntax_issues:
            print(f" - {issue}")
    else:
        print("✅ Syntax looks fine.")

    print("\n🎉 Analysis Complete.\n")

# ========== 🧪 Example ==========

if __name__ == "__main__":
    sample = "Eye no their going too the mall"
    analyze_text(sample)
