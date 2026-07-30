"""
prepare_dataset.py
------------------
Data preparation script for Sanskrit-English Multilingual Embedding & Retrieval.
Curates aligned parallel text datasets (Sanskrit Devanagari + Roman Transliteration + English Translations)
and splits them into train and test evaluation sets in JSONL format for SentenceTransformers.
"""

import os
import json
import random
import urllib.request
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Source 1: Built-in Aligned Bhagavad Gita & Upanishad Verses (Offline / Fallback)
SAMPLE_SANSKRIT_ENGLISH_DATA = [
    {
        "id": "bg_2.47",
        "sanskrit": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
        "transliteration": "karmaṇyevādhikāraste mā phaleṣu kadācana | mā karmaphalaheturbhūrmā te saṅgo'stvakarmaṇi ||",
        "english": "You have a right to perform your prescribed duty, but at no time to its fruits. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.",
        "source": "Bhagavad Gita 2.47",
        "queries": [
            "What does Bhagavad Gita say about performing duty without attachment to results?",
            "Karma yoga concept of right to action not fruits",
            "Sanskrit verse explaining karma and duty"
        ]
    },
    {
        "id": "bg_2.20",
        "sanskrit": "न जायते म्रियते वा कदाचिन्नायं भूत्वा भविता वा न भूयः। अजो नित्यः शाश्वतोऽयं पुराणो न हन्यते हन्यमाने शरीरे॥",
        "transliteration": "na jāyate mriyate vā kadācinnāyaṃ bhūtvā bhavitā vā na भूयः | ajo nityaḥ śāśvato'yaṃ purāṇo na hanyate hanyamāne śarīre ||",
        "english": "The soul is never born nor does it die at any time. It has not come into being, does not come into being, and will not come into being. It is unborn, eternal, ever-existing, and primeval. It is not slain when the body is slain.",
        "source": "Bhagavad Gita 2.20",
        "queries": [
            "Sanskrit verse about the immortality and eternal nature of the soul",
            "Is the soul killed when the body dies according to Gita?",
            "Immortality of Atman in Hindu philosophy"
        ]
    },
    {
        "id": "bg_4.7",
        "sanskrit": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥",
        "transliteration": "yadā yadā hi dharmasya glānirbhavati bhārata | abhyutthānamadharmasya tadātmānaṃ sṛjāmyaham ||",
        "english": "Whenever and wherever there is a decline in righteousness, O descendant of Bharata, and a predominant rise of unrighteousness—at that time I manifest Myself.",
        "source": "Bhagavad Gita 4.7",
        "queries": [
            "When does God avatar manifest on Earth according to Gita?",
            "Decline of Dharma and rise of Adharma verse",
            "Yada Yada Hi Dharmasya meaning and Sanskrit text"
        ]
    },
    {
        "id": "bg_4.8",
        "sanskrit": "परित्राणाय साधूनां विनाशाय च दुष्कृताम्। धर्मसंस्थापनार्थाय सम्भवामि युगे युगे॥",
        "transliteration": "paritrāṇāya sādhūnāṃ vināśāya ca duṣkṛtām | dharmasaṃsthāpanārthāya sambhavāmi yuge yuge ||",
        "english": "To deliver the pious and to annihilate the miscreants, as well as to reestablish the principles of righteousness, I Myself appear, millennium after millennium.",
        "source": "Bhagavad Gita 4.8",
        "queries": [
            "Purpose of Divine incarnation in Hinduism",
            "Protecting the good and destroying evil across ages",
            "Paritranaya sadhunam verse Sanskrit and English translation"
        ]
    },
    {
        "id": "bg_6.5",
        "sanskrit": "उद्धरेदात्मनात्मानं नात्मानमवसादयेत्। आत्मैव ह्यात्मनो बन्धुरात्मैव रिपुरात्मनः॥",
        "transliteration": "uddharedātmanātmānaṃ nātmānamavasādayet | ātmaiva hyātmano bandhurātmaiva ripurātmanaḥ ||",
        "english": "One must elevate oneself by one's own mind, not degrade oneself. For the mind is the friend of the conditioned soul, and his enemy as well.",
        "source": "Bhagavad Gita 6.5",
        "queries": [
            "Self-elevation and mind as friend or enemy in Gita",
            "How to master your own mind Sanskrit verse",
            "Uddhared atmanatmanam verse meaning"
        ]
    },
    {
        "id": "up_isha_1",
        "sanskrit": "ईशा वास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्। तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम्॥",
        "transliteration": "īśā vāsyamidaṃ sarvaṃ yatkiñca jagatyāṃ jagat | tena tyaktena bhuñjīthā mā gṛdhaḥ kasyasviddhanam ||",
        "english": "All this, whatsoever moves in this moving world, is pervaded by the Lord. Through renunciation enjoy it; do not covet anyone's wealth.",
        "source": "Isha Upanishad Verse 1",
        "queries": [
            "Isha Upanishad verse 1 meaning on detachment and divine presence",
            "Everything is pervaded by the Divine Upanishad verse",
            "Isha vasyam idam sarvam Sanskrit text and English meaning"
        ]
    },
    {
        "id": "bg_18.66",
        "sanskrit": "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज। अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥",
        "transliteration": "sarvadharmānparityajya māmekaṃ śaraṇaṃ vraja | ahaṃ tvā sarvapāpebhyo mokṣayiṣyāmi mā śucaḥ ||",
        "english": "Abandon all varieties of religion and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear.",
        "source": "Bhagavad Gita 18.66",
        "queries": [
            "Ultimate surrender to God verse in Bhagavad Gita",
            "Sarva dharman parityajya verse meaning",
            "Liberation from sins through total refuge in Krishna"
        ]
    },
    {
        "id": "bg_2.13",
        "sanskrit": "देहिनोऽस्मिन्यथा देहे कौमारं यौवनं जरा। तथा देहान्तरप्राप्तिर्धीरस्तत्र न मुह्यति॥",
        "transliteration": "dehino'sminyathā dehe kaumāraṃ yauvanaṃ jarā | tathā dehāntaraprāptirdhīrastatra na muhyati ||",
        "english": "As the embodied soul continuously passes, in this body, from boyhood to youth to old age, the soul similarly passes into another body at death. A sober person is not bewildered by such a change.",
        "source": "Bhagavad Gita 2.13",
        "queries": [
            "Reincarnation and passage of soul through body stages",
            "Sanskrit verse explaining transmigration of Atman",
            "Why wise people do not grieve for death in Gita"
        ]
    }
]

def fetch_external_bhagavad_gita_dataset():
    """
    Downloads live Bhagavad Gita Sanskrit + English corpus (700+ verses) from GitHub raw data repository.
    Falls back to curated corpus if offline.
    """
    url = "https://raw.githubusercontent.com/gita/gita/main/data/verse.json"
    print(f"[INFO] Fetching live Bhagavad Gita Sanskrit + English corpus from: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                raw_json = json.loads(response.read().decode('utf-8'))
                fetched_verses = []
                for idx, v in enumerate(raw_json):
                    text_sa = v.get("text", "").strip()
                    translit = v.get("transliteration", "").strip()
                    meanings = v.get("word_meanings", "").strip()
                    
                    # English translation / meanings fallback
                    eng_text = meanings if meanings else f"Bhagavad Gita Chapter {v.get('chapter_number')}, Verse {v.get('verse_number')}"
                    
                    if text_sa:
                        fetched_verses.append({
                            "id": f"bg_{v.get('chapter_number', 1)}.{v.get('verse_number', idx+1)}",
                            "sanskrit": text_sa,
                            "transliteration": translit,
                            "english": eng_text,
                            "source": "Bhagavad Gita Complete Corpus (gita/gita)",
                            "queries": [
                                f"What is the meaning of Bhagavad Gita verse {v.get('chapter_number')}.{v.get('verse_number')}?",
                                f"Bhagavad Gita Chapter {v.get('chapter_number')} Verse {v.get('verse_number')} Sanskrit text"
                            ]
                        })
                if fetched_verses:
                    print(f"[OK] Successfully fetched {len(fetched_verses)} live verses from Bhagavad Gita corpus!")
                    return fetched_verses
    except Exception as e:
        print(f"[NOTE] Live dataset fetch skipped ({e}). Using curated Bhagavad Gita & Upanishads corpus.")
    return SAMPLE_SANSKRIT_ENGLISH_DATA

def generate_pairs(data_list):
    pairs = []
    for item in data_list:
        # Pair English translation directly to Sanskrit Devanagari
        pairs.append({
            "id": f"{item['id']}_trans",
            "anchor": item["english"],
            "positive": item["sanskrit"],
            "transliteration": item.get("transliteration", ""),
            "source": item.get("source", "Sanskrit Aligned Corpus")
        })
        # Pair natural language English queries to Sanskrit Devanagari
        for idx, q in enumerate(item.get("queries", [])):
            pairs.append({
                "id": f"{item['id']}_q{idx}",
                "anchor": q,
                "positive": item["sanskrit"],
                "transliteration": item.get("transliteration", ""),
                "source": item.get("source", "Sanskrit Aligned Corpus")
            })
        # Cross-lingual transliteration pair (Romanized Sanskrit -> Devanagari Sanskrit)
        if item.get("transliteration"):
            pairs.append({
                "id": f"{item['id']}_roman",
                "anchor": item["transliteration"],
                "positive": item["sanskrit"],
                "transliteration": item["transliteration"],
                "source": item.get("source", "Sanskrit Aligned Corpus")
            })
    return pairs

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Ingest data from Bhagavad Gita + Upanishads dataset sources
    corpus_data = fetch_external_bhagavad_gita_dataset()
    
    all_pairs = generate_pairs(corpus_data)
    random.seed(42)
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    train_file = os.path.join(output_dir, "train.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")
    corpus_file = os.path.join(output_dir, "corpus.json")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for entry in train_pairs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(test_file, "w", encoding="utf-8") as f:
        for entry in test_pairs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_SANSKRIT_ENGLISH_DATA, f, ensure_ascii=False, indent=2)
        
    print(f"[OK] Dataset preparation complete!")
    print(f"   Train samples: {len(train_pairs)} -> {train_file}")
    print(f"   Test samples:  {len(test_pairs)} -> {test_file}")
    print(f"   Corpus items:  {len(SAMPLE_SANSKRIT_ENGLISH_DATA)} -> {corpus_file}")

if __name__ == "__main__":
    main()
