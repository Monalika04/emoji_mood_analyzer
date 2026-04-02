"""
GoEmotions dataset loader with automatic fallback.

Priority:
  1. HuggingFace Hub mirror (most reliable, no login needed)
  2. Original Google Research GitHub
  3. Generates 800-sample fallback dataset if both fail

Run:  py data/load_dataset.py
"""

import urllib.request
import os, csv, random
from collections import defaultdict, Counter

random.seed(42)

RAW_DIR  = "data/raw"
OUT_SAMP = "data/real_data_sample.csv"

EKMAN_MAP = {
    "admiration":"joy","amusement":"joy","approval":"joy","caring":"joy",
    "desire":"joy","excitement":"joy","gratitude":"joy","joy":"joy",
    "love":"joy","optimism":"joy","pride":"joy","relief":"joy",
    "disappointment":"sadness","grief":"sadness","remorse":"sadness","sadness":"sadness",
    "anger":"anger","annoyance":"anger","disapproval":"anger",
    "fear":"fear","nervousness":"fear",
    "confusion":"surprise","curiosity":"surprise","realization":"surprise","surprise":"surprise",
    "disgust":"disgust","embarrassment":"disgust",
}

EMOTION_COLS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

URL_SETS = [
    [  # HuggingFace (most reliable)
        "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/main/data/full_dataset/goemotions_1.csv",
        "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/main/data/full_dataset/goemotions_2.csv",
        "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/main/data/full_dataset/goemotions_3.csv",
    ],
    [  # Google GitHub
        "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_1.csv",
        "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_2.csv",
        "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_3.csv",
    ],
]


def try_download(url, dest, timeout=20):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            f.write(r.read())
        if os.path.getsize(dest) < 1000:
            os.remove(dest)
            return False
        return True
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_goemotions():
    os.makedirs(RAW_DIR, exist_ok=True)
    for url_set in URL_SETS:
        print(f"  Trying: {url_set[0][:65]}...")
        paths, ok = [], True
        for i, url in enumerate(url_set, 1):
            dest = f"{RAW_DIR}/goemotions_{i}.csv"
            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                print(f"    Part {i}/3 cached.")
                paths.append(dest); continue
            print(f"    Part {i}/3 ...", end=" ", flush=True)
            if try_download(url, dest):
                print(f"OK ({os.path.getsize(dest)//1024} KB)")
                paths.append(dest)
            else:
                print("FAILED"); ok = False; break
        if ok and len(paths) == 3:
            return paths
        print("  Mirror failed, trying next...\n")
    return None


def parse_goemotions(raw_paths):
    rows = []
    for path in raw_paths:
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2: continue
                text = row[0].strip()
                if not text or len(text) < 4: continue
                try:
                    eids = [int(x) for x in row[1].split(",") if x.strip().isdigit()]
                except Exception:
                    continue
                for eid in eids:
                    if eid < len(EMOTION_COLS):
                        lbl = EMOTION_COLS[eid]
                        if lbl != "neutral" and lbl in EKMAN_MAP:
                            rows.append({"text": text, "mood": EKMAN_MAP[lbl]})
                            break
    print(f"  Parsed {len(rows):,} rows")
    return rows


def generate_large_fallback():
    data = {
    "joy": [
        "Just got promoted at work 😊🎉", "Best day of my life today ✨😄",
        "Aced all my exams this semester 📚😁", "Finally finished building my app 💻🙌",
        "Had the most amazing date night 🥂😍", "Our team won the championship 🏆⚽",
        "Got into my dream university 🎓🥳", "My startup just got funded 💰🎊",
        "Just adopted the cutest puppy 🐶❤️", "Surprise party was absolutely perfect 🎂😊",
        "Finished my marathon in under 4 hours 🏃💪", "My book got published today 📖✨",
        "Finally got my dream job offer 💼🎉", "Just got engaged to my best friend 💍🥰",
        "My artwork sold for the first time 🎨😄", "Passed my driving test first attempt 🚗😁",
        "Got the apartment I have been dreaming of 🏠🎊", "My visa got approved finally ✈️😊",
        "Just hit 10k followers on my channel 📱🎉", "Finished project ahead of schedule 📋😄",
        "Met my childhood idol today ✨😊", "Family reunion was the best thing ever 👨‍👩‍👧💛",
        "Just completed my first triathlon 🏊🚴🏃", "Got a perfect score on my presentation 📊😁",
        "My parents are so proud of me today 🥹💛", "First paycheck from my freelance work 💸😊",
        "Finally learned to play my favourite song 🎸😄", "My garden is blooming beautifully 🌸🌼",
        "Cooked my first proper three course meal 🍽️😁", "Ran into my best friend unexpectedly 😊🤗",
        "Just booked my dream holiday 🏖️✈️🎉", "My short film won the festival award 🎬🏆",
        "Finished reading 50 books this year 📚😄✨", "Dog learned a new trick today 🐕😊",
        "Got the internship at my top choice company 💼🥳", "Woke up feeling genuinely happy 😊☀️",
        "My little sister graduated with distinction 🎓💛", "Just learned I am getting a bonus 💰😁",
        "Charity event raised double the target 💛🙌", "Collaboration with favourite artist confirmed 🎵🎉",
    ],
    "sadness": [
        "My grandmother passed away this morning 💔😭", "Got rejected from every university 😞",
        "Best friend is moving across the country 😢💔", "Long distance relationship is breaking me 😔",
        "Lost my job today without warning 😞💔", "My dog of 12 years is gone 😭❤️",
        "Failed my final year dissertation 😢📖", "Parents announced they are divorcing 😔💔",
        "Another year alone on my birthday 😞🎂", "My startup failed after 2 years 💔😢",
        "Got ghosted by someone I really cared about 😔", "Best friend betrayed my trust 😢💔",
        "Can not afford rent this month again 😞", "Health diagnosis was not good today 😔😢",
        "Had to put my cat to sleep 😭💔", "Childhood home is being sold 😢🏠",
        "Mentor who changed my life passed away 💔😞", "Nobody showed up to my exhibition 😔🎨",
        "Relationship of 5 years ended today 💔😭", "Flight home cancelled on Christmas 😢✈️",
        "Got left out of the friend group again 😞", "Missing home so much it physically hurts 😔🏠",
        "Watched my team lose in the final minute 😢⚽", "Portfolio lost everything this month 😞💸",
        "Feeling completely invisible lately 😔", "Nobody reached out on my hard day 😢💔",
        "Therapy is not working and I feel hopeless 😞", "Watched a sunset alone and cried 😢🌅",
        "Lost the pregnancy we had been hoping for 💔😭", "Mentor told me I am not good enough 😞",
        "Been crying without knowing why again 😭😔", "First holiday without my late father 💔😢",
        "Old photo of us made me cry today 😭💔", "Friends all moving forward without me 😞",
        "The project I loved most got cancelled 😔", "Nobody remembered my work anniversary 😢",
        "Feeling so drained and empty inside 😞💔", "Saw my ex looking so happy without me 😢",
        "The letter I wrote never got a reply 😔💔", "My city flooded and I lost everything 😭💔",
    ],
    "anger": [
        "They promoted someone less qualified over me 😡💢", "Boss took credit for my entire project 🤬",
        "Waited 3 hours and was completely ignored 😠", "They charged me twice and refuse to refund 😡",
        "Landlord still not fixed anything after 6 weeks 😤💢", "Data got leaked and nobody is accountable 😡",
        "They cancelled my flight and offered nothing 😠✈️", "Someone plagiarised my research paper 🤬📄",
        "Contractor disappeared with half my money 😡💸", "Manager screamed at me in front of team 😤🤬",
        "They changed the terms after I signed 😠", "Neighbour parking in my spot for months 😡",
        "Charged for subscription I cancelled twice 😤💢", "They lost my luggage and shrugged it off 😡✈️",
        "Delivery was 3 weeks late and arrived broken 😠📦", "Colleague keeps interrupting me in meetings 🤬",
        "Was discriminated against at the restaurant 😡", "Account deleted with no warning or reason 😤💢",
        "HR protecting the bully not the victim 😡🤬", "Medical records given to wrong person 😠",
        "Exam was nothing like what we were told 😤😡", "Raised prices without any notice 😠💢",
        "Phone was hacked and support is useless 🤬📱", "He lied to my face for months 😡💔",
        "Keep changing the deadline without telling me 😤", "Train cancelled but no refund available 😡🚂",
        "My idea was stolen in the meeting today 😠💢", "Broke my device and blamed me for it 🤬",
        "Customer service hung up on me three times 😡📞", "Hard work was never once acknowledged 😤",
        "Gave away my reserved table without notice 😡🍽️", "False review destroyed my small business 🤬😠",
        "Payment was declined for no reason 😡💸", "Woke the whole building at 3am again 😤💢",
        "Feedback has been ignored for six months 😠", "Hired externally for a role I was promised 😡",
        "My medication given to the wrong patient 🤬😠", "Fined for something that was not my fault 😡",
        "Kept my deposit without any justification 😤💢", "Complaint has been open for two months 😠🤬",
    ],
    "fear": [
        "Biopsy results come back tomorrow 😰😨", "Blood pressure reading shocked the doctor 😨",
        "Earthquake warning just came through 😱⚠️", "Alone in the house and heard something upstairs 😰🌑",
        "First time flying a plane as pilot tomorrow 😨✈️", "Medical test detected something on scan 😰",
        "Walking home alone at midnight tonight 😱", "Anxiety is completely out of control 😨💓",
        "Presenting to 800 people in one hour 😰🎤", "Strange car has been parked outside all week 😨",
        "Got a letter from a debt collector today 😰💸", "Surgery scheduled for next week 😨🏥",
        "Job could be automated away very soon 😰💼", "Partner has not replied in over 12 hours 😨",
        "Doctor wants a second opinion urgently 😰🩺", "Fire alarm going off with no drill scheduled 😱🔥",
        "Child had a seizure for the first time 😨", "Immune results are back and look scary 😰",
        "Standing on the roof of a 40 storey building 😱🏙️", "MRI results appointment is tomorrow 😨",
        "Panic attack starting and I am in public 😰💓", "He knows where I live and I am scared 😱",
        "Made a huge financial mistake I cannot fix 😨💸", "My passport was stolen abroad 😰",
        "Driving in a blizzard with no visibility 😱🚗❄️", "My child has not come home from school 😨",
        "The turbulence is getting worse 😰✈️", "Might be losing my vision slowly 😨👁️",
        "She threatened me and I believe her 😱", "Background check might reveal something bad 😰",
        "Alone in the forest after dark 😱🌲", "Chest pains started again tonight 😨💓",
        "Visa expires in 5 days with no extension 😰🛂", "Someone is following me home I think 😱",
        "Medication ran out and pharmacy is closed 😨💊", "Interview for dream job in 10 minutes 😰",
        "Car brakes feel soft on the motorway 😱🚗", "Symptoms match something very serious 😨",
        "Being investigated at work for something false 😰", "Do not know how I will pay for treatment 😨💸",
    ],
    "surprise": [
        "Flight got upgraded to first class for free 😲✈️", "Found 200 in a jacket I forgot about 😮💵",
        "My tweet went viral overnight 😲📱", "Ran into my twin at an airport abroad 😮",
        "Got a call saying I won the competition 😲🎉", "My ex texted out of nowhere after two years 😳📞",
        "Test came back negative after all 😲😮", "A celebrity followed my account today 😳✨",
        "Got offered a job I never applied for 😲💼", "Landlord dropped the rent for next year 😮💸",
        "Plot twist I did not see coming at all 🤯😮", "My package arrived three weeks early 😲📦",
        "Found out I have a half sibling today 😳😮", "Got selected out of 50000 applicants 🤯🎉",
        "Parents secretly planned a surprise trip 😲✈️", "Company matched my asking salary exactly 😮💰",
        "Totally forgot it was my birthday today 😲🎂", "Random stranger paid for my entire meal 😳",
        "Received a hand-written letter from my idol 😮✨", "My article got picked up by a major outlet 😲📰",
        "Found a first edition book at a car boot sale 😮📚", "Won against the tournament favourites 😲🏆",
        "Got a scholarship I had forgotten I applied for 🤯🎓", "Lightning struck right next to me 😱😮",
        "Boss called just to say I was doing great work 😲💼", "Wi-Fi on the remote island actually worked 😳📶",
        "Old friend tracked me down after 20 years 😮🤝", "Got a standing ovation I did not expect 😲👏",
        "Bumped into my favourite author at a cafe 😮✨", "Last minute bid on the house worked 😲🏠",
        "A company I invested in just IPO-ed 🤯💸", "My shy friend gave a TEDx talk today 😮🎤",
        "Received anonymous flowers with no explanation 😳🌸", "My cold case complaint just got reopened 😲",
        "My homemade jam won a regional award 😮🏆", "Bus driver refused payment and smiled 😳",
        "Found 4-leaf clover on the same day I needed luck 😮🍀", "Stranger returned wallet with everything inside 😲💰",
        "Rejected manuscript got accepted five years later 😳📖", "Walked into my own surprise retirement party 🤯🎊",
    ],
    "disgust": [
        "Found mould in my takeaway food 🤢😖", "Corruption exposed in report is nauseating 🤮😒",
        "That influencer is the most fake person 😒🤮", "Tested products on animals after promising not to 🤢",
        "Office toilets are beyond unacceptable 🤢😫", "Watching them eat with mouth wide open 😒🤢",
        "Company dumps waste directly into the river 🤮😡", "Abuse was covered up for years 😒🤢",
        "Rotten smell in the entire hotel corridor 🤢😖", "He groped her and laughed it off 🤮😡",
        "Found a cockroach inside the sealed food 🤢😱", "Wage theft disguised as company policy 😒🤮",
        "Clickbait preying on grief is despicable 🤢", "Exploitation of elderly residents is sickening 🤮😒",
        "That scene in the documentary was so graphic 🤢😫", "He plagiarised a student thesis for profit 😒🤮",
        "Slaughterhouse footage changed everything for me 🤢", "Fake charity keeping 95% of donations 🤮😡",
        "Gaslighting your partner for years is vile 😒🤢", "Mystery meat in the labelled chicken dish 🤢😖",
        "Predatory lending targeting single mothers 🤮😒", "Maggots in the hospital meal is unforgivable 🤢😱",
        "Deepfake used to destroy her reputation 😒🤮", "Sewage on the supposedly clean beach 🤢😫",
        "Bullying a child with a disability on camera 🤮😡", "Using a tragedy to sell merchandise 😒🤢",
        "Rat droppings found throughout the kitchen 🤢😖", "Historical records of abuse finally coming out 🤮😒",
        "Children used as shields called heroes 🤢😡", "That organic product is actually chemical waste 😒🤮",
        "Fraudulent degree sold to vulnerable students 🤢", "Selling expired medicine in poor countries 🤮😡",
        "Body shaming ad campaign disgusts me completely 😒🤢", "Used cigarette in my freshly served dessert 🤢😖",
        "Covering up the oil spill for three weeks 🤮😒", "Harassment labelled as banter by management 😒🤢",
        "Pet food contained banned slaughterhouse waste 🤢😱", "Hypocrisy of that speech is breathtaking 😒🤮",
        "Selling counterfeit medicine online is criminal 🤮😡", "That comment about refugees made me sick 🤢",
    ]}

    rows = []
    for mood, texts in data.items():
        for text in texts:
            rows.append({"text": text, "mood": mood})
    random.shuffle(rows)
    return rows


def save_dataset(rows, path):
    counts = Counter(r["mood"] for r in rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id","text","mood"])
        writer.writeheader()
        for i, r in enumerate(rows, 1):
            writer.writerow({"text_id":i,"text":r["text"],"mood":r["mood"]})
    print(f"\n  Saved: {path}  ({len(rows):,} rows)")
    for mood, cnt in sorted(counts.items()):
        print(f"    {mood:<10} {cnt}")


if __name__ == "__main__":
    print("\n── Dataset Setup ─────────────────────────────")
    raw_paths = download_goemotions()

    if raw_paths:
        print("\n── Parsing GoEmotions ────────────────────────")
        rows = parse_goemotions(raw_paths)
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["mood"]].append(r)
        balanced = []
        for mood, items in by_class.items():
            random.shuffle(items)
            balanced.extend(items[:1000])
        random.shuffle(balanced)
        save_dataset(balanced, OUT_SAMP)
    else:
        print("\n  No internet / download failed.")
        print("  Generating built-in 800-sample dataset instead...")
        rows = generate_large_fallback()
        save_dataset(rows, OUT_SAMP)

    print(f"\n  Done!  Now run:  py -m streamlit run app.py")
