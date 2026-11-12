"""v2 stimuli: word-free location descriptions and posture-then-touch.
Word-free: each location described without naming it or (mostly) any body part; same 8 sensation templates as v1."""
import stimuli as S

WORDFREE = {
    "wrist": ["the spot where your watch sits", "the place where a bracelet rests on you", "the place a hospital name band is fastened on you",
              "the spot where your sleeve cuff ends", "the joint you turn to check the time", "the spot where a nurse feels for your pulse"],
    "hand": ["the part of you that you clap with", "the part of you a fortune teller reads", "the part of you that you shake with when you meet someone",
             "the part of you that holds your coffee cup", "the part of you that you wave hello with", "the part of you that holds a pen"],
    "finger": ["the part of you that taps a phone screen", "the part of you that presses an elevator button", "the part of you that you point with",
               "the spot where a wedding ring goes", "the part of you that you lick to turn a page", "the spot where a paper cut usually stings"],
    "foot": ["the place where your shoes pinch", "the part of you that you stand on", "the part of you that socks cover",
             "the part of you that touches the floor when you walk barefoot", "the part of you that would land on a lego brick", "the part of you that kicks a ball"],
    "ear": ["the spot you scratch behind when you're puzzled", "the place where an earring hangs", "the part of you that headphones cover",
            "the place where you tuck a pencil to keep it handy", "the part of you that you cup to hear a whisper", "the spot where your glasses hook on"],
    "crown": ["the place where a hat sits", "the place a king's crown would sit", "the spot raindrops hit first when you stand up straight",
              "the place where shampoo lathers", "the spot where your hair parts", "the very topmost point of you"],
    "knee": ["the joint that bends when you sit down on a chair", "the spot you'd scrape falling off a bike", "the place a doctor taps with a little rubber hammer",
             "the spot that shows through ripped jeans", "the joint that clicks when you crouch", "the joint you rest on to propose marriage"],
    "belly": ["the place that growls when you're hungry", "the place where you feel butterflies when nervous", "the soft middle of you",
              "the place that feels full after a big dinner", "the place that tightens when you laugh hard", "the place a seatbelt's lap strap crosses"],
    "lips": ["the place where lipstick goes", "the place a straw rests while you sip", "the place where a first kiss lands",
             "the part of you that blows out birthday candles", "the part of you that you press together to hum", "the place you'd dab a napkin after eating"],
    "shoulder": ["the place where a backpack strap presses", "the place a pirate's parrot perches", "the place a friend taps to get your attention from behind",
                 "the place where a purse strap hangs", "the part of you that you shrug with", "the place a baby rests its head when you burp it"],
    "neck": ["the place a necklace hangs", "the place where a tie is knotted", "the place a scarf wraps around",
             "the spot that bobs when you swallow", "the place where a collar button closes", "the place a doctor feels when you have swollen glands"],
    "elbow": ["the joint you lean on at a table", "the spot people call the funny bone", "the joint you nudge someone with",
              "the joint that bends when you lift a cup to drink", "the joint that rests on a car's window ledge", "the bony joint you bang on a door frame"],
}
# named-part equivalents (v1 location keys, averaged)
NAMED = {"wrist": ["wristL", "wristR"], "hand": ["palmL", "palmR", "backhandL", "backhandR"],
         "finger": [f + s for f in ["index", "middle", "ring"] for s in "LR"], "foot": [f + s for f in ["toes", "sole", "heel"] for s in "LR"],
         "ear": ["earL", "earR"], "crown": ["crown"], "knee": ["kneeL", "kneeR"], "belly": ["belly"], "lips": ["lips"],
         "shoulder": ["shoulderL", "shoulderR"], "neck": ["neckF"], "elbow": ["elbowL", "elbowR"]}

# posture: one hand placed somewhere, then a touch on the back of one hand (same or other)
PLACES = {"head": "on top of your head", "belly": "flat on your belly", "knee": "on your {s} knee", "side": "hanging loose at your side"}
def posture_text(place, posed, touched, tpl):
    p = PLACES[place].format(s=posed)
    verb = "Let" if place == "side" else "Rest"
    first = f"{verb} your {posed} hand {p}." if place != "side" else f"Let your {posed} arm hang loose at your side."
    return first + " Now: " + tpl.format(L=f"the back of your {touched} hand")

def build2():
    rows = []
    for k, ds in WORDFREE.items():
        for d, desc in enumerate(ds):
            for t, tpl in enumerate(S.TEMPLATES):
                rows.append(dict(cond="wordfree", loc=k, li=-1, tpl=t, desc=d, text=tpl.format(L=desc)))
    for place in PLACES:
        for posed in ["left", "right"]:
            for touched in ["left", "right"]:
                for t, tpl in enumerate(S.TEMPLATES):
                    rows.append(dict(cond="posture", loc=place, li=-1, tpl=t, posed=posed, touched=touched,
                                     text=posture_text(place, posed, touched, tpl)))
    return rows

if __name__ == "__main__":
    r = build2(); print(len(r))
    for x in r[::61]: print(x["cond"], x["loc"], "|", x["text"])
