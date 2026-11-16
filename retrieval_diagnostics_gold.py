"""Evidence annotations for the frozen 14-question pilot, independent of retrieval labels.

These are Codex-authored diagnostic annotations, not official AML labels or
independent human adjudication. Source indices refer to the original dataset.
"""


def fact(fid, statement, sid, turn, quote, criteria=""):
    return {
        "id": fid,
        "statement": statement,
        "criteria": criteria,
        "sources": [{"session_id": sid, "turn_index": turn, "quote": quote}],
    }


ANNOTATIONS = {
    "4fd1909e": {
        "facts": [
            fact("venue", "The user attended an Imagine Dragons concert at Xfinity Center.",
                 "answer_2952aee4", 2,
                 "I went to an amazing Imagine Dragons concert recently, it was at the Xfinity Center",
                 "Require the concert/artist and venue relationship; a venue name alone is insufficient."),
        ],
    },
    "d596882b": {
        "facts": [
            fact("restaurant", "By Chloe was an assistant-recommended vegan eatery in New York City.",
                 "answer_ultrachat_252214", 5,
                 "Check out By Chloe, a popular plant-based eatery with multiple locations throughout the city.",
                 "A compressed record of the recommendation is sufficient. Generic New York travel advice or other restaurants are not."),
        ],
    },
    "505af2f5": {
        "facts": [
            fact("almond", "The user uses almond milk in homemade coffee creamer.",
                 "answer_f3164f2c", 0, "making my own flavored creamer with almond milk, vanilla extract, and honey"),
            fact("vanilla", "The user uses vanilla extract in homemade coffee creamer.",
                 "answer_f3164f2c", 0, "making my own flavored creamer with almond milk, vanilla extract, and honey"),
            fact("honey", "The user uses honey in homemade coffee creamer.",
                 "answer_f3164f2c", 0, "making my own flavored creamer with almond milk, vanilla extract, and honey"),
            fact("sugar", "The user's creamer choices should help reduce sugar intake.",
                 "answer_f3164f2c", 0, "I'm trying to reduce my sugar intake and save money"),
            fact("budget", "The user's creamer choices should help save money.",
                 "answer_f3164f2c", 0, "I'm trying to reduce my sugar intake and save money"),
        ],
        "note": "Five independently checkable preference/background facts; full coverage is stricter than merely producing one acceptable recommendation.",
    },
    "1a8a66a6": {
        "facts": [
            fact("new_yorker", "The user has an active subscription to The New Yorker.",
                 "answer_2bd23659_1", 0,
                 "I've been loving my subscription to The New Yorker magazine, which I subscribed to in early February"),
            fact("architectural_digest", "The user receives Architectural Digest as another current magazine subscription.",
                 "answer_2bd23659_2", 0,
                 "I'm also getting Architectural Digest, which I love for home decor inspiration."),
            fact("forbes_cancelled", "The user's Forbes subscription was cancelled and must not be counted as active.",
                 "answer_2bd23659_3", 0,
                 "I just canceled my Forbes magazine subscription in early March"),
        ],
        "note": "A one-off National Geographic purchase is not proof of a subscription. Count source-supported active subscriptions and the explicit cancellation.",
    },
    "gpt4_93159ced": {
        "facts": [
            fact("total_career", "The user has worked professionally for 9 years in total.",
                 "answer_e5131a1b_1", 2, "I've been working professionally for 9 years"),
            fact("novatech_tenure", "The user has worked at NovaTech for 4 years and 3 months.",
                 "answer_e5131a1b_2", 4, "I've been working at NovaTech for about 4 years and 3 months now."),
        ],
        "note": "The two source operands support subtraction to obtain 4 years 9 months. A derived answer alone does not establish recall of both source operands.",
    },
    "cf22b7bf": {
        "facts": [
            fact("latest_weight_loss", "The latest reported cumulative weight loss since consistent gym attendance is 10 pounds.",
                 "answer_ae3a122b_2", 2,
                 "I've lost 10 pounds since I started going consistently to the gym 3 months ago.",
                 "An earlier 5-pound loss is not the target fact. Require loss, not a weight goal or body weight."),
        ],
    },
    "8550ddae": {
        "facts": [
            fact("cocktail", "The user actually tried making lavender gin fizz.",
                 "answer_c8354ae9", 2, "I tried a lavender gin fizz recipe last weekend",
                 "Require trying/making the drink, not planning Lavender Dream or merely liking lavender."),
        ],
        "question_constraint_status": "ambiguous",
        "note": "The May 22 evidence says 'last weekend', while the question is dated May 30. Score the supported core fact only; full support for the question's time condition is not established.",
    },
    "778164c6": {
        "facts": [
            fact("dish", "The assistant's earlier Caribbean dish suggestions included Grilled Snapper with Mango Salsa.",
                 "answer_ultrachat_399000", 3,
                 "Grilled Snapper with Mango Salsa - a popular dish in many Caribbean countries where grilled snapper is topped with a fruity and spicy salsa.",
                 "Preserve snapper plus mango salsa. Generic fish recipes, escovitch fish, or fruit alone do not count."),
        ],
        "note": "The source calls this dish Caribbean rather than specifically Jamaican. Coverage measures the source-supported dish identity, not independent verification of cuisine origin.",
    },
    "0a34ad58": {
        "facts": [
            fact("suica", "The user already has a Suica card for the Tokyo trip.",
                 "answer_cebb7159", 2, "using my Suica card I just got",
                 "Generic advice to buy a Suica card is not evidence that this user has one."),
            fact("tripit", "The user already downloaded TripIt to organize the Tokyo trip.",
                 "answer_cebb7159", 6, "I have downloaded the TripIt app to stay organized",
                 "Generic travel app recommendations do not count."),
        ],
    },
    "37f165cf": {
        "facts": [
            fact("nightingale_pages", "The user finished The Nightingale, which had 440 pages.",
                 "answer_6b9b2b1e_2", 0,
                 '"The Nightingale" by Kristin Hannah, which had 440 pages'),
            fact("other_novel_pages", "The user finished a separate novel with 416 pages.",
                 "answer_6b9b2b1e_1", 0, "I just finished a 416-page novel"),
        ],
        "question_constraint_status": "ambiguous",
        "note": "440 + 416 supports 856, but the May evidence does not establish the question's January/March completion dates. Score the two page-count facts, and report this question separately.",
    },
    "gpt4_7a0daae1": {
        "facts": [
            fact("purchase_date", "The user bought the new tennis racket on March 10, 2023.",
                 "answer_4d5490f1_1", 6, "I just bought a new tennis racket online today",
                 "The original conversation date is 2023-03-10. Accept an explicit event date or 'today' anchored by an original Conversation date in the record. A backend Memory source timestamp alone is not an established event date."),
            fact("receipt_date", "The user received that new tennis racket on March 17, 2023.",
                 "answer_4d5490f1_2", 0, "I just received my new tennis racket today",
                 "The original conversation date is 2023-03-17. Accept an explicit event date or 'today' anchored by an original Conversation date in the record. A backend Memory source timestamp alone is not an established event date."),
        ],
    },
    "e61a7584": {
        "facts": [
            fact("latest_tenure", "The latest report says the user has had the cat Luna for about 9 months.",
                 "answer_f25c32f5_2", 6, "I've had my cat, Luna, for about 9 months now",
                 "The latest report is dated November 30, 2023. An old 6-month tenure alone or the cat's age does not count. A correctly dated adoption around March 2023 is equivalent."),
        ],
    },
}

ABSTENTION_IDS = ("2133c1b5_abs", "gpt4_70e84552_abs")
