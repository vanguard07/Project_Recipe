import spacy
from Levenshtein import distance

nlp = spacy.load("en_core_web_sm")

# ✅ Build your food vocabulary
KNOWN_INGREDIENTS = [
    "tomato", "onion", "garlic", "cheese", "milk", "butter", "egg", "flour", "sugar", "salt", "pepper",
    "olive oil", "chicken", "beef", "pork", "fish", "shrimp", "lamb", "carrot", "potato", "spinach",
    "lettuce", "cucumber", "broccoli", "cauliflower", "green beans", "peas", "corn", "zucchini", "bell pepper",
    "mushroom", "ginger", "yogurt", "cream", "mayonnaise", "vinegar", "soy sauce", "hot sauce", "honey",
    "ketchup", "mustard", "basil", "parsley", "cilantro", "oregano", "thyme", "rosemary", "dill", "chili powder",
    "cumin", "turmeric", "paprika", "coriander", "cinnamon", "nutmeg", "clove", "cardamom", "vanilla",
    "coconut milk", "paneer", "tofu", "tempeh", "rice", "pasta", "noodles", "bread", "tortilla", "quinoa",
    "oats", "barley", "lentils", "chickpeas", "beans", "black beans", "kidney beans", "white beans",
    "green gram", "red lentils", "moong dal", "urad dal", "chana dal", "baking powder", "baking soda",
    "yeast", "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry", "blackberry",
    "pineapple", "mango", "papaya", "pear", "peach", "plum", "cherry", "lemon", "lime", "avocado",
    "nuts", "almond", "cashew", "walnut", "pistachio", "hazelnut", "sunflower seeds", "chia seeds",
    "pumpkin seeds", "raisins", "dates", "figs", "jam", "peanut butter", "maple syrup", "molasses",
    "water", "sparkling water", "juice", "soda", "coffee", "tea", "green tea", "black tea",
    "herbal tea", "eggs", "sausage", "bacon", "hamburger", "hotdog", "steak", "duck", "turkey",
    "cabbage", "eggplant", "radish", "turnip", "beet", "sweet potato", "arugula", "kale",
    "collard greens", "mustard greens", "bok choy", "leek", "scallion", "artichoke", "asparagus",
    "brussels sprouts", "okra", "horseradish", "jicama", "rhubarb", "squash", "acorn squash", "butternut squash",
    "spaghetti squash", "watermelon", "cantaloupe", "honeydew", "grapefruit", "coconut", "lychee", "passion fruit",
    "guava", "star fruit", "kiwi", "dragon fruit", "durian", "tamarind"
]


def fuzzy_match(word: str, max_distance: int = 1) -> str:
    word = word.lower()

    # Direct match with known ingredients
    if word in KNOWN_INGREDIENTS:
        return word

    best_match = None
    best_score = float('inf')

    for ingredient in KNOWN_INGREDIENTS:
        dist = distance(word, ingredient)
        if dist < best_score and dist <= max_distance:
            best_match = ingredient
            best_score = dist

    print(
        f"[DEBUG] word='{word}' → match='{best_match}', distance={best_score}")
    return best_match if best_match else word


def preprocess_query(user_input: str) -> str:
    doc = nlp(user_input.lower())

    matched_ingredients = []

    for token in doc:
        # ✅ Only match if spaCy says it's a noun OR it's already in known ingredients
        if (token.is_alpha) or token.text in KNOWN_INGREDIENTS:
            match = fuzzy_match(token.text)
            if match:
                matched_ingredients.append(match)

    matched_ingredients = list(set(matched_ingredients))  # deduplicate

    # ✨ Return only the ingredient name if there's only one match
    if len(matched_ingredients) == 1:
        return matched_ingredients[0]
    elif matched_ingredients:
        return f"Give me a recipe using the ingredients: {', '.join(matched_ingredients)}."
    else:
        return "Sorry, I couldn’t detect any valid ingredients."
