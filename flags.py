import pycountry

# Manual overrides for FIFA team names that pycountry can't find automatically
MANUAL_FLAGS = {
    "Korea Republic": "KR",
    "Korea DPR": "KP",
    "IR Iran": "IR",
    "Congo DR": "CD",
    "Congo": "CG",
    "Côte d'Ivoire": "CI",
    "FYR Macedonia": "MK",
    "Cape Verde Islands": "CV",
    "Kyrgyz Republic": "KG",
    "Swaziland": "SZ",
    "Sao Tome and Principe": "ST",
    "Brunei Darussalam": "BN",
    "St. Kitts and Nevis": "KN",
    "St. Lucia": "LC",
    "St. Vincent and the Grenadines": "VC",
    "USA": "US",
    "England": "GB",
    "Scotland": "GB",
    "Wales": "GB",
    "Northern Ireland": "GB",
    "Chinese Taipei": "TW",
    "Syria": "SY",
    "Bolivia": "BO",
    "Tanzania": "TZ",
    "Vietnam": "VN",
    "Laos": "LA",
    "Moldova": "MD",
    "Russia": "RU",
    "South Korea": "KR",
}

def get_flag(team_name):
    # Check manual overrides first
    if team_name in MANUAL_FLAGS:
        code = MANUAL_FLAGS[team_name]
        return code_to_flag(code)

    # Try pycountry lookup
    try:
        country = pycountry.countries.search_fuzzy(team_name)
        if country:
            return code_to_flag(country[0].alpha_2)
    except Exception:
        pass

    return ""  # No flag found — return empty string

def code_to_flag(alpha2):
    # Flag emojis are built from two unicode "regional indicator" letters
    # A = U+1F1E6, B = U+1F1E7 etc. So "US" becomes the US flag emoji
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in alpha2.upper())