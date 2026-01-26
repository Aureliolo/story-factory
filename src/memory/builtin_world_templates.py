"""Built-in world templates for genre presets."""

import logging

from src.memory.templates import EntityHints, WorldTemplate

logger = logging.getLogger(__name__)


# =============================================================================
# HIGH FANTASY
# =============================================================================

HIGH_FANTASY = WorldTemplate(
    id="high_fantasy",
    name="High Fantasy",
    description="Epic fantasy with magic, kingdoms, and grand quests",
    is_builtin=True,
    genre="fantasy",
    entity_hints=EntityHints(
        character_roles=[
            "chosen_one",
            "wise_wizard",
            "knight",
            "rogue",
            "healer",
            "dark_lord",
            "princess",
            "dragon_rider",
            "ranger",
            "bard",
        ],
        location_types=[
            "castle",
            "ancient_forest",
            "mountain_fortress",
            "magical_academy",
            "underground_city",
            "sacred_temple",
            "haunted_ruins",
            "dragon_lair",
            "elven_realm",
            "dwarven_mine",
        ],
        faction_types=[
            "royal_court",
            "wizards_council",
            "knights_order",
            "thieves_guild",
            "dark_cult",
            "ancient_order",
            "merchant_guild",
            "resistance_movement",
        ],
        item_types=[
            "legendary_sword",
            "magic_ring",
            "ancient_tome",
            "enchanted_amulet",
            "prophecy_scroll",
            "dragon_egg",
            "crown_of_power",
            "staff_of_ages",
        ],
        concept_types=[
            "ancient_prophecy",
            "forbidden_magic",
            "blood_oath",
            "sacred_bond",
            "the_balance",
            "elemental_forces",
            "soul_magic",
        ],
    ),
    relationship_patterns=[
        "sworn_to",
        "mentor_of",
        "destined_enemies",
        "blood_bonded",
        "magical_contract",
        "royal_allegiance",
    ],
    naming_style="Use fantasy names with apostrophes, compound words, or elvish/dwarven influences. "
    "Mix Celtic, Norse, and invented linguistic roots.",
    recommended_counts={
        "characters": (5, 12),
        "locations": (4, 10),
        "factions": (2, 6),
        "items": (3, 8),
        "concepts": (2, 5),
    },
    atmosphere="Epic, wondrous, and mythic. Magic permeates the world, ancient powers stir, "
    "and heroes rise to meet destiny.",
    tags=["fantasy", "epic", "magic", "medieval", "quest"],
)


# =============================================================================
# CYBERPUNK
# =============================================================================

CYBERPUNK = WorldTemplate(
    id="cyberpunk",
    name="Cyberpunk",
    description="Neon-lit dystopia with megacorps, hackers, and transhumanism",
    is_builtin=True,
    genre="science_fiction",
    entity_hints=EntityHints(
        character_roles=[
            "netrunner",
            "corpo_exec",
            "street_samurai",
            "fixer",
            "techie",
            "ripperdoc",
            "ai_construct",
            "rebel_leader",
            "investigator",
            "mercenary",
        ],
        location_types=[
            "megacorp_tower",
            "underground_club",
            "black_market",
            "data_fortress",
            "slum_district",
            "chrome_clinic",
            "abandoned_factory",
            "orbital_station",
            "neon_strip",
        ],
        faction_types=[
            "megacorporation",
            "street_gang",
            "hacker_collective",
            "black_ops_unit",
            "resistance_cell",
            "crime_syndicate",
            "ai_cult",
            "merc_outfit",
        ],
        item_types=[
            "neural_implant",
            "prototype_weapon",
            "encrypted_data",
            "black_ice",
            "cyberdeck",
            "stolen_tech",
            "corporate_secrets",
            "military_hardware",
        ],
        concept_types=[
            "digital_consciousness",
            "corporate_control",
            "human_augmentation",
            "the_net",
            "synthetic_souls",
            "data_as_currency",
        ],
    ),
    relationship_patterns=[
        "employer_of",
        "owes_debt_to",
        "rival_netrunner",
        "informant_for",
        "augmented_by",
        "blackmailing",
    ],
    naming_style="Mix street handles, corporate names, and tech-influenced monikers. "
    "Use Japanese, Chinese, and invented tech-speak influences.",
    recommended_counts={
        "characters": (4, 10),
        "locations": (4, 8),
        "factions": (3, 6),
        "items": (3, 8),
        "concepts": (2, 4),
    },
    atmosphere="Gritty, neon-soaked, and morally gray. Technology is everywhere but trust is rare. "
    "The powerful exploit the weak while rebels fight from the shadows.",
    tags=["scifi", "dystopia", "tech", "noir", "corporate"],
)


# =============================================================================
# URBAN FANTASY
# =============================================================================

URBAN_FANTASY = WorldTemplate(
    id="urban_fantasy",
    name="Urban Fantasy",
    description="Hidden supernatural world within modern cities",
    is_builtin=True,
    genre="fantasy",
    entity_hints=EntityHints(
        character_roles=[
            "reluctant_hero",
            "ancient_vampire",
            "werewolf_alpha",
            "urban_witch",
            "fae_exile",
            "demon_hunter",
            "psychic_detective",
            "fallen_angel",
            "necromancer",
            "shapeshifter",
        ],
        location_types=[
            "nightclub_haven",
            "occult_bookshop",
            "abandoned_subway",
            "gothic_cathedral",
            "fae_crossing",
            "modern_apartment",
            "underground_arena",
            "penthouse_lair",
            "cemetery_gate",
        ],
        faction_types=[
            "vampire_court",
            "werewolf_pack",
            "witch_coven",
            "hunter_organization",
            "fae_court",
            "demon_cabal",
            "supernatural_council",
            "secret_society",
        ],
        item_types=[
            "enchanted_weapon",
            "binding_contract",
            "ancient_grimoire",
            "protective_charm",
            "cursed_artifact",
            "soul_container",
            "magical_focus",
            "supernatural_phone",
        ],
        concept_types=[
            "the_masquerade",
            "supernatural_law",
            "old_magic",
            "balance_of_power",
            "mortal_ignorance",
            "supernatural_politics",
        ],
    ),
    relationship_patterns=[
        "sire_of",
        "pack_member",
        "coven_sister",
        "sworn_hunter",
        "fae_debt",
        "blood_bound",
    ],
    naming_style="Mix modern names with ancient titles and supernatural epithets. "
    "Some characters use human names to blend in, others use ancient names.",
    recommended_counts={
        "characters": (4, 10),
        "locations": (3, 8),
        "factions": (2, 5),
        "items": (2, 6),
        "concepts": (2, 4),
    },
    atmosphere="The mundane and magical coexist uneasily. Ancient supernatural politics play out "
    "in modern cities while humans remain blissfully unaware.",
    tags=["fantasy", "urban", "supernatural", "modern", "paranormal"],
)


# =============================================================================
# SPACE OPERA
# =============================================================================

SPACE_OPERA = WorldTemplate(
    id="space_opera",
    name="Space Opera",
    description="Galactic empires, alien species, and interstellar adventure",
    is_builtin=True,
    genre="science_fiction",
    entity_hints=EntityHints(
        character_roles=[
            "starship_captain",
            "alien_diplomat",
            "rogue_pilot",
            "imperial_officer",
            "rebel_commander",
            "alien_warrior",
            "ship_engineer",
            "psion",
            "bounty_hunter",
            "trader",
        ],
        location_types=[
            "space_station",
            "alien_homeworld",
            "imperial_capital",
            "frontier_outpost",
            "smuggler_haven",
            "ancient_ruins",
            "gas_giant_mining",
            "generation_ship",
            "diplomatic_hub",
        ],
        faction_types=[
            "galactic_empire",
            "rebel_alliance",
            "merchant_league",
            "pirate_fleet",
            "alien_hive",
            "ancient_order",
            "colonial_government",
            "exploration_corps",
        ],
        item_types=[
            "ancient_artifact",
            "prototype_drive",
            "alien_tech",
            "encrypted_coordinates",
            "psionic_crystal",
            "superweapon",
            "diplomatic_seal",
            "navigation_chart",
        ],
        concept_types=[
            "hyperspace",
            "galactic_law",
            "ancient_progenitors",
            "psionics",
            "first_contact",
            "the_void",
        ],
    ),
    relationship_patterns=[
        "crew_of",
        "imperial_subject",
        "sworn_enemy",
        "blood_debt",
        "diplomatic_ally",
        "telepathic_bond",
    ],
    naming_style="Mix human names with alien-sounding names. Use compound words, "
    "apostrophes, and creative syllable combinations for aliens.",
    recommended_counts={
        "characters": (5, 12),
        "locations": (4, 10),
        "factions": (3, 7),
        "items": (3, 8),
        "concepts": (2, 5),
    },
    atmosphere="Vast and adventurous. Ancient mysteries await discovery, empires clash, "
    "and individuals can shape the fate of worlds.",
    tags=["scifi", "space", "aliens", "adventure", "epic"],
)


# =============================================================================
# POST-APOCALYPTIC
# =============================================================================

POST_APOCALYPTIC = WorldTemplate(
    id="post_apocalyptic",
    name="Post-Apocalyptic",
    description="Survival in a world after civilization's collapse",
    is_builtin=True,
    genre="science_fiction",
    entity_hints=EntityHints(
        character_roles=[
            "survivor",
            "wasteland_warlord",
            "trader",
            "scavenger",
            "settlement_leader",
            "mutant",
            "preacher",
            "mechanic",
            "scout",
            "healer",
        ],
        location_types=[
            "ruined_city",
            "survivor_settlement",
            "bunker",
            "trading_post",
            "warlord_fortress",
            "contaminated_zone",
            "abandoned_mall",
            "highway_junction",
            "water_source",
        ],
        faction_types=[
            "survivor_community",
            "raider_gang",
            "trading_caravan",
            "cult",
            "military_remnant",
            "scavenger_crew",
            "mutant_tribe",
            "tech_hoarders",
        ],
        item_types=[
            "pre_war_tech",
            "clean_water",
            "ammunition",
            "medical_supplies",
            "working_vehicle",
            "radiation_gear",
            "preserved_food",
            "old_world_map",
        ],
        concept_types=[
            "the_old_world",
            "radiation",
            "the_collapse",
            "resource_scarcity",
            "mutation",
            "hope_for_future",
        ],
    ),
    relationship_patterns=[
        "trades_with",
        "protects",
        "raids",
        "owes_life_debt",
        "exile_from",
        "blood_feud",
    ],
    naming_style="Mix practical nicknames, pre-collapse names, and descriptive titles. "
    "Names often reflect skills, appearance, or origin.",
    recommended_counts={
        "characters": (4, 10),
        "locations": (3, 8),
        "factions": (2, 6),
        "items": (4, 10),
        "concepts": (2, 4),
    },
    atmosphere="Harsh and unforgiving but not without hope. Resources are scarce, trust is earned, "
    "and survival depends on community and cunning.",
    tags=["scifi", "survival", "wasteland", "dystopia", "gritty"],
)


# =============================================================================
# NOIR DETECTIVE
# =============================================================================

NOIR_DETECTIVE = WorldTemplate(
    id="noir_detective",
    name="Noir Detective",
    description="Crime, moral ambiguity, and shadowy investigations",
    is_builtin=True,
    genre="crime",
    entity_hints=EntityHints(
        character_roles=[
            "private_detective",
            "femme_fatale",
            "crime_boss",
            "corrupt_cop",
            "desperate_client",
            "loyal_partner",
            "informant",
            "hired_muscle",
            "politician",
            "journalist",
        ],
        location_types=[
            "smoky_office",
            "jazz_club",
            "back_alley",
            "penthouse",
            "police_station",
            "dockside_warehouse",
            "cheap_motel",
            "mansion",
            "underground_casino",
        ],
        faction_types=[
            "crime_syndicate",
            "police_department",
            "political_machine",
            "smuggling_ring",
            "protection_racket",
            "wealthy_elite",
            "union",
            "gang",
        ],
        item_types=[
            "incriminating_evidence",
            "revolver",
            "blackmail_photos",
            "forged_documents",
            "hidden_fortune",
            "murder_weapon",
            "surveillance_recordings",
            "coded_message",
        ],
        concept_types=[
            "moral_ambiguity",
            "corruption",
            "redemption",
            "betrayal",
            "justice",
            "the_past",
        ],
    ),
    relationship_patterns=[
        "owes_money_to",
        "former_partner",
        "blackmailing",
        "protective_of",
        "double_crossing",
        "affair_with",
    ],
    naming_style="Use period-appropriate names. Mix ethnic backgrounds. "
    "Nicknames are common, especially for criminals.",
    recommended_counts={
        "characters": (4, 10),
        "locations": (3, 7),
        "factions": (2, 5),
        "items": (3, 7),
        "concepts": (2, 4),
    },
    atmosphere="Dark, moody, and morally complex. Rain-slicked streets, cigarette smoke, "
    "and secrets that everyone wants to keep buried.",
    tags=["crime", "noir", "mystery", "detective", "dark"],
)


# =============================================================================
# HISTORICAL FICTION
# =============================================================================

HISTORICAL_FICTION = WorldTemplate(
    id="historical_fiction",
    name="Historical Fiction",
    description="Period-accurate settings with fictional characters and events",
    is_builtin=True,
    genre="historical",
    entity_hints=EntityHints(
        character_roles=[
            "noble",
            "servant",
            "soldier",
            "merchant",
            "scholar",
            "religious_figure",
            "spy",
            "revolutionary",
            "artisan",
            "peasant",
        ],
        location_types=[
            "manor_house",
            "market_square",
            "battlefield",
            "palace",
            "monastery",
            "tavern",
            "port",
            "workshop",
            "parliament",
        ],
        faction_types=[
            "noble_house",
            "military_regiment",
            "religious_order",
            "guild",
            "royal_court",
            "revolutionary_cell",
            "merchant_company",
            "spy_network",
        ],
        item_types=[
            "family_heirloom",
            "sealed_letter",
            "period_weapon",
            "valuable_cargo",
            "historical_document",
            "religious_relic",
            "map",
            "jewels",
        ],
        concept_types=[
            "honor",
            "duty",
            "class_struggle",
            "faith",
            "progress",
            "tradition",
        ],
    ),
    relationship_patterns=[
        "sworn_to",
        "married_to",
        "servant_of",
        "rival_house",
        "secret_lover",
        "apprentice_of",
    ],
    naming_style="Use historically appropriate names for the era and region. "
    "Titles and ranks are important markers of status.",
    recommended_counts={
        "characters": (4, 10),
        "locations": (3, 8),
        "factions": (2, 5),
        "items": (2, 6),
        "concepts": (2, 4),
    },
    atmosphere="Authentic to the period but accessible to modern readers. "
    "Social structures, customs, and conflicts of the era shape every interaction.",
    tags=["historical", "period", "drama", "authentic", "social"],
)


# =============================================================================
# BLANK CANVAS
# =============================================================================

BLANK_CANVAS = WorldTemplate(
    id="blank_canvas",
    name="Blank Canvas",
    description="No hints - full creative freedom for unique worlds",
    is_builtin=True,
    genre="any",
    entity_hints=EntityHints(
        character_roles=[],
        location_types=[],
        faction_types=[],
        item_types=[],
        concept_types=[],
    ),
    relationship_patterns=[],
    naming_style="Create your own naming conventions that fit your unique world.",
    recommended_counts={},
    atmosphere="Define your own atmosphere - let the story brief guide the world's mood.",
    tags=["custom", "creative", "freeform"],
)


# =============================================================================
# BUILT-IN TEMPLATES REGISTRY
# =============================================================================

BUILTIN_WORLD_TEMPLATES: dict[str, WorldTemplate] = {
    "high_fantasy": HIGH_FANTASY,
    "cyberpunk": CYBERPUNK,
    "urban_fantasy": URBAN_FANTASY,
    "space_opera": SPACE_OPERA,
    "post_apocalyptic": POST_APOCALYPTIC,
    "noir_detective": NOIR_DETECTIVE,
    "historical_fiction": HISTORICAL_FICTION,
    "blank_canvas": BLANK_CANVAS,
}


def get_world_template(template_id: str) -> WorldTemplate | None:
    """Get a world template by ID.

    Args:
        template_id: The template ID to look up.

    Returns:
        The world template if found, None otherwise.
    """
    template = BUILTIN_WORLD_TEMPLATES.get(template_id)
    if template:
        logger.debug(f"Retrieved world template: {template_id}")
    else:
        logger.warning(f"World template not found: {template_id}")
    return template


def list_world_templates() -> list[WorldTemplate]:
    """List all built-in world templates.

    Returns:
        List of all world templates.
    """
    templates = list(BUILTIN_WORLD_TEMPLATES.values())
    logger.debug(f"Listed {len(templates)} world templates")
    return templates
