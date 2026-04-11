"""Mapping of correct card names to their on-disk filenames.

Forge card scripts occasionally contain typos or non-standard naming
that differs from the canonical card name used in match outcomes.
This table maps the *sanitized* correct name to the *sanitized*
on-disk stem so the embedding loader can find the file.

Format:  correct_sanitized_name → actual_filename_stem
         or  correct_sanitized_name → letter_dir/actual_filename_stem
         (use dir/ prefix when the first-letter directory differs)
"""

FILENAME_CORRECTIONS: dict[str, str] = {
    "fanatical_strength": "fanatical_strenght",
    "mintstrosity": "minstrosity",
    "fear_of_surveillance": "fear_of_surveilance",
    "winter_cursed_rider": "winter_cursed_raider",
    "2_mace": "p/+2_mace",
    "summon_chocomog": "summon_choco_mog",
    "streetwise_negotiator": "streetwise_negotiatior",
    "village_watch": "scorned_villager_moonscarred_werewolf",
    "summon_gf_cerberus": "summon_g_f_cerberus",
    "fallaji_archaeologist": "fallaji_archeologist",
    "pathfinding_axejaw": "pathfinding_exejaw",
    "summon_gf_ifrit": "summon_g_f_ifrit",
    "vhal_candlekeep_researcher": "vhal_candelkeep_researcher",
    "voldaren_thrillseeker": "voldaren_thrilseeker",
    "sita_varma_masked_racer": "sita_varma_masker_racer",
    "spdr_piloted_by_peni": "sp_dr_piloted_by_peni",
    "minsc__boo_timeless_heroes": "minsc_boo_timeless_heroes",
    "duke_ulder_ravengard": "duke_ulder_ravenguard",
}
