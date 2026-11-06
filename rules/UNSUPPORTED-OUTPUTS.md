# Map outputs no tagged record carries

Input for the T124 map rerun. For each trained rule of `rules/heads/`, the outputs its map entry names that none of its tagged records carries, so `build-corpus` leaves them out of the rule's effective outputs and neither the rule mask nor the rule tests read them (research R1, FR-060a). A rule whose every output is listed has no effective output and is named untested by the selection-set freeze.

Measured on `output/effects/corpus-gen3-pilot3-roles` with `scripts/rule_entity_audit.py --unsupported`; the tagged-record count is summed over the families that tag the rule.

| rule | outputs not carried | effective outputs | tagged by | tagged records | why |
|---|---|---|---|---|---|
| 101.4 | trigger_fired, life_delta | zone_outcome | outcome_event | 54 | its tagged records never supervise it at the tag's outcome entities |
| 105.3 | type_color_duration | colors_gained, colors_lost | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 106.3 | zone_outcome | mana_delta_b | outcome_event, verdict_bit | 2273 | a decision record supervises its verdict and target legality alone |
| 107.4f | life_delta | can_play, affordable | verdict_bit | 3 | a decision record supervises its verdict and target legality alone |
| 107.16a | trigger_fired | can_play | verdict_bit | 38 | a decision record supervises its verdict and target legality alone |
| 111.2 | trigger_fired | created_objects | outcome_event | 106 | its tagged records never supervise it at the tag's outcome entities |
| 111.5 | gate | created_objects | outcome_event | 106 | its tagged records never supervise it at the tag's outcome entities |
| 111.10a | zone_outcome, mana_delta_g | created_objects | outcome_event | 106 | its tagged records never supervise it at the tag's outcome entities |
| 111.10b | affordable | created_objects, life_delta | outcome_event | 7 | its tagged records never supervise it at the tag's outcome entities |
| 111.10f | affordable | created_objects, cards_drawn | outcome_event | 9 | its tagged records never supervise it at the tag's outcome entities |
| 111.10g | affordable | created_objects, cards_discarded, cards_drawn | outcome_event | 6 | its tagged records never supervise it at the tag's outcome entities |
| 111.10q | trigger_fired, life_delta, zone_outcome | power_delta, toughness_delta | continuous_contribution | 22 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 111.10s | counters_delta_p1p1, affordable | created_objects, library_events | outcome_event | 4 | its tagged records never supervise it at the tag's outcome entities |
| 111.10w | zone_outcome, gate | (none) | outcome_event | 106 | its tagged records never supervise it at the tag's outcome entities |
| 111.12 | gate | created_objects | outcome_event | 22 | its tagged records never supervise it at the tag's outcome entities |
| 112.1 | zone_outcome | has_legal_target, can_play, target_legal | resolution_outcome, verdict_bit | 3333 | the tagged record is the cost half, which holds no events; a decision record supervises its verdict and target legality alone |
| 113.6k | damage_taken | trigger_fired | trigger_mode | 9 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 116.2c | blocker_legal | (none) | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
| 118.11 | life_delta, can_play | (none) | rewrite_event | 77 | a rewrite that ran another ability carries no per-entity target |
| 119.8 | life_delta, toughness_delta | can_play | verdict_bit | 12 | a decision record supervises its verdict and target legality alone |
| 119.9 | life_delta | trigger_fired | trigger_mode | 4 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 120.4b | trigger_fired | damage_taken, life_delta | rewrite_event | 65 | a rewrite that ran another ability carries no per-entity target |
| 120.6 | tap_state | damage_taken, zone_outcome | sba_zone_move | 152 | its tagged records never supervise it at the tag's outcome entities |
| 120.9 | life_delta | trigger_fired | trigger_mode | 2 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 121.1 | zone_outcome | cards_drawn | outcome_event | 109 | its tagged records never supervise it at the tag's outcome entities |
| 121.2 | trigger_fired | cards_drawn, life_delta | outcome_event | 109 | its tagged records never supervise it at the tag's outcome entities |
| 121.2a | cards_drawn | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 121.6a | life_delta, cards_drawn | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 121.6b | life_delta, cards_drawn | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 121.6c | cards_drawn, counters_delta_loyalty | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 121.7 | trigger_fired | cards_drawn, damage_taken, life_delta | outcome_event, rewrite_event | 168 | a rewrite that ran another ability carries no per-entity target |
| 122.1d | counters_delta_stun, tap_state | (none) | rewrite_event | 14 | a rewrite that ran another ability carries no per-entity target |
| 122.1h | zone_outcome | (none) | rewrite_event | 77 | a rewrite that ran another ability carries no per-entity target |
| 202.2f | type_color_duration | colors_gained, colors_lost | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 304.2 | life_delta | zone_outcome | outcome_event | 39 | its tagged records never supervise it at the tag's outcome entities |
| 305.7 | affordable, can_play | (none) | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 307.2 | life_delta | zone_outcome | outcome_event | 39 | its tagged records never supervise it at the tag's outcome entities |
| 308.1 | life_delta | can_play, zone_outcome | outcome_event, verdict_bit | 3227 | a decision record supervises its verdict and target legality alone |
| 310.6 | counters_delta_defense, damage_taken | (none) | outcome_event | 163 | its tagged records never supervise it at the tag's outcome entities |
| 400.6 | zone_outcome | (none) | rewrite_event | 77 | a rewrite that ran another ability carries no per-entity target |
| 400.7e | zone_outcome | trigger_fired | trigger_mode | 62 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 400.7f | zone_outcome | trigger_fired | trigger_mode | 3 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 405.5 | zone_outcome, damage_taken | (none) | resolution_outcome | 3309 | the tagged record is the cost half, which holds no events |
| 405.6d | zone_outcome | can_play | verdict_bit | 16 | a decision record supervises its verdict and target legality alone |
| 405.6f | has_legal_target | zone_outcome | sba_zone_move | 152 | its tagged records never supervise it at the tag's outcome entities |
| 406.2 | life_delta | zone_outcome | outcome_event | 71 | its tagged records never supervise it at the tag's outcome entities |
| 508.2b | life_delta | trigger_fired | trigger_mode | 110 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 508.3a | life_delta | trigger_fired | trigger_mode | 110 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 508.3b | life_delta | trigger_fired | trigger_mode | 30 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 508.3c | life_delta | trigger_fired | trigger_mode | 30 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 508.3d | life_delta | trigger_fired | trigger_mode | 30 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 508.3e | created_objects | trigger_fired | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 508.3f | poison_delta | trigger_fired | trigger_mode | 12 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.1i | life_delta | trigger_fired | trigger_mode | 13 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.2a | life_delta | trigger_fired | trigger_mode | 13 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3a | counters_delta_p1p1 | trigger_fired | trigger_mode | 14 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3b | created_objects | trigger_fired | trigger_mode | 4 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3c | life_delta | trigger_fired | trigger_mode | 13 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3d | power_delta, toughness_delta | trigger_fired | trigger_mode | 16 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3e | keywords_gained | trigger_fired | trigger_mode | 27 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3f | zone_outcome | trigger_fired | trigger_mode | 16 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 509.3g | poison_delta | trigger_fired | trigger_mode | 12 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 510.3a | counters_delta_p1p1, life_delta | trigger_fired | trigger_mode | 44 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 514.1 | cards_discarded | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 602.2a | zone_outcome, damage_taken, life_delta | (none) | resolution_outcome | 3309 | the tagged record is the cost half, which holds no events |
| 603.3a | life_delta | (none) | trigger_mode | 525 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 603.4 | gate | trigger_fired | trigger_mode | 268 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 603.5 | life_delta, gate | (none) | trigger_mode | 162 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 604.5 | can_play | zone_outcome | outcome_event | 27 | its tagged records never supervise it at the tag's outcome entities |
| 607.2i | life_delta, zone_outcome | trigger_fired | trigger_mode | 7 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 607.2k | life_delta | gate, zone_outcome | outcome_event | 26 | its tagged records never supervise it at the tag's outcome entities |
| 607.4 | mana_delta_r, life_delta | trigger_fired | trigger_mode | 8 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 608.2c | tap_state | zone_outcome | outcome_event | 23 | its tagged records never supervise it at the tag's outcome entities |
| 608.2f | trigger_fired | zone_outcome, life_delta | outcome_event | 121 | its tagged records never supervise it at the tag's outcome entities |
| 611.2c | pt_duration | power_delta, toughness_delta | rewrite_event | 65 | a rewrite that ran another ability carries no per-entity target |
| 611.2e | colors_gained | trigger_fired | trigger_mode | 525 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 611.3b | zone_outcome | power_delta | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 611.3c | cards_drawn | trigger_fired | trigger_mode | 525 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 613.1b | control_change | power_delta, toughness_delta | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 613.1e | type_color_duration | colors_gained, colors_lost, power_delta, toughness_delta | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 613.5 | type_color_duration | colors_gained, colors_lost | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 613.6 | control_change, tap_state | keywords_gained, types_gained, power_delta, toughness_delta | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 614.3 | zone_outcome, tap_state | (none) | rewrite_event | 9 | a rewrite that ran another ability carries no per-entity target |
| 614.4 | zone_outcome, tap_state | (none) | rewrite_event | 9 | a rewrite that ran another ability carries no per-entity target |
| 614.6 | trigger_fired | zone_outcome | rewrite_event | 142 | a rewrite that ran another ability carries no per-entity target |
| 614.8 | zone_outcome, tap_state, damage_taken | trigger_fired | rewrite_event, trigger_mode | 53 | a rewrite that ran another ability carries no per-entity target; a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 614.11a | cards_milled, cards_drawn, zone_outcome | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 614.11b | cards_milled, cards_discarded, zone_outcome | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 614.12 | tap_state | (none) | rewrite_event | 293 | a rewrite that ran another ability carries no per-entity target |
| 614.17d | zone_outcome | (none) | rewrite_event | 293 | a rewrite that ran another ability carries no per-entity target |
| 700.3c | zone_outcome, trigger_fired | (none) | outcome_event | 5 | its tagged records never supervise it at the tag's outcome entities |
| 700.13 | counters_delta_p1p1 | trigger_fired | trigger_mode | 12 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 701.5a | zone_outcome | can_play, affordable | resolution_outcome, verdict_bit | 5078 | the tagged record is the cost half, which holds no events; a decision record supervises its verdict and target legality alone |
| 701.6a | zone_outcome, life_delta | (none) | resolution_outcome | 3309 | the tagged record is the cost half, which holds no events |
| 701.7b | created_objects | (none) | rewrite_event | 293 | a rewrite that ran another ability carries no per-entity target |
| 701.8b | tap_state | zone_outcome | sba_zone_move | 152 | its tagged records never supervise it at the tag's outcome entities |
| 701.9c | zone_outcome, life_delta | (none) | outcome_event | 33 | its tagged records never supervise it at the tag's outcome entities |
| 701.16a | zone_outcome | created_objects | outcome_event | 8 | its tagged records never supervise it at the tag's outcome entities |
| 701.19b | zone_outcome, tap_state | (none) | rewrite_event | 9 | a rewrite that ran another ability carries no per-entity target |
| 701.24g | trigger_fired | (none) | outcome_event | 89 | its tagged records never supervise it at the tag's outcome entities |
| 701.26a | tap_state | trigger_fired | trigger_mode | 26 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 701.26b | tap_state | trigger_fired | trigger_mode | 23 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 701.27a | zone_outcome | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.27c | gate | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.27f | zone_outcome | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.28a | zone_outcome | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.28c | gate | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.28e | zone_outcome | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.28f | zone_outcome | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.34a | poison_delta | counters_delta_p1p1 | outcome_event | 4 | its tagged records never supervise it at the tag's outcome entities |
| 701.35a | attacker_legal | blocker_legal, can_play | legality_membership, verdict_bit | 3161 | a playability record supervises the legality fields alone; a decision record supervises its verdict and target legality alone |
| 701.44d | counters_delta_p1p1 | (none) | outcome_event | 10 | its tagged records never supervise it at the tag's outcome entities |
| 701.47a | counters_delta_p1p1 | created_objects | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 701.47b | trigger_fired | (none) | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 701.47c | counters_delta_p1p1, damage_taken | (none) | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 701.50a | cards_drawn, cards_discarded | counters_delta_p1p1 | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 701.50c | cards_drawn, cards_discarded, counters_delta_p1p1 | (none) | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 701.50d | cards_drawn, cards_discarded | counters_delta_p1p1 | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 701.58g | zone_outcome, trigger_fired | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 701.66a | target_legal, trigger_fired | counters_delta_p1p1, types_gained, keywords_gained, power_delta | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 701.70a | cards_drawn, cards_discarded | created_objects | outcome_event | 3 | its tagged records never supervise it at the tag's outcome entities |
| 702.6c | attached_to_change, keywords_lost | target_legal | legality_membership | 504 | a playability record supervises the legality fields alone |
| 702.6e | attached_to_change | target_legal | legality_membership | 504 | a playability record supervises the legality fields alone |
| 702.15e | life_delta | trigger_fired | trigger_mode | 4 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.16d | attached_to_change | (none) | outcome_event | 4 | its tagged records never supervise it at the tag's outcome entities |
| 702.19d | counters_delta_loyalty | damage_taken, life_delta | keyword_combat | 106 | its tagged records never supervise it at the tag's outcome entities |
| 702.21a | trigger_fired, zone_outcome | (none) | resolution_outcome | 739 | the tagged record is the cost half, which holds no events |
| 702.23a | trigger_fired, power_delta, toughness_delta, pt_duration | (none) | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 702.23b | power_delta, toughness_delta, pt_duration | (none) | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 702.24a | trigger_fired, tap_state | counters_delta_other, zone_outcome | outcome_event | 25 | its tagged records never supervise it at the tag's outcome entities |
| 702.24b | trigger_fired, tap_state | counters_delta_other, zone_outcome | outcome_event | 25 | its tagged records never supervise it at the tag's outcome entities |
| 702.25a | power_delta, toughness_delta, pt_duration | trigger_fired | trigger_mode | 1 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.25b | power_delta, toughness_delta, pt_duration | trigger_fired | trigger_mode | 1 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.29a | can_play | cards_discarded, cards_drawn | outcome_event | 37 | its tagged records never supervise it at the tag's outcome entities |
| 702.30a | zone_outcome, tap_state | trigger_fired | trigger_mode | 18 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.43b | trigger_fired | counters_delta_p1p1 | outcome_event | 8 | its tagged records never supervise it at the tag's outcome entities |
| 702.45a | power_delta, toughness_delta, pt_duration | trigger_fired | trigger_mode | 8 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.46a | zone_outcome | trigger_fired | trigger_mode | 14 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.49a | tap_state | zone_outcome, gate | outcome_event | 4 | its tagged records never supervise it at the tag's outcome entities |
| 702.51a | tap_state, mana_paid | affordable | verdict_bit | 7 | a decision record supervises its verdict and target legality alone |
| 702.51b | mana_paid | affordable | verdict_bit | 7 | a decision record supervises its verdict and target legality alone |
| 702.52a | cards_milled, cards_drawn, zone_outcome | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 702.55a | zone_outcome | trigger_fired | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 702.62a | zone_outcome, counters_delta_time, keywords_gained, mana_paid | (none) | trigger_mode, verdict_bit | 60 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record; a decision record supervises its verdict and target legality alone |
| 702.63b | counters_delta_time, zone_outcome | (none) | trigger_mode | 9 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.63c | counters_delta_time, zone_outcome | (none) | trigger_mode | 9 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.66a | zone_outcome, mana_paid | affordable | verdict_bit | 2 | a decision record supervises its verdict and target legality alone |
| 702.66b | mana_paid | affordable | verdict_bit | 2 | a decision record supervises its verdict and target legality alone |
| 702.68a | power_delta, pt_duration | trigger_fired | trigger_mode | 12 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.74a | trigger_fired, mana_paid | zone_outcome, affordable, can_play | verdict_bit | 13 | a decision record supervises its verdict and target legality alone |
| 702.77a | counters_delta_p1p1, zone_outcome | can_play | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
| 702.79a | trigger_fired | zone_outcome, counters_delta_m1m1 | outcome_event | 3 | its tagged records never supervise it at the tag's outcome entities |
| 702.80b | counters_delta_m1m1 | (none) | outcome_event | 163 | its tagged records never supervise it at the tag's outcome entities |
| 702.80c | counters_delta_m1m1 | (none) | outcome_event | 163 | its tagged records never supervise it at the tag's outcome entities |
| 702.83a | power_delta, toughness_delta, pt_duration | trigger_fired | trigger_mode | 10 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.84a | zone_outcome, keywords_gained, attacker_legal | can_play | trigger_mode, verdict_bit | 1772 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record; a decision record supervises its verdict and target legality alone |
| 702.86a | zone_outcome | trigger_fired | trigger_mode | 12 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.89a | zone_outcome, damage_taken | (none) | rewrite_event | 9 | a rewrite that ran another ability carries no per-entity target |
| 702.90d | counters_delta_m1m1 | (none) | outcome_event | 163 | its tagged records never supervise it at the tag's outcome entities |
| 702.92a | created_objects, attached_to_change | (none) | trigger_mode | 24 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.93a | zone_outcome, counters_delta_p1p1, control_change | trigger_fired | trigger_mode | 2 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.97a | zone_outcome, counters_delta_p1p1 | can_play | verdict_bit | 5 | a decision record supervises its verdict and target legality alone |
| 702.100a | counters_delta_p1p1 | trigger_fired | trigger_mode | 12 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.101a | life_delta, mana_paid | trigger_fired | trigger_mode | 10 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.103a | can_play, affordable, mana_paid | (none) | outcome_event | 5 | its tagged records never supervise it at the tag's outcome entities |
| 702.107a | tap_state | counters_delta_p1p1, can_play, affordable | outcome_event, verdict_bit | 3 | a decision record supervises its verdict and target legality alone |
| 702.108a | power_delta, toughness_delta, pt_duration | trigger_fired | trigger_mode | 33 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.108b | power_delta, toughness_delta, pt_duration | trigger_fired | trigger_mode | 33 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.110a | trigger_fired | zone_outcome | outcome_event | 1 | its tagged records never supervise it at the tag's outcome entities |
| 702.114a | target_legal, zone_outcome | colors_lost | continuous_contribution | 2049 | a continuous record supervises P/T, keyword, type and colour changes alone |
| 702.122a | tap_state, can_play, affordable | types_gained, type_color_duration | outcome_event | 31 | its tagged records never supervise it at the tag's outcome entities |
| 702.122e | types_gained | trigger_fired | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 702.123a | counters_delta_p1p1, created_objects | trigger_fired | trigger_mode | 28 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.123b | counters_delta_p1p1, created_objects | trigger_fired | trigger_mode | 28 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.126a | tap_state, mana_paid | can_play, affordable | verdict_bit | 1 | a decision record supervises its verdict and target legality alone |
| 702.126b | mana_paid | can_play, affordable | verdict_bit | 1 | a decision record supervises its verdict and target legality alone |
| 702.126c | mana_paid | can_play, affordable | verdict_bit | 1 | a decision record supervises its verdict and target legality alone |
| 702.128a | zone_outcome, created_objects | can_play, affordable | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
| 702.129a | zone_outcome | created_objects, can_play, affordable | outcome_event, verdict_bit | 1770 | a decision record supervises its verdict and target legality alone |
| 702.133a | cards_discarded | can_play, zone_outcome | outcome_event, verdict_bit | 1774 | a decision record supervises its verdict and target legality alone |
| 702.134a | counters_delta_p1p1 | trigger_fired | trigger_mode | 10 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.134b | counters_delta_p1p1 | trigger_fired | trigger_mode | 10 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.135a | created_objects | trigger_fired | trigger_mode | 4 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.135b | created_objects | trigger_fired | trigger_mode | 4 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.138a | mana_paid | can_play, affordable | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 702.141a | zone_outcome, life_delta | created_objects, can_play, affordable | outcome_event, verdict_bit | 3 | a decision record supervises its verdict and target legality alone |
| 702.145b | zone_outcome | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 702.147a | zone_outcome | blocker_legal | legality_membership, trigger_mode | 1190 | a playability record supervises the legality fields alone; a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.148a | zone_outcome | has_legal_target | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
| 702.150a | counters_delta_loyalty | (none) | rewrite_event | 293 | a rewrite that ran another ability carries no per-entity target |
| 702.151a | target_legal, can_play | attached_to_change | outcome_event | 4 | its tagged records never supervise it at the tag's outcome entities |
| 702.171a | tap_state | can_play | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
| 702.174a | keywords_gained | created_objects | outcome_event | 11 | its tagged records never supervise it at the tag's outcome entities |
| 702.174b | trigger_fired, zone_outcome | cards_drawn | outcome_event | 332 | its tagged records never supervise it at the tag's outcome entities |
| 702.174j | zone_outcome | (none) | resolution_outcome | 3327 | the tagged record is the cost half, which holds no events |
| 702.175a | created_objects | trigger_fired | trigger_mode | 14 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.176a | counters_delta_time | types_lost, trigger_fired | continuous_contribution, trigger_mode | 2059 | a continuous record supervises P/T, keyword, type and colour changes alone; a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.180a | can_play, affordable | zone_outcome | outcome_event | 29 | its tagged records never supervise it at the tag's outcome entities |
| 702.182a | created_objects, attached_to_change | trigger_fired | trigger_mode | 14 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.183a | damage_taken, zone_outcome | can_play, affordable | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
| 702.189a | mana_delta_r, affordable | trigger_fired | trigger_mode | 8 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 702.192a | trigger_fired | zone_outcome, counters_delta_p1p1 | outcome_event | 5 | its tagged records never supervise it at the tag's outcome entities |
| 703.4a | zone_outcome | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 703.4c | tap_state | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 703.4d | cards_drawn | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 703.4f | counters_delta_lore | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 703.4n | cards_discarded | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 704.5g | tap_state | zone_outcome | sba_zone_move | 152 | its tagged records never supervise it at the tag's outcome entities |
| 706.2 | zone_outcome | life_delta | outcome_event | 7 | its tagged records never supervise it at the tag's outcome entities |
| 706.3a | zone_outcome | library_events, gate | outcome_event | 7 | its tagged records never supervise it at the tag's outcome entities |
| 707.10 | life_delta | trigger_fired | trigger_mode | 162 | a trigger record supervises `trigger_fired` alone; what the trigger does arrives as its resolution's record |
| 707.10b | counters_delta_p1p1 | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 712.9 | zone_outcome, gate | (none) | outcome_event | 17 | its tagged records never supervise it at the tag's outcome entities |
| 714.3c | trigger_fired, counters_delta_lore | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 715.3c | life_delta, has_legal_target | (none) | (untagged) | 0 | no record of the curated corpus is tagged with it (tagged only on `turn_action` records, which no stratum holds, or a protection rule drawn for copies alone) |
| 715.4 | has_legal_target, target_legal | zone_outcome | outcome_event | 185 | its tagged records never supervise it at the tag's outcome entities |
| 717.6 | zone_outcome | (none) | rewrite_event | 77 | a rewrite that ran another ability carries no per-entity target |
| 719.3b | types_gained, power_delta, toughness_delta | can_play | verdict_bit | 1769 | a decision record supervises its verdict and target legality alone |
