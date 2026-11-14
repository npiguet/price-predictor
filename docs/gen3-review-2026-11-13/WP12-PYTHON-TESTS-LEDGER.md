# WP12 Python unit-test ledger (2026-11-14)

Paths below are relative to `tests/unit/effects/` unless they start with `tests/`.
Entry numbers in brackets are the line numbers of `WP12P-candidates.txt` as `WP12P-find.py` prints them.

## Source item: fork refusal tally

- written: `price_predictor.domain.worker_failures.FORK_REFUSALS_LINE`, `ForkRefusalCounts`, `fork_refusal_document`, `render_fork_refusals`; `WorkerLogScan/WorkerLogReport.fork_refusals`; `manifest_fields()["fork_refusals"]`; info line in both round reports; specs cli.md, spec.md FR-058j, effects/price_predictor CLAUDE.md. Commit e3bb7d7d. Tests seen red first.

## TP2 — decoding, schema, shards, sidecars

### DELETE
- [1] domain/test_records.py::TestPlayabilityCarriesNoPolicy (3) — deleted (no sibling; literal mirror of the dataclass field sets, a change detector)
- [2] domain/test_records.py::TestRewrite (2) — test 1 deleted (sibling: test_record_codec_wp7_wire.py::TestAnUnknownIncomingEvent::test_a_known_incoming_event_is_what_arrived pins `rewritten is True` on a golden); test 2 KEPT and strengthened: it is the only test of `rewritten is False` (it asserted `not rewritten`, which None also passes)
- [3] domain/test_records.py::test_the_land_play_limit_is_a_check_of_its_own_after_the_activation_limit — deleted (sibling: test_python_and_java_spell_the_playability_checks_alike_in_order pins order and spelling against Java; land_play None default read by test_verdict_bit)
- [4] TestRewriteResultTracksForge::test_the_two_partitions_only_hold_real_results — deleted (restates the two production frozensets)
- [5] TestGen2Flags::test_random_seat_defaults_to_on_policy — deleted (builder default)
- [6] TestFlagSignatures::test_an_ordinary_observation_carries_neither_flag — deleted (builder defaults; non-combat is_probe False in test_an_interventional_resolution_is_both)
- [7] TestFlagSignatures::test_a_probe_names_the_combat_it_mirrors — deleted (sibling: test_a_damage_step_probe_is_a_fork_that_intervened_in_nothing asserts is_probe; mirror_of guard in test_mirror_of_requires_the_fork_flag)
- [8] TestGen3Envelope::test_the_metadata_list_names_every_gen3_member — deleted (sibling: test_schema_compatibility.py::test_every_listed_metadata_field_is_flipped_here, exact set)
- [9] test_schema_compatibility.py::TestTheActorPlayerException::test_the_exception_covers_exactly_the_playability_subkinds — KEPT: it is the only reader of `_REDEFINED_IN_GEN2`; a new playability subkind must be weighed against the documented actor_player exception
- [10] TestRuleTwoEnumValuesMayBeAddedNotReused::test_an_existing_value_keeps_its_payload_type — deleted (one pair restating PAYLOAD_TYPES; golden decode tests read activation halves as ActivationPayload)
- [11] TestRuleThreeTiersAreAdditive (3) — deleted (enum values and builder defaults); rule 3 "absent means uncollected" was tested by none: written test_record_codec.py::TestRoundTrip::test_a_snapshot_naming_no_tier_reads_as_none_collected (seen red)
- [12] test_state_snapshot.py::test_a_gen2_snapshot_reads_no_gen3_member — deleted (builder snapshot, not a decode)
- [13] TestTheJavaGoldensRead::test_every_entity_reads_in_the_gen3_shape — deleted (siblings in the class read real values: pt, lines, owner/face/history)
- [14] test_a_player_carries_its_land_plays_and_unlimited_stays_apart_from_absent — KEPT: the only test of `_player_from_json`'s null-means-unlimited versus absent
- [15]-[18] test_record_codec.py TestRoundTrip: snapshot_survives_in_full, two_grant_channels, wire_field_global, every_record_kind_survives_a_line — deleted (siblings: test_every_record_kind_round_trips; the strengthened golden write-back test pins wire names and line bytes)
- [19] TestTheRewriteResult: three round trips deleted (in_place_edit, declined, every_result; siblings: test_every_record_kind_round_trips, golden write-back, TestRewriteResultTracksForge spelling pin); test_a_substitution_writes_null_rather_than_a_copy KEPT (pins wire null)
- [20] TestInternedProvenanceKeys::test_interned_records_equal_freshly_built_ones — deleted (round trip; identity tests stay)
- [21] TestGen2EnvelopeFields::test_random_seat_round_trips — deleted (sibling: test_every_gen2_record_writes_back_as_it_was_read over a fixture holding random_seat true)
- [22] [23] TestTheProbedPerturbation round trips — deleted (siblings: golden write-back over record-combat-probe-prevented; gen-2 write-back pops the observed combat's null probe members)
- [24] [25] [26] parsed/golden/gen3-fixture "survives a write and a read" — deleted (siblings: the byte-level write-back tests in the same classes)

### STRENGTHEN
- golden write-back (TestTheGen3EntityShape::test_a_golden_record_writes_back_as_the_collector_wrote_it): now `format_record_line` against the Java line byte for byte, normalizing only overlay keywords and counter types (the old `_spelling_free` nulled `granted_temporary` whole); seen red on a writer dropping a hidden keyword
- [27] test_sidecar_io.py::TestRowAlignment::test_row_i_is_the_ith_entry_of_lines — deleted (sibling: test_the_row_is_the_position_not_the_rendered_line_index, same assertion and more)
- [28] TestSerialization::test_a_written_sidecar_is_valid_json — STRENGTHENED (the coordinator's borderline case) into test_every_converter_written_sidecar_reads_and_writes_back_member_for_member: the 680 Java-written gen-3 fixture sidecars read and write back byte-equal in JSON; seen red on a reordered role span (the Python-written round trip passes it)
- [29] [30] [31] test_dropped_keys_survive, test_a_read_key_carries_the_headers_script_file, test_role_spans_and_sub_ability_links_survive — deleted (sibling: test_a_sidecar_round_trips, whole-object equality; the converter test above pins the JSON)
- [32] TestOptionKeys::test_a_gen1_sidecar_reads_with_no_option — deleted (sibling: test_a_sidecar_round_trips reads keys written without option; test_option_is_written_only_when_set)
- [33] test_shard_io.py::TestShardDirectory — deleted (siblings: test_both_spellings_load_together, test_shards_are_read_in_name_order read several shards)
- [34] TestShardsInSubdirectories::test_a_nested_gz_is_not_claimed_twice — deleted (sibling: test_it_finds_both_spellings_when_nested counts exactly 2, a double claim gives 3)
- [35] TestShardsInSubdirectories::test_the_order_is_stable_across_calls — STRENGTHENED, not deleted: it was the only test of iter_shards' sort over subdirectories; now pins the literal path order over both spellings interleaved (renamed ..._is_path_order_across_directories_and_spellings); seen red without `sorted` (the old one passed)
- [36] test_event_schema_completeness.py::TestVocabulary — deleted (sibling: TestTheJavaVocabularyBothWays pins every spelling to EventType.java)
- [37] test_an_event_accepts_the_params_its_type_normalizes — deleted (trivial; the reject test stays)
- [38] TestTheZoneChangeRow (3) — deleted (siblings: test_every_parameter_the_connector_writes_is_one_python_reads covers library_position; test_an_event_round_trips_through_its_dict_form)
- [39] TestTheCauseBearingTypes::test_any_type_may_still_carry_a_cause — deleted (the goldens carry `cause` on card_drawn, counter_change, damage_dealt, life_change and five more; every golden decode test reads them)
- [40] TestAttributionIsTriState — the three named deleted (four_states_are_distinct restates the first three; sentinel_not_mistakable reads constants; sentinel_survives_the_wire is a round trip); the four reading tests kept
- [41] [42] test_the_known_list_names_only_declared_types, test_the_ai_capability_gap_is_part_of_known_unemitted — deleted (test data checking test data; a misspelt KNOWN_UNEMITTED key fails test_no_declared_type_is_unreachable_by_surprise)
- [43] TestSupersededTypes — deleted (a re-added type fails test_every_java_event_type_is_a_python_one_and_back unless listed as unemitted)
- [44] test_fork_operation.py::test_the_catalogue_names_the_same_enum — KEPT: it is what makes the Java parity test cover rule_catalogue's ForkOperation (and [141] is deleted on its strength)
- [45] test_relation_types.py::test_the_table_s_families_are_all_there — deleted (siblings: test_java_and_python_spell_the_families_alike_in_table_order, test_the_order_is_frozen)
- [46] test_script_text.py::test_root_param_reads_every_fixture_line_as_its_former_parser_did — KEPT: the module says why (the parser effect families placed records with; a family that moved would re-place records a corpus already placed); it reads every real fixture line

### TP2 STRENGTHEN (commit cea7a233 unless noted)
- codec TestRoundTrip/ALL_KINDS/TestTheWireFormIsPinned gen-2 shape, digests blind — strengthened (ef7245e9): golden write-back compares format_record_line with the Java line byte for byte
- test_the_gen3_payload_members_read_into_their_fields asserts little — strengthened beside it: test_every_golden_payload_member_reads_into_the_field_of_its_name (every golden payload member, per kind, against the decoded field of its name; seen red on `fired=True`)
- TestTheGen3Fixture (12), test_state_snapshot TestTheGen3FixtureRead (10) on the superseded shape — skipped: waits for the fixture regeneration (later part of WP12)
- test_an_unknown_field_never_shadows_a_known_one — strengthened (an extra spelled like a known key; seen red on `out[key] = value`)
- test_a_turn_action_record_reads_from_the_collector_s_shape (hand-written) — skipped: the only turn_action golden is game_end (covered in wp7_wire); a draw/cleanup turn_action golden needs the Java package
- test_a_truncated_final_member_stops_the_read (one-line members) — strengthened beside it: test_a_block_cut_midway_keeps_every_whole_record_before_the_cut (seen red on a reader that buffers a member)
- test_the_head_is_read_off_a_written_line (Python-written) — strengthened beside it: test_the_head_is_read_off_every_line_the_java_writer_wrote
- TestCompressedShards no Java-written shard — written: TestJavaWrittenShards reads ShardCharacterizationTest's lines as a shard (records, marker, ended_games; seen red on a broken _GAME_END). A Java-written gzip member layout has no golden: needs the Java package
- test_no_declared_type_is_unreachable_by_surprise one direction — done-before (TestTheJavaVocabularyBothWays)
- schema-compat rules compare Python with literals — written: TestTheEnvelopeEnumsAreTheWriters (RecordKind, Moment, PlayabilitySubkind, TurnActionSubkind, AttributionMode, ActivationOutcome, TraitKind against their Java enums, in order)
- sidecar-keyword test omits gen3-sidecars — strengthened (gen3-sidecars added)
- test_a_parameter_is_dropped / test_a_keyword_line_reads_its_keyword_without_the_value pin the quality drop — done-before (WP9: qualities kept; a ward value is still dropped by design)
- test_a_temporary_grant_rides_the_overlay_not_the_tokens hand-built — strengthened beside it: a golden Cryptolith Rite grant is one overlay entry and no token
- test_provenance_resolution.py keys all the sidecar's own file — written: test_keys_alike_but_for_their_file_join_each_its_own_card_s_line over the 680 converter sidecars. FOUND: every chapter line of a saga carries every chapter's key (230 sagas in output/cardsfolder), so chapters II and III join chapter I's line: test_a_saga_s_chapter_trigger_joins_its_own_chapter_line xfail(strict) WP12-N1 (converter, Java)
- two line_or_none tests pin the uncounted swallow — done-before (WP9 counts it)
- TestSerialization Python-written only — strengthened (ef7245e9)
- test_shard_generation.py no generation-key cases — done-before (WP9)
- test_effect_families.py gen-1 only, truthiness — strengthened: the truthiness test now checks a name or sentinel; written test_a_gen3_record_takes_its_family_through_the_gen3_line_model
- test_records_key_sites.py misses floating effects/delayed triggers/link ends — written two tests (goldens without entities; a delayed trigger), seen red
- test_every_card_names_its_owner_face_and_history all-front — strengthened beside it: every face CardFacts.faceState writes decodes as its FaceState (the goldens hold only `front`)
- test_shard_streaming.py docstring promises four properties — strengthened: docstring says what the file tests and where the four properties are tested; dead `_held()` removed

### TP2 Top missing
- decode each Java characterization record and assert payload values per kind — written (above)
- refuse generation 3 in gen-2 payload shape — done-before (test_record_codec_wp4_wire.py::TestTheSupersededShape; the switch flips with the fixture)
- corrupt middle gzip member; located decode errors — done-before (TestDamagedShards)
- closed-list checks at decode; absent mode — done-before (test_wire_closed_lists.py)
- Java EventType both ways — done-before; EVENT_PARAMS Java ⊆ Python done-before; Python ⊆ Java skipped: 21 declared params are written by no `.param("…")` call, and run-parameter events name theirs through RunParamNormalizer's tables, so the writer's set cannot be read off the source reliably
- null scalars — skipped: the Java writer writes primitives there; what a null should read as is undecided
- ShardGeneration.of other values — done-before

### TP2 SUPPORT
- builders' gen-2-shaped defaults / make_gen3_record — skipped: moves ~45 files at once; belongs with the fixture regeneration
- `_EVENT` identity life_change — skipped (tidying)
- dead `_held()` in test_shard_streaming.py — done

## TP1 — collection orchestration, supervision, CLI

### DELETE
- [47] test_collect_coverage.py::TestExclusionsWithoutACheckpoint::test_both_sources_on_a_config_are_refused_when_it_runs — deleted (sibling: test_both_sources_at_once_is_refused; the candidate's DeckHoldout(...) raises in its constructor before run is reached, the same assertion)
- [48] TestParallelCounting::test_already_counted_shards_stay_counted_once — deleted (sibling: TestIncrementalCounting::test_a_second_pass_over_the_same_shards_adds_nothing; dedup is _ShardCounter._count_fresh's, before any worker)
- [49] TestCountingReportsProgress::test_the_held_out_round_reports_progress_too — KEPT: the only test that TextCoverageCounter's constructor hands progress_every to the shared counter (the bug it was written for)
- [50] TestHeldOutTextUnit::test_satisfaction_is_distinct_games_at_the_floor — deleted (sibling: test_the_round_ends_without_waiting_on_uncastable_texts completes at exactly five distinct games)
- [51] test_collect_variants.py::TestTheSeed::test_the_seed_is_settable_so_a_second_run_draws_differently — deleted (dataclass/argparse; sibling test_two_seeds_perturb_the_same_script_differently)
- [52] TestVariantGeneration::test_a_variant_is_never_converted_to_prose — deleted (sibling: test_a_variant_is_written_with_its_own_name finds the case-sensitive `Name:` line)
- [53] TestVariantGeneration::test_no_variant_is_generated_from_a_held_out_card — KEPT: the only test of generate_variants' held-out rule, the one the module calls silently wrong
- [54] TestOnlyLoadableVariantsAreDecked::test_the_list_is_no_card_script — STRENGTHENED: asserts the property (the list is no `.txt`) instead of the spelling, which its sibling pins to the Java writer
- [55] TestTheVariantCandidateMatchesAConvertedCard::test_a_script_without_a_cost_names_none — deleted (sibling: test_script_variants.py, script_mana_cost without ManaCost is None; deck_candidate passes it through)
- [56] test_coverage.py::test_satisfied_is_records_at_or_above_target — deleted (siblings: test_a_satisfied_card_drops_out_of_the_draw at 50/50, test_a_card_one_short_still_gets_drawn at 49)
- [57] TestConsultRanksButNeverDrops::test_no_card_is_ever_removed_by_the_consult — deleted (sibling: test_an_uncastable_card_keeps_a_weight)
- [58] test_rarity.py::test_the_buckets_are_in_ascending_order — deleted (restates the constant; test_every_edge_lands_in_its_bucket pins every edge)
- [59] test_sampling.py::TestRarityWeighting::test_a_rarer_text_weighs_more — deleted (siblings: exact inverse-square-root and below-cap values)
- [60] TestUniformClasses::test_a_decision_records_examples_key_on_the_candidates_text — deleted (sibling: test_combat_and_playability_legality_have_no_acting_line)
- [61] [63] [64] [65] [66] [67] cap defaults on the parser — deleted, REPLACED by TestSharedCapFlags::test_an_unflagged_run_hands_the_worker_every_cap_at_its_default (both collectors' resolved caps equal CollectionCaps' defaults, every field); the literal values not pinned elsewhere moved to test_collection_caps.py::test_the_contract_defaults_not_pinned_beside_their_own_rule (mana cap 1, two interventions, two probes, no random seat, no dropped-event counts)
- [62] TestSharedCapFlags::test_the_constants_and_the_parsed_defaults_agree — REPLACED by the test above (it read the parser only, not the resolved caps)
- [68]-[71] "flag reaches property" per field — deleted, REPLACED by test_each_cap_s_flag_reaches_its_worker_property parametrized over every CollectionCaps field and both collectors (directed caps on collect-variants); seen red on a resolution that pins mana_cap
- [72]-[79] "a removed flag stays removed" (9) — deleted (each passes for any never-defined name; the property set is pinned both ways by TestTheJvmReadsExactlyWhatThePythonSideWrites)
- [80] TestTheHeldOutCoverageRound::test_only_cards_beside_exclude_cards_is_refused — KEPT: the only test of _refuse_holdout_beside_only_cards
- [81] test_directed_fork_flags.py::TestTheCollectors::test_collect_coverage_runs_no_check — deleted (siblings: test_the_label_takes_no_startup_check, test_collect_coverage_takes_only_the_label)
- [82] test_holdout_cards.py::test_the_default_trees_are_the_card_and_token_trees — deleted (a dataclass default the CLI does not read; the CLI takes its trees from _add_cards_folder)
- [83] [84] test_cli_run_manifests.py::test_the_manifest_is_no_sidecar, test_reports_write_none — deleted (constants against constants; sibling test_shard_discovery_never_takes_the_manifest_for_a_shard)
- [85] test_card_sources.py::TestCorpusLoaders::test_card_files_are_keyed_by_the_name_line — KEPT and STRENGTHENED: the only test of card_files' value (the tree-prefixed script file); its fixture now spells the name lower-case as the converter does
- [86] test_collector_connector.py::test_records_only_mode_is_unaffected_by_the_log_file — done-before (gone)
- [87] test_forge_jvm.py::TestWorkerLogFiles::test_two_runs_never_share_a_file — deleted (sibling: test_get_creates_the_directory_and_a_file_named_by_run_and_worker pins `run-1.3.log`)
- [88] test_forge_worker_pool.py::test_the_error_comes_within_about_two_seconds — deleted (wall-clock; sibling bounds the attempts)
- [89] test_command_starts_with_java — deleted (a change detector: a resolved java path is no regression)
- [90] test_command_includes_classpath_flag — KEPT: the only test that the built classpath reaches the command
- [91] test_start_returns_popen_handle — KEPT: the only test of start()'s return, which the pool reads
- [92] test_jar_not_found_raises_file_not_found — deleted (restates propagation; the prerequisite check is tested on its own)
- [93] test_a_respawned_slot_is_launched_identically — KEPT: the only test catching any per-call value (time, random) in the command; test_the_lifetime_token_is_never_supplied_by_the_launcher checks one property name only
- [94] test_odd_values_accepted_including_large_ones — deleted (sibling: test_best_of_passed_as_system_property)
- [95] test_side_paths_omitted_when_none — KEPT: the absent case of the side-deck properties, like the decks-only and progress-file ones beside it; no other test sees a None reach the command
- [96] [97] [98] worker count stored — deleted (constructor argument read back)
- [99] [100] [101] side paths / best_of stored — deleted (siblings: test_side_paths_forwarded_to_connector, test_best_of_forwarded_to_connector; the CLI default weight is pinned in test_cli.py)
- [102] test_status_interval_still_sixty_seconds — KEPT: the interval is the recycle of runs without the stop protocol (_kill_oldest_worker: do not lengthen or remove it); the class docstring rewritten to say so
- [103] test_status_line_is_unchanged_by_the_extraction — deleted (sibling: test_forge_worker_pool.py pins the exact line)
- [104] [105] status counts output lines — deleted (siblings: test_forge_worker_pool.py output_line_count tests)
- [106] test_no_crash_when_no_workers — deleted (sibling: the pool's own no-worker _kill_oldest_worker test)
- [107] test_explicit_two_workers_spawns_two — deleted (siblings: test_spawns_correct_number_of_workers, test_single_worker_spawns_exactly_one)
- [108] test_no_side_paths_passes_none — deleted (sibling: test_side_paths_forwarded_to_connector; the connector's omission kept as [95])
- [109] test_run_id_forwarded_alongside_side_paths — deleted (siblings: test_run_id_stable_across_worker_restarts, test_side_paths_forwarded_to_connector)
- [110] test_cli_match_outcomes_effect_flags.py::TestOtherSealedCommandsAreUnaffected — deleted (hasattr of a never-defined flag)

### TP1 STRENGTHEN (commit 896d80d2 unless noted)
- SAFETY taskkill in test_match_outcomes.py; counter tests with workers=None — done-before (WP0)
- test_forge_worker_pool.py::TestStopCondition::test_no_recycle_when_stopping (branch never reached) — strengthened: should_stop answers no at the loop's poll and yes at the tick, so the tick's own check is reached; seen red with that check removed
- test_collect_coverage.py: test_the_run_finishes_reports_and_returns_zero, test_an_uninterrupted_run_plays_every_round_it_needs (exit 0 on zero records) — done-before (WP8: a record is written / exit 1)
- test_collector_connector.py: progress from a file, second round after a stop, decks file reaching the worker with every argument — done-before (WP8)
- test_collection_caps.py JVM-default regexes read 6 of 15 args — written: TestTheJvmDefaultsAreThePythonDefaultsArgumentByArgument (record components in Python field order; each positional default against Python's; aimed_at). FOUND STILL PRESENT: CollectionCaps.defaults() names no probe keyword (Python: all eight), so a worker started without the property probes nothing: test_the_probe_keywords_default_is_the_python_one xfail(strict) "TP1 strengthen / C5 readability"; Java fix for the TC4 package (CollectionCapsTest.theDefaultCapsAreTheContracts pins the empty list)
- property-name scan misses MatchWorkerMain's effect.* literals — written: test_match_worker_connector.py::TestEveryPropertyTheWorkerReadsIsOneTheLauncherWrites (a full launch writes exactly the properties MatchWorkerMain, HoldoutCards and WorkerStop declare)
- test_script_variants.py two tests — done-before (WP8 V4 #7: exact one-parameter diff, swap asserted unconditionally with a count)
- test_text_holdout.py permille 1000 / same-side / no literal crc — strengthened: test_a_text_s_verdict_is_its_crc_residue_against_the_permille pins two literal residues at the edge on both sides
- test_run_manifest.py::TestResume::test_appends_the_command_and_runs_again — done-before (WP8 V4 #4: per-invocation start and commits)
- tests/integration pilot skips — skipped: tests/integration is a separate package
- test_sampling.py::test_every_class_is_reachable — strengthened: every record shape of PAYLOAD_TYPES (turn actions aside) lands in a class, all eight reached
- test_one_shot_workers.py::test_a_failed_consult_yields_no_verdicts — strengthened: a non-zero exit that left an answer is still distrusted, and the warning is asserted; seen red without the exit-code check
- test_forge_jvm.py child-process log test with no child — strengthened: two real child processes write stdout and stderr through the slot's handle
- test_import_boundaries.py plain `import pkg.module` passes — strengthened: a plain import of a name-restricted module is a violation; seen red on a probe module

### TP1 Top missing
- kill_process_tree on a real child with a grandchild — done-before (test_a_real_child_and_its_grandchild_both_die)
- pool run when the loop raises — done-before (test_every_worker_is_killed_when_the_loop_raises)
- a round whose worker writes a cut shard — done-before (test_a_round_whose_last_game_was_cut_credits_nothing_and_says_so)
- zero-records coverage run exits non-zero and surfaces worker-log failures — done-before (WP8)
- caps end to end to the java command line — written: test_collector_connector.py::TestTheCapsReachTheJavaCommandLine (argv -> collector_config -> CollectorSupervisor -> real MatchWorkerConnector command); seen red on a supervisor sending default caps
- MatchWorkerConnector.start(variant_scripts=..., exclude_cards_path=...) — written (the full-launch test above)
- consult_castability itself — done-before (test_one_shot_workers.py, test_collect_coverage.py timed-out consult)
- CLI bounds — done-before (TestTheBounds)
- restarted collect-variants — done-before (--continues tests)
- _check_workers_can_start missing files; collect-variants manifest inputs — done-before (TestWorkersMustBeAbleToStart; test_cli_run_manifests.py::TestTheCollectorsListWhatTheyRead)

### TP1 SUPPORT
- fake_supervisor MagicMock(spec) — done: create_autospec for the instance and the constructor (175 tests pass under it)
- fakes cannot append progress, grow/cut a shard, exit, log, fail a deck — skipped: WP8/WP7 tests write logs, progress and cut shards from their pool stand-ins; a general fake would be churn
- _texts_by_game reimplements production — skipped (tidying)
- test_corpus_side_modules_never_load_torch omits the collector modules — skipped: each module is one more interpreter start in an already 53-interpreter test

## TP3 — corpus build, curation, derived records, selection set

### DELETE
- [111] test_damage_step_predicates.py::TestTheTableIsTheSingleSource::test_every_row_derives_its_predicate_from_its_effects — done-before (gone)
- [112] ::test_a_row_qualifies_exactly_when_an_effect_is_observable — deleted (restates `qualifies = any(effect.observable)`, through the test's own helpers)
- [113] ::test_the_participant_is_frozen — done-before (gone)
- [114] TestLifelink::test_a_zero_power_lifelinker_gains_nothing — deleted (sibling: TestDealsDamageAtAll::test_a_powerless_carrier_qualifies_for_nothing, lifelink among its parameters, both steps)
- [115] test_split.py::test_a_corpus_shaped_holdout_carries_its_texts_too — deleted (a test builder's round trip)
- [116] test_freeze_selection_set.py::test_rule_test_entities_come_from_the_tag — deleted (sibling: test_rule_test_entities_are_the_tag_s_outcome_entities, same assertion with the condition moved away)
- [117] test_corpus_curation.py::test_merging_equal_cap_heaps_with_overlapping_ranges — deleted (sibling: test_merging_two_heaps_keeps_the_cap_smallest_of_the_union)
- [118] TestOutcomeSignature::test_a_real_effect_halfs_signature_is_stable — done-before (gone)
- [119] test_build_corpus.py::test_build_writes_the_three_strata_and_a_manifest — deleted (siblings: the per-stratum count and manifest tests read the same build)
- [120] test_two_builds_with_one_seed_write_the_same_training_records — deleted (sibling: test_two_builds_of_one_corpus_agree, digest and ids)
- [121] test_the_retired_game_count_flag_is_rejected_by_argparse — deleted (a removed flag stays removed)
- [122] test_the_max_events_help_says_zero_disables_the_rule — KEPT: the only check that the documented "0 disables" reading reaches --help
- [123] test_build_corpus_decide.py::test_two_decisions_with_one_seed_are_identical — deleted (same process, same object; sibling test_budget_allocation.py::test_set_valued_members_are_ordered_independently_of_string_hashing across hash seeds)
- test_corpus_store.py::test_the_store_lays_out_the_dataset — KEPT by decision D5 (the on-disk corpus layout is frozen for gen 3)

## TP4 — validation, reports, rule detection

### DELETE
- [124] test_validate_corpus.py::test_every_named_invariant_is_reported (24 fragments) — REPLACED by the exact set: the 36 check names, in report order (_EVERY_CHECK)
- [125] TestRecordIdCollision::test_a_collision_planted_in_a_real_shard_directory_fails — deleted (siblings: test_a_planted_collision_fails_the_run, test_two_lifetimes_that_namespace_their_ids_pass reads the real directory)
- [126] TestKeywordKeysJoinTheirSidecar::test_a_planted_mismatch_in_a_real_shard_directory_is_rejected — deleted (sibling: test_a_planted_ordinal_mismatch_is_rejected)
- [127] [128] TestTheCli watched findings counted apart / never fail the run — deleted after moving both assertions ("watched, not judged", the zero-probe [WATCH] line) into test_a_clean_directory_exits_zero
- [129] test_a_runtime_only_trait_past_the_declared_tail_is_not_a_mismatch — deleted after moving `keyword.ok` into its sibling test_a_key_outside_every_declared_range_is_counted_not_joined (same board, same key)
- [130] test_a_card_with_no_sidecar_fails_rather_than_being_skipped — deleted after moving the "no sidecar" detail assertion into its sibling test_an_unconverted_script_is_not_read_as_outside_the_range (same orphan)
- [131] test_without_a_converted_tree_the_check_reports_unchecked — deleted (sibling: test_both_findings_say_so_when_no_tree_was_readable)
- [132] TestAHealthyWindowPasses::test_a_watched_finding_reports_a_number_and_judges_nothing — deleted (siblings: test_nothing_is_reported_broken, test_every_invariant_reports_a_measurement_not_just_a_verdict)
- [133] test_field_coverage.py::TestTheRatchet::test_a_coverage_row_reports_what_it_saw — deleted (a dataclass property)
- [134] TestTheRatchet::test_every_known_constant_names_a_field_the_schema_has — STRENGTHENED, not deleted (it could not fail: its reachable set was empty): every KNOWN_CONSTANT_FIELDS entry must be a path the walk reaches over the Java goldens (the list is empty today; the guard holds for the first entry)
- [135] TestTheRatchet::test_only_the_unvarying_fields_read_as_constant — deleted (siblings: TestWhatCountsAsConstant)
- [136] TestTheRewriteChannelIsNotExcused::test_the_walk_reaches_them — deleted (sibling: test_a_hook_that_passes_no_result_reads_as_a_constant_field indexes both paths)
- [137] test_cli_contract_validate_corpus.py::TestEveryThresholdHasAFlag::test_the_two_sets_are_the_same_set — deleted (sibling: test_the_parsed_defaults_rebuild_the_default_thresholds reads every field off the parsed args)
- [138] TestTheWatchedRatesTakeAFloor (4) — deleted (argparse float parsing; the floor's effect is test_validate_corpus.py::test_a_floor_passed_on_the_command_line_turns_a_watch_into_a_failure)
- [139] TestTheMirrorTurnCheckIsJudgedAtZero::test_it_can_be_widened — deleted (argparse float parsing; the zero default kept)
- [140] test_rule_catalogue.py::test_the_derived_output_list_follows_the_layout — deleted (restates head_outputs)
- [141] test_fork_operation_order_is_the_contract_order — deleted (sibling: test_fork_operation.py::test_java_and_python_spell_the_operations_alike, which covers the catalogue's enum through [44])
- [142] test_combat_keyword_effect.py::test_the_family_is_registered — deleted (a constant)
- [143] test_sba_zone_move.py::test_the_family_runs_on_gen2_records — deleted (a constant)

### TP4 STRENGTHEN (field coverage)
- test_every_schema_field_is_walked_without_being_listed (`... or True`) — strengthened: every equality field of EntityState, GlobalState and ResolutionPayload is walked (seen red on a walk skipping `history`)
- test_the_gen3_fixture_reaches_every_listed_path_and_varies_most excuses members the fixture predates — kept as is (fixture); written beside it test_the_members_the_fixture_predates_are_written_and_vary_on_the_goldens
- test_the_section_follows_the_constant_fields — strengthened: the section's header is after every constant row, after a blank line, before the event parameters

### TP3 STRENGTHEN (commits 25d2544d, b662ea43)
- test_damage_step_predicates.py step-blind rows; TestIndestructible/TestDeathtouch/TestTrample first-strike-step cases, indestructible opponent, deathtouch trampler — done-before (WP10: test_no_damage_changes_hands_in_a_step_nobody_strikes_in, TestTheCarriersOwnStep incl. test_an_indestructible_opponent_dies_to_no_hit and test_a_deathtouch_trampler_needs_assign_only_one_damage_a_blocker); the pinning test is gone
- test_lethality_counterfactuals.py::test_an_indestructible_or_enchanted_creature_is_not_copied first half vacuous — strengthened (a survivor that qualifies, refused only for the indestructible keyword; the Aura case beside its qualifying twin)
- PINNED_X / x_paid — done-before (WP10)
- TestQualify exclusions untested — written: each SOURCE_EXCLUSIONS keyword (wither, infect, lifelink, deathtouch), a shield counter (seen red), a non-creature hit, a cost half, a second damage segment with Defined$ Remembered, DamageSource$, DividedAsYouChoose$; fork/interventional/directed was done-before
- test_protection_copies.py TestAuraToGraveyard contract contradiction — done-before (decided: a fizzled Aura half holds its one move; the test asserts FIZZLED and the targets as cast)
- TestDamagePrevented::test_forks_and_synthetic_records_are_never_copied — strengthened (interventional and directed forks, on a record that qualifies)
- TestDraw::test_the_same_seed_draws_the_same_copy — strengthened (another seed draws another copy)
- test_grid_edits.py counts only — partly done-before (TestTheDamageGridsOutcomes checks each cell's frozen outcome against the rule; the five-colour refusal); the evasion success test still checks a count only — skipped (budget)
- test_freeze_selection_set.py::test_the_deviance_sample_is_capped_by_a_seeded_draw (passed for "first seven") — strengthened: the sample is no prefix of the half's records
- test_halves_are_the_manifests_disjoint_and_deterministic, test_sc010_on_the_fixture (thresholds 2.0, superseded fixture) — skipped: the sc010 test waits for the fixture regeneration; the halves test asserts four real properties now
- test_corpus_plan.py unknown game TRAINING — done-before (test_a_game_the_survey_never_saw_is_dropped_never_trained); 64-bit thresholds — kept (pinned literals are the point)
- test_corpus_curation.py TestConditionBalance _Lines fake — done-before (raises KeyError; responsible_static passed); TestGameDisjointDraw restated crc — strengthened (literal draw and hash pinned)
- test_rule_budgets.py text cap all tagged — done-before (test_the_text_cap_counts_the_texts_untagged_records_too)
- test_validation_samples.py quota unique ids — done-before (test_the_copies_of_one_record_take_one_place)
- test_corpus_manifest.py digest synthetic / gen-2 field count — skipped: a gen-3 digest guard needs a real gen-3 corpus (after the pilot); the count is 39 and each field asserted absent
- test_build_corpus.py per-stratum `> 0` — strengthened (each stratum's count equals the distinct records its directory holds); game-disjoint forge-AI-only, protection copies, grown corpus, repack, validation sample — kept (the survey shows they assert more than the review read, or need the fixture's random-seat games: not confirmed)
- test_build_corpus_survey.py rule-counts comparison / closed lists `<=` — skipped (budget): needs the survey and rule-counts run on one shard with one catalogue
- test_build_vocab.py `big > small` — skipped (budget)

### TP3 Top missing
- records dir changing between survey and write; undecodable line and cut gzip member; kill-cut game through the survey; missing rules/ refused before clearing — done-before (WP10, WP7)
- EvasionGrid five-colour pair — done-before (refusal case); one real grid_sources freeze — skipped (budget)
- write_selection_set twice with different content — done-before (test_selection_set.py "already holds")
- first-strike-step boards for the five rows — done-before
- protection copies for split/Adventure/MDFC — done-before for split (test_a_split_card_s_half_is_not_copied); mode-targeted records — skipped
- held-out game never TRAINING/GAME_DISJOINT property — skipped (property test; the unknown-game test covers the defect)
- aimed_at=() pinned across consumers — done-before (stratum placement, rule mask, collectors)

### TP4 STRENGTHEN (commit 20e92aa2 unless noted)
- _healthy() gen-2; TestOutcome gen-2 only — strengthened: test_on_gen3_the_effect_halves_carry_it; TestLinkHalvesPair lone gen-3 cost half / all missing — done-before (WP11)
- TestTriggerNegatives ceiling only; four "kind absent means ok" — done-before (WP11: "trigger records hold positives", TestEveryRecordKindIsPresent)
- TestOutgoingAgreesWithTheResult / test_results_that_run_no_ability_are_outside_the_rate pin "prevented never runs an ability" — done-before: WP7 put a prevention's rider in `buffered`, `replaced_by` stays empty by contract, test_a_prevention_with_a_rider_passes
- test_a_key_in_a_tree_this_run_was_not_given_is_unchecked_not_broken (ok at 100% unchecked) — done-before (now ..._fails_as_unchecked)
- TestGen3FieldPresence one member per site — strengthened: test_every_owed_snapshot_member_missing_is_named over every member of the five owed lists (29 cases)
- TestGen2FieldPresence ok only — strengthened (measured figures asserted)
- TestValidateDirectory always-truthy assert — strengthened (the envelope finding's ok and figure)
- test_field_coverage.py `... or True`, excused constants, section order — strengthened (b59612b2, see TP4 DELETE)
- test_rule_counts.py review list on an empty list — strengthened: a planted disagreeing record lands on it; aimed source — written (aimed_at present, empty, absent)
- test_legality_membership.py bare-list pins — done-before (WP11 per-rule declarations; the family semantics stay); four "either outcome" asserts — strengthened to the outcome they get (three; the fourth was already exact)
- combat_keyword_effect / keyword_combat (and four more) real-record tests skip silently — strengthened: they fail when the fixture lacks the case (8 skips)
- toxic/poisonous regular step only — done-before (WP11)
- test_resolution_outcome.py fizzled by editing a resolved half — done-before (test_a_fizzled_half_reads_its_targets_as_cast)
- test_continuous_contribution.py pt without value — done-before
- test_outcome_event.py combat filters forks out — strengthened (a probe's combat events read alike)
- test_sba_zone_move.py untouched pair, wither kill — done-before
- test_rule_outcomes.py split second — done-before (special actions allowed)

### TP4 Top missing
- whole-catalogue test; characterization records through tag_record; fork/interventional/degraded; presence per kind and fired floor; undecodable line; zero-tag rules; event params in field_coverage; gen-3 record-moment members — done-before (test_real_catalogue.py, test_catalogue_on_data.py, WP11)
- duplicate ids in rules_catalogue_io — written test-first and FIXED (dfc813ef): load_scenarios and load_heads refuse a duplicate naming both files (the real catalogue has none)

### TP4 SUPPORT
- RawShardWriter + _healthy() loop repeated; identical _write in two classes; substring assertions; TestTheCli setattr thresholds; _rules duplicated — skipped (tidying)
- stale __pycache__ pyc of test_rule_breadth — not touched (untracked cache)
