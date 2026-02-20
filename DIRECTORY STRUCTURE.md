DIRECTORY STRUCTURE
Directory structure:
└── t3rm1nus-hrm/
    ├── __init__.py
    ├── auto_learning_system.py
    ├── bootstrap.py
    ├── config_loader.py
    ├── execution_gateway.py
    ├── fix_l3_dominance.py
    ├── integration_auto_learning.py
    ├── runtime_loop.py
    ├── archived/
    │   ├── agent-modelfile.txt
    │   ├── auto_learning_config.json
    │   ├── convergence_config.json
    │   ├── global_system_state.json
    │   ├── initial_state.json
    │   └── sentiment_cache_timestamp.json
    ├── backtesting/
    │   ├── __init__
    │   ├── backtesting_utils.py
    │   ├── config.json
    │   ├── descargar.py
    │   ├── descargar_datos_para_modelo_finrl.py
    │   ├── getdata.py
    │   ├── L1_predictions.json
    │   ├── main.py
    │   ├── performance_analyzer.py
    │   ├── report_generator.py
    │   └── temp_test.py
    ├── comms/
    │   ├── config.py
    │   ├── data_validation.py
    │   ├── message_bus.py
    │   └── schemas.py
    ├── config/
    │   └── data_feed.py
    ├── configs/
    │   ├── config_backtest.yaml
    │   └── config_live.yaml
    ├── core/
    │   ├── __init__.py
    │   ├── async_balance_helper.py
    │   ├── async_processor.py
    │   ├── config.py
    │   ├── configuration_manager.py
    │   ├── convergence_config.py
    │   ├── core_utils.py
    │   ├── correlation_position_sizer.py
    │   ├── cycle_metrics.py
    │   ├── data_validation.py
    │   ├── data_validator.py
    │   ├── error_handler.py
    │   ├── exceptions.py
    │   ├── exchange_adapter.py
    │   ├── feature_engineering.py
    │   ├── hrm.py
    │   ├── incremental_signal_verifier.py
    │   ├── l3_processor.py
    │   ├── logger.py
    │   ├── logging.py
    │   ├── memory_manager.py
    │   ├── model_factory.py
    │   ├── paper_exchange_adapter.py
    │   ├── persistent_logger.py
    │   ├── portfolio_rebalancer.py
    │   ├── position_rotator.py
    │   ├── scheduler.py
    │   ├── selling_strategy.py
    │   ├── signal_hierarchy.py
    │   ├── simulated_exchange_client.py
    │   ├── state_manager.py
    │   ├── technical_indicators.py
    │   ├── technical_indicators.py.backup.20260202_021322
    │   ├── test_portfolio_rebalancer.py
    │   ├── trading_metrics.py
    │   ├── unified_validation.py
    │   ├── weight_calculator.py
    │   ├── weight_calculator_config.json
    │   ├── weight_calculator_config.py
    │   └── config/
    │       └── convergence_config.json
    ├── data/
    │   ├── __init__.py
    │   └── loaders.py
    ├── docs/
    │   ├── ASYNC_BALANCE_FIX_SUMMARY.md
    │   ├── AUTOLEARNING_ANALYSIS_AND_PLAN.md
    │   ├── CHANGELOG.md
    │   ├── fix_zero_balances_summary.md
    │   ├── INFORME_SISTEMA_LIMPIEZA.md
    │   ├── LISTADO_9_IAS_Y_MODELOS.md
    │   ├── PATCH_README.md
    │   └── Sistema_Normalizado.md
    ├── hacienda/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── demo_tax_system.py
    │   ├── posiciones_fifo.json
    │   ├── tax_tracker.py
    │   ├── tax_utils.py
    │   └── test_tax_tracker.py
    ├── l1_operational/
    │   ├── __init__.py
    │   ├── ai_pipeline.py
    │   ├── binance_client.py
    │   ├── binance_client.py.backup
    │   ├── binance_client.py.backup2
    │   ├── binance_client.py.backup3
    │   ├── bus_adapter.py
    │   ├── config.py
    │   ├── config.py.backup
    │   ├── data_feed.py
    │   ├── executor.py
    │   ├── genera_dataset_modelo1.py
    │   ├── l1_operational.py
    │   ├── metrics.py
    │   ├── mock_market_data.py
    │   ├── models.py
    │   ├── order_executors.py
    │   ├── order_intent_builder.py
    │   ├── order_manager.py
    │   ├── order_validators.py
    │   ├── portfolio.py
    │   ├── position_manager.py
    │   ├── realtime_loader.py
    │   ├── requirements.txt
    │   ├── risk_guard.py
    │   ├── signal_processor.py
    │   ├── simulated_exchange_client.py
    │   ├── smart_cooldown_manager.py
    │   ├── test_clean_l1.py
    │   ├── trend_ai.py
    │   └── enums/
    │       └── __init__.py
    ├── l2_tactic/
    │   ├── __init__.py
    │   ├── ai_model_integration.py
    │   ├── btc_eth_synchronizer.py
    │   ├── bus_integration.py
    │   ├── config.py
    │   ├── deepseek_config.py
    │   ├── feature_extractors.py
    │   ├── finrl_integration.py
    │   ├── finrl_processor.py
    │   ├── finrl_sb3_integration.py
    │   ├── finrl_wrapper.py
    │   ├── l2_utils.py
    │   ├── main_processor.py
    │   ├── metrics.py
    │   ├── model_loaders.py
    │   ├── models.py
    │   ├── observation_builders.py
    │   ├── path_mode_generator.py
    │   ├── path_modes.py
    │   ├── performance_optimizer.py
    │   ├── performance_optimizer.py.backup
    │   ├── position_sizer.py
    │   ├── procesar_l2.py
    │   ├── requeriments.txt
    │   ├── risk_overlay.py
    │   ├── safe_model_loader.py
    │   ├── signal_components.py
    │   ├── signal_generator_refactored.py
    │   ├── signal_generators.py
    │   ├── signal_validator.py
    │   ├── similarity_detector.py
    │   ├── test_grok_models.py
    │   ├── test_tight_range_handler.py
    │   ├── tight_range_handler.py
    │   ├── todos_restantes_claude.md
    │   ├── weight_calculator_integration.py
    │   ├── ensemble/
    │   │   ├── __init__.py
    │   │   ├── blender.py
    │   │   └── voting.py
    │   ├── generators/
    │   │   ├── mean_reversion.py
    │   │   └── technical_analyzer.py
    │   ├── indicators/
    │   │   └── technical.py
    │   ├── risk_controls/
    │   │   ├── __init__.py
    │   │   ├── alerts.py
    │   │   ├── manager.py
    │   │   ├── portfolio.py
    │   │   ├── positions.py
    │   │   └── stop_losses.py
    │   ├── technical/
    │   │   ├── __init__.py
    │   │   ├── multi_timeframe.py
    │   │   ├── patterns.py
    │   │   └── support_resistance.py
    │   └── tests/
    │       ├── __init__.py
    │       ├── conftest.py
    │       ├── test_integration.py
    │       ├── test_metrics.py
    │       ├── test_position_sizer.py
    │       ├── test_risk_control.py
    │       ├── test_signal_generator.py
    │       └── test_stop_loss.py
    ├── l3_strategy/
    │   ├── __init__.py
    │   ├── bus_integration.py
    │   ├── config.py
    │   ├── data_fetcher.py
    │   ├── data_provider.py
    │   ├── exposure_manager.py
    │   ├── filters.py
    │   ├── hrm_bl.py
    │   ├── l1_processor.py
    │   ├── l2_processor.py
    │   ├── l3_aggregator.py
    │   ├── l3_inference_pipeline.py
    │   ├── l3_logger.py
    │   ├── l3_utils.py
    │   ├── models.py
    │   ├── procesar_l3.py
    │   ├── range_detector.py
    │   ├── regime_classifier.py
    │   ├── regime_features.py
    │   ├── regime_specific_models.py
    │   ├── risk_manager.py
    │   ├── run_pipeline.py
    │   ├── sentiment_inference.py
    │   ├── test_regime_classifier.py
    │   ├── universe_filter.py
    │   └── volatility_inference_pipeline.py
    ├── ml_training/
    │   ├── modelo1_train_lgbm_modelo1.py
    │   ├── modelo1_train_logreg_modelo1.py
    │   ├── modelo1_train_rf_modelo1.py
    │   ├── train_grok_ultra_optimized.py
    │   ├── train_lgbm_modelo3.py
    │   ├── train_modelo_3_claude.py
    │   ├── train_rf_modelo2.py
    │   └── L3/
    │       ├── combinar_data_sentimel.py
    │       ├── download_portfolio_data.py
    │       ├── download_sentiment_data.py
    │       ├── download_volatility_data.py
    │       ├── kk.py
    │       ├── obtener_datos_regime_detection.py
    │       ├── train_portfolio_model.py
    │       ├── train_Regime_Detection.py
    │       ├── train_sentiment_model.py
    │       └── train_volatility_model.py
    ├── persistent_state/
    │   └── portfolio_state_live.json
    ├── public/
    │   └── index.html
    ├── scripts/
    │   ├── change_dns.ps1
    │   ├── disable_ipv6.ps1
    │   ├── full_test.py
    │   ├── integration_auto_learning.py
    │   ├── patch_portfolio_autolearning.py
    │   ├── start.ps1
    │   ├── sync_portfolios.py
    │   ├── analysis/
    │   │   └── emergency_analysis.py
    │   ├── checks/
    │   │   ├── check_9_layers_protection.py
    │   │   ├── check_autolearning_status.py
    │   │   ├── check_indicators.py
    │   │   └── check_logs.py
    │   └── debug/
    │       ├── debug_env.py
    │       └── debug_rebalance.py
    ├── sentiment/
    │   └── sentiment_manager.py
    ├── storage/
    │   ├── __init__.py
    │   ├── csv_writer.py
    │   └── paper_trade_logger.py
    ├── system/
    │   ├── __init__.py
    │   ├── auto_learning_bridge.py
    │   ├── bootstrap.py
    │   ├── component_extractor.py
    │   ├── config.py
    │   ├── error_recovery_manager.py
    │   ├── external_adapter.py
    │   ├── logging.py
    │   ├── market_data_manager.md
    │   ├── market_data_manager.py
    │   ├── models.py
    │   ├── orchestrator.py
    │   ├── state_coordinator.py
    │   ├── system_cleanup.py
    │   └── trading_pipeline_manager.py
    ├── tests/
    │   ├── backtester.py
    │   ├── final_fix_realtime_paper.py
    │   ├── fix_paper_mode.py
    │   ├── fix_realtime_data.py
    │   ├── fix_realtime_data.py.backup
    │   ├── fix_realtime_data_simple.py
    │   ├── fix_realtime_data_simple.py.backup
    │   ├── force_paper_mode.py
    │   ├── force_realtime_paper_mode.py
    │   ├── integration_auto_learning.py
    │   ├── integration_test.py
    │   ├── paper_trading_documentation.md
    │   ├── quick_log_test.py
    │   ├── README_AUTO_LEARNING.md
    │   ├── README_LIVE_TRADING.md
    │   ├── README_MODULARIZATION.md
    │   ├── README_SYSTEM_CLEANUP.md
    │   ├── readmeL1.md
    │   ├── readmeL2.md
    │   ├── readmeL3.md
    │   ├── security_checklist.md
    │   ├── security_validation.py
    │   ├── setup_testnet_credentials.py
    │   ├── simple_integration_test.py
    │   ├── system_cleanup.py
    │   ├── test_aggressive_mode.py
    │   ├── test_allocation_tiers.py
    │   ├── test_assertion_mechanism.py
    │   ├── test_bert_cache_system.py
    │   ├── test_blind_mode_handling.py
    │   ├── test_btc_eth_synchronization.py
    │   ├── test_confidence_normalization.py
    │   ├── test_configuration_manager.py
    │   ├── test_convergence_comprehensive.py
    │   ├── test_convergence_flags.py
    │   ├── test_convergence_integration.py
    │   ├── test_convergence_sizing.py
    │   ├── test_error_handler.py
    │   ├── test_error_recovery_manager.py
    │   ├── test_exceptional_override.py
    │   ├── test_finrl_integration.py
    │   ├── test_fix.py
    │   ├── test_fixes.py
    │   ├── test_hold_signals.py
    │   ├── test_initial_deployment.py
    │   ├── test_l1_models.py
    │   ├── test_l2_l3_fix.py
    │   ├── test_l2_l3_setup_fix.py
    │   ├── test_l3_authority.py
    │   ├── test_l3_confidence_preservation.py
    │   ├── test_l3_models.py
    │   ├── test_l3_regime_models.py
    │   ├── test_l3_strategic_control.py
    │   ├── test_l3_strategic_override.py
    │   ├── test_l3_unified.py
    │   ├── test_logging.py
    │   ├── test_market_data_manager.py
    │   ├── test_override_changes.py
    │   ├── test_paper_trades.py
    │   ├── test_path_mode_validation.py
    │   ├── test_path_modes.py
    │   ├── test_portfolio_fix.py
    │   ├── test_portfolio_persistence.py
    │   ├── test_portfolio_simple.py
    │   ├── test_portfolio_sync.py
    │   ├── test_portfolio_unified.py
    │   ├── test_position_size_cli_helper.py
    │   ├── test_profit_taking.py
    │   ├── test_profitability_fixes.py
    │   ├── test_protection_mechanism.py
    │   ├── test_risk_adjusted_portfolio.py
    │   ├── test_risk_adjusted_sizing.py
    │   ├── test_safety_features.py
    │   ├── test_selling_strategy.py
    │   ├── test_sentiment.py
    │   ├── test_signal_processing.py
    │   ├── test_similarity_detector.py
    │   ├── test_simulated_client.py
    │   ├── test_singleton_fix.py
    │   ├── test_state_manager_cycle_stats.py
    │   ├── test_stop_loss_validation.py
    │   ├── test_system_direct.py
    │   ├── test_tactical_sell.py
    │   ├── test_transition_state_removal.py
    │   ├── test_trend_following.py
    │   ├── test_trending_fix.py
    │   ├── test_unified_validation.py
    │   ├── test_weight_calculator.py
    │   ├── testnet_setup_instructions.md
    │   ├── todo_list.txt
    │   ├── validate_testnet_config.py
    │   ├── verify_paper_mode_status.py
    │   └── verify_realtime_functionality.py
    └── utils/
        ├── __init__.py
        ├── paper_trading_fix.py
        ├── position_size_cli_helper.py
        └── safe_indicators.py
