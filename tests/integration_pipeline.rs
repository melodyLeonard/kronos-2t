use kronos_2t::{
    model::engine::{KronosConfig, KronosEngine},
    storage::k2t_format::{encode_expert_page, K2tExpertPageView, K2tHeader, K2T_PAGE_SIZE},
};

#[test]
fn test_k2t_corruption_detection_header_and_expert_pages() {
    let mut raw_expert = encode_expert_page(3, 42, 256, 128, 0.05);
    {
        let view = K2tExpertPageView::from_page_slice(&raw_expert).expect("valid page");
        assert!(view.verify_checksum());
        assert_eq!(view.layer_id, 3);
        assert_eq!(view.expert_id, 42);
    }

    // Flip 1 bit in a Kronecker seed byte -> FNV-1a checksum must catch corruption
    raw_expert[64] ^= 0x01;
    let corrupted_view = K2tExpertPageView::from_page_slice(&raw_expert).expect("parses structure");
    assert!(
        !corrupted_view.verify_checksum(),
        "corrupted seed byte must fail FNV-1a integrity verification"
    );

    // Corrupt header magic must fail decode
    let bad_header = [0u8; K2T_PAGE_SIZE];
    assert!(K2tHeader::decode_page(&bad_header).is_err());
}

#[test]
fn test_full_concept_end_to_end_pipeline_invariants() {
    let k2t_path = std::env::temp_dir().join("kronos_2t_integration_suite.k2t");
    let cfg = KronosConfig {
        vocab_size: 1024,
        num_layers: 12,
        d_model: 256,
        d_ff: 128,
        num_heads: 8,
        kronecker_rank: 2,
        router_sub_keys: 500, // 250,000 virtual micro-experts per layer
        physical_experts_per_layer: 64,
        top_k_experts: 4,
        lookahead_layers: 6,
        lookahead_rank: 12,
        ram_ring_buffer_pages: 256,
        residual_depth_scale: None,
        seed: 20_261_025,
    };

    // > 250,000 * 12 * (3 * 256 * 128) = > 294 Billion virtual addressable parameters in test config
    assert!(cfg.virtual_total_parameters() > 290_000_000_000);

    let mut engine = KronosEngine::build_and_open(cfg, &k2t_path).expect("engine build failed");
    let initial_ssd_bytes = engine.total_ssd_state_bytes();

    // Run 8-token generation
    let prompt = [1, 19, 84, 512];
    let short_run = engine.generate(&prompt, 8).expect("short run failed");
    assert_eq!(short_run.ssd_context_bytes, initial_ssd_bytes);

    // Continue generating 250 more tokens without resetting -> context memory must stay strictly O(1)
    let long_run = engine.generate(&[short_run.generated_tokens[7]], 250).expect("long run failed");
    assert_eq!(long_run.ssd_context_bytes, initial_ssd_bytes);
    assert!(
        long_run.equivalent_fp16_kv_bytes > long_run.ssd_context_bytes,
        "KV cache grows linearly while SSD state stays fixed"
    );
    assert!(
        long_run.pager_hit_rate_percent > 95.0,
        "6-layer manifold-aligned lookahead should exceed 95% hit rate, got {:.1}%",
        long_run.pager_hit_rate_percent
    );

    // Verify all logits are finite and non-constant
    let logits = engine.forward_token(42).expect("single token forward failed");
    assert!(logits.iter().all(|v| v.is_finite()));
    let min_l = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_l > min_l, "logits must have non-zero variance");

    let _ = std::fs::remove_file(&k2t_path);
}
